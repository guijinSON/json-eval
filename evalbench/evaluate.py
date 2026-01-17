import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from tqdm import tqdm

from evalbench.datasets import load_jsonschemabench, load_schemabench
from evalbench.models import LiteLLMRunner, VLLMRunner
from evalbench.types import EvalItem


def _extract_json_candidate(text: str) -> Tuple[Optional[Any], Optional[str]]:
    """Try to recover a JSON object from the model output."""
    cleaned = text.strip()
    # strip code fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
    # attempt direct parse
    try:
        return json.loads(cleaned), None
    except Exception as exc:
        error = str(exc)
    # fall back to first JSON-looking block
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate), None
        except Exception as exc:
            return None, str(exc)
    return None, error


def _validate(obj: Any, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    try:
        jsonschema.validate(obj, schema)
        return True, None
    except Exception as exc:
        return False, str(exc)


def _load_dataset(args: argparse.Namespace) -> List[EvalItem]:
    if args.dataset == "jsonschemabench":
        return load_jsonschemabench(
            data_root=args.jsonschemabench_root,
            hf_split=args.hf_split,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
        )
    schemabench_split = args.dataset.replace("schemabench-", "")
    return load_schemabench(
        data_root=args.schemabench_root,
        split=schemabench_split,
        limit=args.limit,
        subset_size=args.subset_size,
        seed=args.seed,
    )


def _build_runner(args: argparse.Namespace):
    if args.backend == "litellm":
        return LiteLLMRunner(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            extra_kwargs={"temperature": args.temperature, "max_tokens": args.max_tokens},
        )
    if args.backend == "vllm":
        return VLLMRunner(model=args.model, max_tokens=args.max_tokens, stop=args.stop)
    raise ValueError(f"Unknown backend {args.backend}")


def run(args: argparse.Namespace):
    items = _load_dataset(args)
    runner = _build_runner(args)
    results = []
    valid_json = 0
    valid_schema = 0
    for item in tqdm(items, desc="Evaluating"):
        start = time.time()
        output = runner.generate(item.prompt, stop=args.stop)
        latency = time.time() - start
        parsed, parse_error = _extract_json_candidate(output)
        schema_ok = False
        schema_error = None
        if parsed is not None:
            valid_json += 1
            schema_ok, schema_error = _validate(parsed, item.verify_schema or item.schema)
            if schema_ok:
                valid_schema += 1
        results.append(
            {
                "id": item.id,
                "prompt": item.prompt,
                "output": output,
                "parsed": parsed,
                "parse_error": parse_error,
                "schema_ok": schema_ok,
                "schema_error": schema_error,
                "latency_sec": latency,
                "extra": item.extra,
            }
        )
    metrics = {
        "total": len(items),
        "valid_json": valid_json,
        "valid_schema": valid_schema,
        "valid_json_rate": valid_json / len(items) if items else 0.0,
        "valid_schema_rate": valid_schema / len(items) if items else 0.0,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
        summary_path = out_path.with_suffix(".metrics.json")
        summary_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LLMs on JSONSchemaBench and SchemaBench.")
    parser.add_argument(
        "--dataset",
        default="jsonschemabench",
        choices=[
            "jsonschemabench",
            "schemabench-complex",
            "schemabench-custom",
            "schemabench-escape",
            "schemabench-all",
        ],
    )
    parser.add_argument("--model", required=True, help="Model name for vLLM or LiteLLM.")
    parser.add_argument("--backend", default="litellm", choices=["litellm", "vllm"])
    parser.add_argument("--limit", type=int, default=None, help="Max items to evaluate.")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size for JSONSchemaBench.")
    parser.add_argument("--subset-size", type=int, default=100, help="Subset size for SchemaBench custom/escape.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jsonschemabench-root", default=None, help="Local data root; if omitted use HF dataset.")
    parser.add_argument("--hf-split", default="train", help="HF split name for JSONSchemaBench.")
    parser.add_argument("--schemabench-root", default="./data/schemabench", help="Local SchemaBench data root.")
    parser.add_argument("--output", default=None, help="Path to write JSONL outputs.")
    parser.add_argument("--api-key", default=None, help="API key for LiteLLM-compatible endpoints.")
    parser.add_argument("--base-url", default=None, help="Base URL for LiteLLM-compatible endpoints.")
    parser.add_argument("--stop", nargs="*", default=None, help="Optional stop sequences.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
