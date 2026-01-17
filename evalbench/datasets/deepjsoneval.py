import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from evalbench.types import EvalItem

DEFAULT_PROMPT = (
    "You are given natural language text and a JSON Schema. Extract the structured "
    "information from the text and return a single JSON object that strictly conforms "
    "to the schema. Do not include Markdown fences or explanations.\n\n"
    "Text:\n{text}\n\nSchema:\n{schema}"
)


def _load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _load_json(path: Path) -> Sequence[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError(f"Expected list or {{'data': [...]}} in {path}")


def _coerce_schema(schema_value: Any) -> Dict[str, Any]:
    if isinstance(schema_value, dict):
        return schema_value
    if isinstance(schema_value, str):
        return json.loads(schema_value)
    raise ValueError("Schema must be a dict or JSON string")


def load_deepjsoneval(
    data_path: str,
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 42,
) -> List[EvalItem]:
    """
    Load DeepJSONEval examples from a local JSONL/JSON file. Each record should contain
    fields: schema (JSON schema or string), text (source passage), json (ground truth
    object), category, and true_depth.
    """

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"DeepJSONEval file not found at {path}")

    if path.suffix.lower() == ".jsonl":
        rows = list(_load_jsonl(path))
    elif path.suffix.lower() == ".json":
        rows = list(_load_json(path))
    else:
        raise ValueError("DeepJSONEval expects a .jsonl or .json file")

    if sample is not None and sample < len(rows):
        random.seed(seed)
        rows = random.sample(rows, sample)
    if limit is not None and limit < len(rows):
        rows = rows[:limit]

    items: List[EvalItem] = []
    for idx, row in enumerate(rows):
        schema = _coerce_schema(row["schema"])
        text = row["text"]
        prompt = DEFAULT_PROMPT.format(text=text, schema=json.dumps(schema, indent=2))
        items.append(
            EvalItem(
                id=row.get("id") or f"deepjsoneval:{idx}",
                prompt=prompt,
                schema=schema,
                verify_schema=schema,
                extra={
                    "dataset": "deepjsoneval",
                    "category": row.get("category"),
                    "true_depth": row.get("true_depth"),
                    "reference_json": row.get("json"),
                },
            )
        )
    return items
