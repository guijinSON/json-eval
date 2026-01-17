import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from datasets import load_dataset

from evalbench.types import EvalItem

DEFAULT_PROMPT = (
    "You are given a JSON Schema. Return a single JSON object that is valid for "
    "this schema. Do not include Markdown fences or explanations.\n\nSchema:\n{schema}"
)


def _local_schema_files(root: Path) -> Iterable[Path]:
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        yield from sorted(dataset_dir.glob("*.json"))


def _items_from_files(files: Sequence[Path], limit: Optional[int]) -> List[EvalItem]:
    items: List[EvalItem] = []
    for idx, path in enumerate(files):
        if limit is not None and idx >= limit:
            break
        with path.open() as f:
            schema = json.load(f)
        dataset_name = path.parent.name
        item_id = f"{dataset_name}:{path.stem}"
        prompt = DEFAULT_PROMPT.format(schema=json.dumps(schema, indent=2))
        items.append(
            EvalItem(
                id=item_id,
                prompt=prompt,
                schema=schema,
                verify_schema=schema,
                extra={"dataset": dataset_name, "source": "local-jsonschemabench"},
            )
        )
    return items


def load_jsonschemabench(
    data_root: Optional[str] = None,
    hf_split: str = "train",
    limit: Optional[int] = None,
    sample: Optional[int] = None,
    seed: int = 42,
) -> List[EvalItem]:
    """
    Load JSONSchemaBench either from a local checkout (matching the upstream
    guidance-ai/jsonschemabench data layout) or via the Hugging Face dataset
    `epfl-dlab/JSONSchemaBench`.
    """

    if data_root:
        root = Path(data_root)
        files = list(_local_schema_files(root))
        items = _items_from_files(files, limit)
    else:
        dataset = load_dataset("epfl-dlab/JSONSchemaBench")[hf_split]
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        items = []
        for row in dataset:
            schema = row["json_schema"]
            uid = row.get("unique_id") or str(len(items))
            prompt = DEFAULT_PROMPT.format(schema=json.dumps(schema, indent=2))
            items.append(
                EvalItem(
                    id=f"jsonschemabench:{uid}",
                    prompt=prompt,
                    schema=schema,
                    verify_schema=schema,
                    extra={"dataset": "jsonschemabench", "source": "hf"},
                )
            )

    if sample is not None and sample < len(items):
        random.seed(seed)
        items = random.sample(items, sample)

    return items
