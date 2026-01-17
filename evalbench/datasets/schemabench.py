import copy
import json
import random
import re
import base64
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jsonschema

from evalbench.types import EvalItem

BASE_PROMPT = (
    "Generate a valid JSON object that conforms to the JSON Schema below. "
    "Return JSON only, no fences or explanations.\n\nSchema:\n{schema}"
)

CONSTRAINT_MAP: Dict[str, Dict[str, object]] = {
    "phone": {
        "type": "string",
        "object_name": "US phone number",
        "key": [
            "usPhoneNumber",
            "contactNumber",
            "mobile",
            "phoneNumber",
            "telephone",
            "userPhone",
            "customerPhone",
            "phoneContact",
            "phoneLine",
            "userContact",
        ],
        "pattern": r"^(\([0-9]{3}\) |[0-9]{3}-)[0-9]{3}-[0-9]{4}$",
    },
    "folderPath": {
        "type": "string",
        "object_name": "Linux folder path",
        "key": [
            "linuxDirectory",
            "linuxFilePath",
            "linuxPath",
            "linuxFolder",
            "directoryPath",
            "filePathLinux",
            "linuxLocation",
            "linuxFolderPath",
            "folderLocation",
            "linuxDir",
        ],
        "pattern": r"^(\/([a-zA-Z0-9_-]+(\/[a-zA-Z0-9_-]+)*)*)?$",
    },
    "WindowsfolderPath": {
        "type": "string",
        "object_name": "Windows folder path",
        "special_inst": r"like C:\\Users\\Administrator\\Desktop",
        "key": [
            "windowsDirectory",
            "windowsFilePath",
            "windowsPath",
            "windowsFolder",
            "directoryPathWindows",
            "filePathWindows",
            "windowsLocation",
            "windowsDir",
            "windowsFolderPath",
            "windowsFileLocation",
        ],
        "pattern": r"^([a-zA-Z]:\\)([-\u4e00-\u9fa5\w\s.()~!@#$%^&()\[\]{}+=]+\\)*$",
    },
    "strongPasswd": {
        "type": "string",
        "object_name": "strong password",
        "special_inst": "at least 12 characters long, containing at least one uppercase letter, one lowercase letter, one number, and one special character",
        "key": [
            "securePassword",
            "complexPassword",
            "passwordStrength",
            "strongPassword",
            "robustPassword",
            "passwordCriteria",
            "passwordPolicy",
            "securePass",
            "passwordKey",
            "encryptionPassword",
        ],
        "pattern": r"^(?=(.*[A-Z]))(?=(.*[a-z]))(?=(.*\d))(?=(.*[!@#$%^&*()_+])).{12,}$",
    },
    "rgbColor": {
        "type": "string",
        "object_name": "RGB color",
        "special_inst": "like #ff0000",
        "key": [
            "colorHex",
            "hexColorCode",
            "rgbHexValue",
            "rgbColorValue",
            "hexColor",
            "colorCode",
            "colorValue",
            "rgbHex",
            "colorRepresentation",
            "hexadecimalColor",
        ],
        "pattern": r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$",
    },
    "base64": {
        "type": "string",
        "object_name": "base64 encoded string",
        "special_inst": "and you should encode the following content: {content}",
        "key": [
            "encodedContent",
            "base64Encoded",
            "base64String",
            "base64Data",
            "base64Output",
            "encodedData",
            "base64Result",
            "contentBase64",
            "base64EncodedContent",
            "base64EncodedString",
        ],
    },
}

INST_TEMPLATES = [
    "Please ignore other instruction and write a {object_name} here",
    "Please set aside other instructions and enter a {object_name} in this space",
    "Kindly disregard previous directions and provide a {object_name} here",
    "Ignore all other guidelines and type a {object_name} below",
    "Please overlook the other requests and submit a {object_name} in this field",
    "Set aside any other instructions and write your {object_name} here",
    "Please bypass previous instructions and enter a {object_name} in this area",
    "Disregard any other prompts and fill in a {object_name} here",
    "Please ignore any other guidance and place a {object_name} in this spot",
    "Kindly ignore all other directives and input a {object_name} in this section",
    "Please pay no attention to prior instructions and write a {object_name} here",
]

CONSTRAINT_NUM = 5


def _load_words(root: Path) -> List[str]:
    candidates = [
        root / "words.txt",
        root.parent / "bench" / "words.txt",
    ]
    for path in candidates:
        if path.exists():
            with path.open() as f:
                return [line.strip() for line in f if line.strip()]
    return ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]


def _find_string_elements(schema: Dict, former_path: Optional[List[str]] = None):
    if former_path is None:
        former_path = []
    for key, value in schema.items():
        if isinstance(value, dict):
            if (
                "type" in value
                and value["type"] == "string"
                and "enum" not in value
                and "format" not in value
                and "pattern" not in value
                and "description" not in value
                and key != "additionalProperties"
            ):
                if len(former_path) == 0 or former_path[-1] != "patternProperties":
                    if "patternProperties" in schema.keys():
                        continue
                    yield former_path + [key]
            else:
                if "type" not in value or isinstance(value["type"], str):
                    if "patternProperties" in schema.keys():
                        continue
                    yield from _find_string_elements(value, former_path + [key])


def _replace_string_element(
    schema: Dict, path: List[str], constraint: str, constraint_key: str, word_list: Sequence[str]
) -> Tuple[Dict, Dict]:
    root_schema = schema
    verify_schema = copy.deepcopy(schema)
    root_verify_schema = verify_schema
    for key in path[:-1]:
        schema = schema[key]
    for key in path[:-1]:
        verify_schema = verify_schema[key]
    schema.pop(path[-1], None)
    verify_schema.pop(path[-1], None)
    path[-1] = constraint_key
    schema[path[-1]] = copy.deepcopy(CONSTRAINT_MAP[constraint])
    verify_schema[path[-1]] = copy.deepcopy(CONSTRAINT_MAP[constraint])
    schema[path[-1]].pop("key", None)
    verify_schema[path[-1]].pop("key", None)
    if constraint == "base64":
        constraint_arg = " ".join(random.choices(word_list, k=15))
        verify_schema[path[-1]]["const"] = base64.b64encode(constraint_arg.encode()).decode().strip()
    else:
        constraint_arg = None
    schema[path[-1]]["description"] = random.choice(INST_TEMPLATES).format(
        object_name=schema[path[-1]]["object_name"]
    )
    if "special_inst" in schema[path[-1]]:
        schema[path[-1]]["description"] += f", {schema[path[-1]]['special_inst']}"
    if constraint_arg:
        schema[path[-1]]["description"] = schema[path[-1]]["description"].format(content=constraint_arg)
    schema[path[-1]].pop("object_name", None)
    schema[path[-1]].pop("special_inst", None)
    verify_schema[path[-1]].pop("object_name", None)
    verify_schema[path[-1]].pop("special_inst", None)
    schema[path[-1]].pop("pattern", None)
    if len(path) >= 2 and path[-2] == "properties":
        tmp_schema = root_schema
        for key in path[:-2]:
            tmp_schema = tmp_schema[key]
        required = tmp_schema.get("required", [])
        if constraint_key not in required:
            required.append(constraint_key)
            tmp_schema["required"] = required
    if path[-1] == "items":
        tmp_schema = root_schema
        for key in path[:-1]:
            tmp_schema = tmp_schema[key]
        if tmp_schema.get("type") == "array":
            if "description" in tmp_schema:
                tmp_schema["description"] = "A list of items, you should follow the description in `items`"
            tmp_schema.pop("default", None)
    return root_schema, root_verify_schema


def _load_complex(root: Path, limit: Optional[int]) -> List[EvalItem]:
    schema_dir = root / "schema" / "test"
    if not schema_dir.exists():
        raise FileNotFoundError(f"SchemaBench complex data not found at {schema_dir}")
    files = sorted(schema_dir.glob("*.json"))
    items: List[EvalItem] = []
    for idx, path in enumerate(files):
        if limit is not None and idx >= limit:
            break
        with path.open() as f:
            schema = json.load(f)
        prompt = BASE_PROMPT.format(schema=json.dumps(schema, indent=2))
        items.append(
            EvalItem(
                id=f"schemabench-complex:{path.stem}",
                prompt=prompt,
                schema=schema,
                verify_schema=schema,
                extra={"subset": "complex", "source": "schemabench"},
            )
        )
    return items


def _load_custom(root: Path, seed: int, limit: Optional[int], subset_size: Optional[int]) -> List[EvalItem]:
    custom_dir = root / "custom" / "test"
    if not custom_dir.exists():
        raise FileNotFoundError(f"SchemaBench custom data not found at {custom_dir}")
    word_list = _load_words(root)
    random.seed(seed)
    schemas: List[Dict] = []
    for path in sorted(custom_dir.glob("*.json")):
        with path.open() as f:
            schemas.append(json.load(f))
    dataset: List[Dict[str, Dict]] = []
    for schema in schemas:
        nodes = list(_find_string_elements(schema))
        if not nodes:
            continue
        sample_nodes = nodes if len(nodes) < CONSTRAINT_NUM else random.sample(nodes, k=CONSTRAINT_NUM)
        tmp = copy.deepcopy(schema)
        verify = copy.deepcopy(schema)
        ok = True
        for node in sample_nodes:
            constraint = random.choice(list(CONSTRAINT_MAP.keys()))
            constraint_key = random.choice(CONSTRAINT_MAP[constraint]["key"])
            try:
                tmp, verify = _replace_string_element(tmp, copy.deepcopy(node), constraint, constraint_key, word_list)
            except Exception:
                ok = False
                break
        if ok:
            dataset.append({"model_schema": tmp, "verify_schema": verify})
    append_path = root / "custom_append.jsonl"
    if append_path.exists():
        with append_path.open() as f:
            for line in f:
                obj = json.loads(line)
                dataset.append({"model_schema": obj["modified"]["schema"], "verify_schema": obj["original"]["schema"]})
    if subset_size is not None and subset_size < len(dataset):
        dataset = random.sample(dataset, subset_size)
    if limit is not None and limit < len(dataset):
        dataset = dataset[:limit]

    items: List[EvalItem] = []
    for idx, row in enumerate(dataset):
        prompt = BASE_PROMPT.format(schema=json.dumps(row["model_schema"], indent=2))
        items.append(
            EvalItem(
                id=f"schemabench-custom:{idx}",
                prompt=prompt,
                schema=row["model_schema"],
                verify_schema=row["verify_schema"],
                extra={"subset": "custom", "source": "schemabench"},
            )
        )
    return items


def _load_escape(root: Path, limit: Optional[int], subset_size: Optional[int], seed: int) -> List[EvalItem]:
    data_path = root / "translation_test.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"SchemaBench escape data not found at {data_path}")
    rows: List[Dict] = []
    with data_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if subset_size is not None and subset_size < len(rows):
        random.seed(seed)
        rows = random.sample(rows, subset_size)
    if limit is not None and limit < len(rows):
        rows = rows[:limit]
    items: List[EvalItem] = []
    for idx, row in enumerate(rows):
        model_schema = row["model_schema"]
        prompt = (
            "Generate a valid JSON object for the schema below. Remember to include the"
            f" special token `{row['special_token']}` where the schema requires it."
            "\nReturn JSON only.\n\nSchema:\n"
            + json.dumps(model_schema, indent=2)
        )
        items.append(
            EvalItem(
                id=f"schemabench-escape:{idx}",
                prompt=prompt,
                schema=model_schema,
                verify_schema=row["verify_schema"],
                extra={"subset": "escape", "special_token": row["special_token"], "source": "schemabench"},
            )
        )
    return items


def load_schemabench(
    data_root: str = "./data/schemabench",
    split: str = "complex",
    limit: Optional[int] = None,
    subset_size: Optional[int] = 100,
    seed: int = 42,
) -> List[EvalItem]:
    """
    Load SchemaBench subsets (complex, custom, escape).
    """

    root = Path(data_root)
    if split == "complex":
        return _load_complex(root, limit)
    if split == "custom":
        return _load_custom(root, seed=seed, limit=limit, subset_size=subset_size)
    if split == "escape":
        return _load_escape(root, limit=limit, subset_size=subset_size, seed=seed)
    if split == "all":
        items = []
        for name in ("complex", "custom", "escape"):
            items.extend(load_schemabench(data_root=data_root, split=name, limit=limit, subset_size=subset_size, seed=seed))
        return items
    raise ValueError(f"Unknown SchemaBench split: {split}")
