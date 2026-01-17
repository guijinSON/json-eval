# JSON Schema Evaluation Harness

Run JSON-schema generation evaluations against **JSONSchemaBench** and **SchemaBench** with either **vLLM** or **LiteLLM** backends.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # LiteLLM + core deps
pip install -r requirements-vllm.txt     # add vLLM support (optional)
```

## Datasets

### JSONSchemaBench
- Default: streamed from Hugging Face `epfl-dlab/JSONSchemaBench`.
- Optional local copy: clone [`guidance-ai/jsonschemabench`](https://github.com/guidance-ai/jsonschemabench/tree/main) and point `--jsonschemabench-root` at its `data/` folder.

### SchemaBench
- Download the released data from Google Drive or Tsinghua Cloud listed in the [Schema Reinforcement Learning repo](https://github.com/thunlp/SchemaReinforcementLearning/tree/main/schemabench) and copy into `data/schemabench/` so it matches:
  ```
  data/schemabench/
    custom/test/*.json
    schema/test/*.json
    custom_append.jsonl
    translation_test.jsonl
    bench/words.txt        # optional; used for custom subset prompts
  ```
- `--subset-size` controls the sampled size for the custom/escape subsets (default 100).

### DeepJSONEval
- Download the DeepJSONEval benchmark JSONL (or JSON) file from the upstream project [`GTS-AI-Infra-Lab-SotaS/DeepJSONEval`](https://github.com/GTS-AI-Infra-Lab-SotaS/DeepJSONEval). Place it at `data/deepjsoneval/deepjsoneval.jsonl` (or point `--deepjson-path` to your copy).
- Each record should include `schema`, `text`, `json` (ground truth), `category`, and `true_depth`.
- Use `--sample` or `--limit` to reduce the evaluated subset if desired.

## Running evaluations

### LiteLLM (OpenAI-compatible endpoints)
```bash
python -m evalbench.evaluate \
  --backend litellm \
  --model gpt-4o-mini \
  --api-key $OPENAI_API_KEY \
  --dataset jsonschemabench \
  --limit 20 \
  --output runs/jsonschemabench_gpt4o.jsonl
```

### vLLM (local models)
```bash
python -m evalbench.evaluate \
  --backend vllm \
  --model /path/to/model/or/hf-repo \
  --dataset schemabench-all \
  --schemabench-root ./data/schemabench \
  --limit 50 \
  --max-tokens 256 \
  --output runs/schemabench_vllm.jsonl
```

### DeepJSONEval example
```bash
python -m evalbench.evaluate \
  --backend litellm \
  --model gpt-4o-mini \
  --dataset deepjsoneval \
  --deepjson-path ./data/deepjsoneval/deepjsoneval.jsonl \
  --limit 25 \
  --output runs/deepjsoneval_gpt4o.jsonl
```

### Common flags
- `--dataset`: `jsonschemabench`, `deepjsoneval`, `schemabench-complex`, `schemabench-custom`, `schemabench-escape`, or `schemabench-all`.
- `--limit`: cap number of evaluated items.
- `--sample`: random sample for JSONSchemaBench HF data or DeepJSONEval.
- `--subset-size`: sampling size for SchemaBench custom/escape tasks.
- `--temperature`: sampling temperature passed through to LiteLLM or vLLM (default 0.0).
- `--stop`: optional stop sequences (space-separated).
- Results are written to `--output` (JSONL) plus a sibling `*.metrics.json` summary.

## What gets measured
- Parses model output to JSON when possible (removes fences, extracts first JSON block).
- Validates against the provided schema using `jsonschema.validate`.
- Reports:
  - `valid_json_rate`: fraction of outputs that parsed as JSON.
  - `valid_schema_rate`: fraction that both parsed and validated.
  - Per-example outputs and errors saved in the JSONL file.
