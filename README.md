# SkillOrchestra

This repository contains the code for [SkillOrchestra](https://arxiv.org/abs/2602.19672), a framework for skill-aware orchestration of compound AI systems. Instead of learning a routing policy end-to-end, SkillOrchestra learns fine-grained skills from execution experience and models agent-specific competence and cost under those skills. At deployment, the orchestrator infers the skill demands of the current interaction and selects agents that best satisfy them under an explicit performance-cost trade-off.

---

## Setup

### 1. Environment

```bash
# All environments (pipeline + SGLang + retriever)
./scripts/setup/run.sh

# Or individually:
./scripts/setup/env.sh --pipeline   # so_env: pipeline, exploration, learning, selection, testing
./scripts/setup/env.sh --sglang     # sglang_env: model serving via SGLang
./scripts/setup/retriever.sh        # retriever: Qwen3-Embedding, FAISS
```

### 2. Configuration

- Copy `.env.example` to `.env` and fill in your values:
  ```bash
  cp .env.example .env
  ```
- Set at minimum:
  - `OPENAI_API_KEY` or `OPENAI_GATEWAY_KEY` — for LLM calls during learning
  - `INDEX_DIR` — for FRAMES: path to retrieval index (e.g. `wiki.index`, `wiki.jsonl`)
  - `HF_HOME` — optional, Hugging Face cache

---

## Serving Models

Models must be running before the pipeline can explore, select, or test.

### Model Routing

```bash
conda activate sglang_env
./scripts/serve/serve_routing.sh
```

### Agent Orchestration

```bash
conda activate sglang_env
./scripts/serve/serve_orchestration.sh # retriever use a different retriever environment
```

Ensure `INDEX_DIR` is set for the retriever. Edit `config/eval_config.json` to match your model host/ports if not using localhost.

---

## Running the Full Pipeline

Activate the pipeline environment and run:

```bash
conda activate so_env
```

### Model Routing

```bash
# Full pipeline: explore → learn → select → test
python scripts/pipeline.py model-routing \
    --dataset nq_validation_qwen \
    --output-dir output/nq \
    --test-dataset nq_test_qwen \
    --phases explore,learn,select,test \
    --exploration-samples 40 \
    --train-samples 20 \
    --val-samples 20

# Use existing exploration data (skip explore)
python scripts/pipeline.py model-routing \
    --dataset nq_validation_qwen \
    --exploration-data /path/to/inference_results.jsonl \
    --output-dir output/nq \
    --phases learn,select,test \
    --test-dataset nq_test_qwen
```

### Agent Orchestration

```bash
# Full pipeline
python scripts/pipeline.py frames \
    --output-dir output/frames \
    --eval-script orchestration/eval_frames.py \
    --test-samples data/frames_test.jsonl \
    --exploration-samples 40 \
    --train-samples 20 \
    --val-samples 20
```

### Phases

- `explore` — Run all pool models on the dataset; collect execution traces
- `learn` — Learn a skill handbook from traces (skills, competence, cost) with refinement
- `select` — Generate candidate handbooks and select Pareto-optimal with live validation
- `test` — Evaluate the selected handbook on the test set

Use `--phases explore`, `--phases learn,select`, etc. to run subsets.

---

## Citation

If you find this work helpful, please consider giving a ⭐ and citing our paper 😊

```bibtex
@misc{wang2026skillorchestra,
      title={SkillOrchestra: Learning to Route Agents via Skill Transfer}, 
      author={Jiayu Wang and Yifei Ming and Zixuan Ke and Shafiq Joty and Aws Albarghouthi and Frederic Sala},
      year={2026},
      eprint={2602.19672},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.19672}, 
}
```
