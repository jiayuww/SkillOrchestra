# Environment Setup

All environment creation and setup scripts live here.

| Script | Purpose |
|--------|---------|
| `run.sh` | Unified entry: run all or individual setups |
| `env.sh` | Create `so_env` (pipeline) and `sglang_env` (SGLang) |
| `retriever.sh` | Create `retriever` env (Qwen3-Embedding, FAISS) |
| `agent_orchestration.sh` | Full agent eval setup (FRAMES, HLE) |

## Quick start

```bash
# All environments
./scripts/setup/run.sh

# Individual
./scripts/setup/env.sh --sglang      # SGLang only
./scripts/setup/retriever.sh         # Retriever only
./scripts/setup/agent_orchestration.sh --setup-envs
```

## After setup: serve scripts

- **Model Routing:** `./scripts/serve/serve_routing.sh`
- **Agent Orchestration:** `./scripts/serve/serve_orchestration.sh`