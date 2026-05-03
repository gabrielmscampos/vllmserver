# vLLM Serving Runtime

A [KServe](https://kserve.github.io/website/)-based model server that serves vLLM-compatible transformer models with an OpenAI-compatible API.

## Requirements

- Python 3.12+
- CUDA-capable GPU
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
uv sync
```

## Usage

### Local

```bash
python -m vllmserver --model_dir ./hf-models/Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ
```

Load a model directly from the Hugging Face Hub:

```bash
python -m vllmserver --model_id Qwen/Qwen2-1.5B-Instruct
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | `/mnt/models` | Path to a local model directory |
| `--model_id` | — | Hugging Face model ID (overrides `--model_dir`) |
| `--model_revision` | — | Hugging Face model revision |
| `--max_model_len` | — | Maximum number of tokens the model can process |
| `--trust_remote_code` | `false` | Allow loading models with custom code |
| `--disable_log_requests` | `false` | Disable per-request logging |

All [vLLM engine arguments](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) (e.g. `--dtype`, `--tensor-parallel-size`, `--enable-auto-tool-choice`) are also accepted.

## API

The server listens on port `8080` and exposes an OpenAI-compatible API under `/openai/v1`.

| Endpoint | Description |
|---|---|
| `GET /v2/health/ready` | Readiness health check |
| `POST /openai/v1/chat/completions` | Chat completions (streaming supported) |
| `POST /openai/v1/completions` | Text completions (streaming supported) |
| `POST /openai/v1/embeddings` | Embeddings (requires an embedding model) |
| `POST /openai/v1/rerank` | Reranking (requires a reranking model) |

### Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/openai/v1", api_key="placeholder")

response = client.chat.completions.create(
    model="Qwen3.5-0.8B-W4A16-AutoRound-AWQ",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

The `model` field must match the model name reported by the server. By default this is the directory or Hub ID basename; override it with `--served-model-name`.

## Container image

The container image is built and uploaded to ghcr.io automatically whenever a tag is pushed to the repository. You can build it locally with:

```bash
docker build -t localhost/vllmserver:latest --build-arg BUILD_VERSION=0.0.0-local .
```

## Notes

- Some models (e.g. Qwen3.x) require up-to-date transformers: `pip install --upgrade transformers`
- Tensor parallelism is configured automatically based on the number of available GPUs
