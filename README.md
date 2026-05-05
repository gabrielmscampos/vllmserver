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

## Hot-Reload

Hot-reload mode lets the server swap models on demand without restarting. When a request arrives for a model that isn't loaded, the server unloads the current model, loads the requested one, and then serves the request. Only one model is resident in GPU memory at a time.

Activate it with `--hot-reload-config` pointing to a YAML file that declares all available models:

```yaml
# config.yaml
models:
  - name: model1
    model_dir: ./hf-models/Vishva007/Qwen3.5-0.8B-W4A16-AutoRound-AWQ
    default: true
    vllm_args:
      reasoning_parser: qwen3
      enable_auto_tool_choice: true
      tool_call_parser: qwen3_coder
  - name: model2
    model_dir: ./hf-models/some-other-model
```

Start the server:

```bash
python -m vllmserver --hot-reload-config config.yaml
```

Any vLLM engine flag still applies:

```bash
python -m vllmserver --hot-reload-config config.yaml --dtype bfloat16 --enforce-eager
```

**Behavior:**
- The server boots on the `default: true` model.
- Sending `{"model": "model2", ...}` to any endpoint triggers a swap to `model2`. Subsequent requests for `model2` are served immediately.
- If a model fails to load, the server falls back to the default model automatically.
- The config file is re-read every 60 seconds. Newly added models become available for routing without a restart; removals are not applied until the next server start.
- All model directories are validated at startup — a missing path exits immediately.

### Swap Endpoint

Use `POST /swap` to trigger a model swap in the background. The call returns immediately — poll `GET /swap/status` to know when the model is ready.

```bash
# Trigger swap (returns immediately)
curl -X POST http://localhost:8080/swap \
  -H "Content-Type: application/json" \
  -d '{"model": "model2"}'
# 200 → {"status": "swapping", "model": "model2"}
# 404 → model name not in the registry
# 409 → a swap is already in progress

# Poll until ready
curl http://localhost:8080/swap/status
# while swapping → {"model": "model1", "ready": true, "swapping": true, "swapping_to": "model2", "error": null}
# when done      → {"model": "model2", "ready": true, "swapping": false, "swapping_to": null, "error": null}
# on failure     → {"model": "model1", "ready": true, "swapping": false, "swapping_to": null, "error": "..."}
```

Recommended workflow when switching models:
1. `POST /swap` → 200 (swap starts in background)
2. Poll `GET /swap/status` until `"swapping": false`
3. Check `"model"` matches the target and `"error"` is null
4. Update your load balancer / routing rules
5. Inference requests are served immediately with no swap delay

### Per-model vLLM Options

Each model entry in the config can include a `vllm_args` map to override vLLM engine options on a per-model basis. Keys use Python attribute names (underscores, not dashes):

```yaml
models:
  - name: qwen3-27b
    model_dir: ./hf-models/Qwen3.5-27B-AWQ
    default: true
    vllm_args:
      reasoning_parser: qwen3
      enable_auto_tool_choice: true
      tool_call_parser: qwen3_coder
  - name: bge-m3
    model_dir: ./hf-models/bge-m3
    # no vllm_args — uses CLI defaults
```

Any option not listed in `vllm_args` falls back to the value provided on the command line. CLI flags such as `--dtype bfloat16` still apply globally unless explicitly overridden per model. Settings never leak between models: each swap resets to the CLI baseline first, then applies the model's own overrides.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_dir` | `/mnt/models` | Path to a local model directory |
| `--model_id` | — | Hugging Face model ID (overrides `--model_dir`) |
| `--model_revision` | — | Hugging Face model revision |
| `--max_model_len` | — | Maximum number of tokens the model can process |
| `--trust_remote_code` | `false` | Allow loading models with custom code |
| `--disable_log_requests` | `false` | Disable per-request logging |
| `--hot-reload-config` | — | Path to a YAML config for on-demand model hot-reload |

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
| `POST /swap` | Trigger a background model swap (hot-reload mode only) |
| `GET /swap/status` | Swap progress and currently loaded model (hot-reload mode only) |

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

The image takes about ~30 GB of disk space after the build is complete.

```
IMAGE                               ID             DISK USAGE   CONTENT SIZE   EXTRA
gabrielmscampos/vllmserver:latest   29ab31479f3e       30.2GB         8.13GB
```

### Debugging the image

You can override the image's entrypoint to inspect the image and debug its contents:

```bash
docker run --rm -it --device nvidia.com/gpu=all --entrypoint bash docker.io/gabrielmscampos/vllmserver:latest
```

From within the container, you can start the server as normal:

```bash
python -m vllmserver --model_id Qwen/Qwen3.5-0.8B
```

### Running the container

Simply specify the arguments directly after the image identifier:

```bash
docker run --rm -it --device nvidia.com/gpu=all -p 8080:8080 -v ./hf-models:/mnt/hf-models docker.io/gabrielmscampos/vllmserver:latest --model_dir /mnt/hf-models/Qwen/Qwen3.5-0.8B
```

The model above was loaded from a local directory, you can download it with:

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir ./hf-models/Qwen/Qwen3.5-0.8B
```

## Notes

- Tensor parallelism is configured automatically based on the number of available GPUs
