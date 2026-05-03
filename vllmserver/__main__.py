# Copyright 2024 The KServe Authors.
# Copyright 2026 Gabriel Moreira da Silva Campos.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import copy
import sys
from pathlib import Path
from typing import Any

import kserve
from fastapi import HTTPException
from kserve import logging
from kserve.logging import logger
from pydantic import BaseModel
from vllm.utils.argparse_utils import FlexibleArgumentParser

from ._version import __version__
from .model import HotReloadVLLMModel, VLLMModel
from .request_logger import RequestLogger as KServeCustomRequestLogger
from .utils import add_vllm_cli_parser, infer_vllm_supported_from_model_architecture, list_of_strings


if __name__ == "__main__":
    parser = FlexibleArgumentParser(parents=[kserve.model_server.parser])

    parser.add_argument(
        "--model_dir",
        required=False,
        default="/mnt/models",
        help="A URI pointer to the model binary",
    )
    parser.add_argument("--model_id", required=False, default=None, help="Huggingface model id")
    parser.add_argument("--model_revision", required=False, default=None, help="Huggingface model revision")
    parser.add_argument(
        "--tokenizer_revision",
        required=False,
        default=None,
        help="Huggingface tokenizer revision",
    )
    parser.add_argument(
        "--max_length",
        dest="max_model_len",
        type=int,
        required=False,
        help="max sequence length for the tokenizer. will be deprecated in favour of --max_model_len",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        required=False,
        help="max number of tokens the model can process/tokenize. If not mentioned, uses model's max position encodings",
    )
    parser.add_argument(
        "--disable_lower_case",
        action="store_true",
        help="do not use lower case for the tokenizer",
    )
    parser.add_argument(
        "--disable_special_tokens",
        action="store_true",
        help="the sequences will not be encoded with the special tokens relative to their model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=False,
        help="allow loading of models and tokenizers with custom code",
    )
    parser.add_argument(
        "--tensor_input_names",
        type=list_of_strings,
        default=None,
        help="the tensor input names passed to the model",
    )
    parser.add_argument("--return_token_type_ids", action="store_true", help="Return token type ids")
    parser.add_argument(
        "--return_offsets_mapping",
        action="store_true",
        default=False,
        help="Return start/end character offsets for each token (token_classification only).",
    )

    # Create a mutually exclusive group for output format options
    # This group allows the user to choose between returning probabilities or disabling postprocessing.
    output_format_group = parser.add_mutually_exclusive_group()
    output_format_group.add_argument(
        "--return_probabilities",
        action="store_true",
        help="Return probabilities instead of logits for classification tasks such as token classification, text classification and fill-mask.",
    )
    output_format_group.add_argument(
        "--return_raw_logits",
        action="store_true",
        help="Return raw logits without processing. Supported only classification tasks such as token classification, text classification and fill-mask.",
    )
    parser.add_argument("--disable_log_requests", action="store_true", help="Disable logging requests")
    parser.add_argument(
        "--hot-reload-config",
        dest="hot_reload_config",
        required=False,
        default=None,
        help="Path to a YAML config for on-demand model hot-reload",
    )

    # The initial_args are required to determine whether the vLLM backend is enabled.
    initial_args, _ = parser.parse_known_args()
    parser = add_vllm_cli_parser(parser)
    args, _ = parser.parse_known_args()

    logger.info("Starting vllmserver model server version %s", __version__)

    if args.configure_logging:
        logging.configure_logging(args.log_config_file)

    try:
        model_server = kserve.ModelServer()

        if args.hot_reload_config:
            from .hot_reload import HotReloadManager, load_hot_reload_config

            hr_config = load_hot_reload_config(args.hot_reload_config)
            HotReloadManager.validate_config(hr_config)

            default_spec = next(s for s in hr_config.models if s.default)
            base_args = copy.deepcopy(args)
            args.model = default_spec.model_dir
            args.model_name = default_spec.name
            args.served_model_name = [default_spec.name]
            for key, val in (default_spec.vllm_args or {}).items():
                setattr(args, key.replace("-", "_"), val)
            if default_spec.vllm_args:
                logger.info("Applied vllm_args for default model '%s': %s", default_spec.name, default_spec.vllm_args)

            if not infer_vllm_supported_from_model_architecture(Path(default_spec.model_dir)):
                raise ValueError(f"Default model at '{default_spec.model_dir}' is not supported by vLLM")

            request_logger = (
                None if args.disable_log_requests else KServeCustomRequestLogger(max_log_len=args.max_log_len)
            )
            model = HotReloadVLLMModel(default_spec.name, args, request_logger=request_logger)  # type: ignore[arg-type]
            model._base_args = base_args
            model.load()

            manager = HotReloadManager(
                hr_config,
                model,
                register_fn=lambda name: model_server.register_model(model, name),
            )
            manager._current = default_spec
            model.set_manager(manager)

            from kserve.model_server import app as _kserve_app

            async def _wait_for_engine() -> None:
                await model._engine_ready.wait()

            _kserve_app.router.on_startup.append(_wait_for_engine)

            class _SwapRequest(BaseModel):
                model: str

            _swap_state: dict[str, Any] = {"task": None, "target": None, "error": None}

            @_kserve_app.post("/swap")
            async def _swap_endpoint(req: _SwapRequest) -> dict:
                spec = manager.get_spec(req.model)
                if spec is None:
                    raise HTTPException(status_code=404, detail=f"Model '{req.model}' is not in the registry")
                task: asyncio.Task | None = _swap_state["task"]
                if task is not None and not task.done():
                    raise HTTPException(
                        status_code=409,
                        detail=f"Swap to '{_swap_state['target']}' is already in progress",
                    )

                async def _do_swap() -> None:
                    try:
                        await manager.ensure_loaded(req.model)
                        _swap_state["error"] = None
                    except Exception as exc:  # noqa: BLE001
                        _swap_state["error"] = str(exc)

                _swap_state["target"] = req.model
                _swap_state["error"] = None
                _swap_state["task"] = asyncio.create_task(_do_swap())
                return {"status": "swapping", "model": req.model}

            @_kserve_app.get("/swap/status")
            async def _swap_status() -> dict:
                current = manager._current
                task: asyncio.Task | None = _swap_state["task"]
                swapping = task is not None and not task.done()
                return {
                    "model": current.name if current else None,
                    "ready": model.ready,
                    "swapping": swapping,
                    "swapping_to": _swap_state["target"] if swapping else None,
                    "error": _swap_state["error"],
                }

            for spec in hr_config.models:
                model_server.register_model(model, spec.name)

            model_server.start([model])

        else:
            from kserve.model_server import app as _kserve_app

            model = VLLMModel.load_model_from_cli(args)

            async def _wait_for_engine() -> None:
                await model._engine_ready.wait()

            _kserve_app.router.on_startup.append(_wait_for_engine)

            if args.lora_modules:
                for lora_module in args.lora_modules:
                    model_server.register_model(model, lora_module.name)

            model_server.start([model])

    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to start model server: {e}", exc_info=True)
        sys.exit(1)
