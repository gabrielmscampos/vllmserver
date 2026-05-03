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
from argparse import Namespace
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from kserve.logging import logger
from kserve_storage import Storage
from transformers import AutoConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.model_executor.models import ModelRegistry
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser


def list_of_strings(arg):
    return arg.split(",")


def get_model_id_or_path(args: Namespace) -> str | Path:
    # If --model_id is specified then pass model_id to HF API, otherwise load the model from /mnt/models
    if args.model_id:
        return cast(str, args.model_id)
    return Path(Storage.download(args.model_dir))


def infer_vllm_supported_from_model_architecture(
    model_config_path: Path | str,
    trust_remote_code: bool = False,
) -> bool:
    model_config = AutoConfig.from_pretrained(model_config_path, trust_remote_code=trust_remote_code)
    for architecture in model_config.architectures:
        if architecture not in ModelRegistry.get_supported_archs():
            logger.info("not a supported model by vLLM")
            return False
    return True


def add_vllm_cli_parser(parser: FlexibleArgumentParser) -> FlexibleArgumentParser:
    return make_arg_parser(parser)


def build_vllm_engine_args(args) -> "AsyncEngineArgs":
    return AsyncEngineArgs.from_cli_args(args)


@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args: AsyncEngineArgs,
) -> AsyncIterator[EngineClient]:
    """
    Create V1 AsyncLLM EngineClient.

    Returns the Client or None if the creation failed.
    """

    usage_context = UsageContext.OPENAI_API_SERVER
    vllm_config = await asyncio.to_thread(engine_args.create_engine_config, usage_context=usage_context)

    from vllm.v1.engine.async_llm import AsyncLLM

    async_llm: AsyncLLM | None = None
    try:
        async_llm = await asyncio.to_thread(
            AsyncLLM.from_vllm_config,
            vllm_config=vllm_config,
            usage_context=usage_context,
            enable_log_requests=engine_args.enable_log_requests,
            disable_log_stats=engine_args.disable_log_stats,
        )
        yield async_llm
    finally:
        logger.info("V1 AsyncLLM build complete")
