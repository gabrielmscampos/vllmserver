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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from kserve.logging import logger


if TYPE_CHECKING:
    from .model import HotReloadVLLMModel


@dataclass
class ModelSpec:
    name: str
    model_dir: str
    default: bool = False
    vllm_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class HotReloadConfig:
    models: list[ModelSpec]
    path: Path = field(repr=False, compare=False)


def load_hot_reload_config(path: str | Path) -> HotReloadConfig:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not raw or "models" not in raw:
        raise ValueError(f"Hot-reload config at '{path}' must contain a 'models' key")
    models = [ModelSpec(**entry) for entry in raw["models"]]
    return HotReloadConfig(models=models, path=path)


class HotReloadManager:
    def __init__(
        self,
        config: HotReloadConfig,
        model: "HotReloadVLLMModel",
        register_fn: Callable[[str], None],
    ):
        self._config = config
        self._model = model
        self._register_fn = register_fn
        self._lock = asyncio.Lock()
        self._current: ModelSpec | None = None
        self._swapping_to_default = False
        self._registry: dict[str, ModelSpec] = {s.name: s for s in config.models}

    @staticmethod
    def validate_config(config: HotReloadConfig) -> None:
        defaults = [s for s in config.models if s.default]
        if len(defaults) != 1:
            raise ValueError(
                f"Hot-reload config must have exactly one model with 'default: true', found {len(defaults)}"
            )
        for spec in config.models:
            if not Path(spec.model_dir).is_dir():
                raise ValueError(f"model_dir does not exist or is not a directory: '{spec.model_dir}'")

    def get_default(self) -> ModelSpec:
        return next(s for s in self._registry.values() if s.default)

    def get_spec(self, name: str) -> ModelSpec | None:
        return self._registry.get(name)

    async def ensure_loaded(self, model_name: str) -> None:
        if self._current is not None and self._current.name == model_name:
            return
        async with self._lock:
            if self._current is not None and self._current.name == model_name:
                return
            spec = self._registry[model_name]
            await self._swap_to(spec)

    async def _swap_to(self, spec: ModelSpec) -> None:
        current_name = self._current.name if self._current else "none"
        logger.info("Hot-reload: swapping from '%s' to '%s'", current_name, spec.name)
        logger.info("Hot-reload: vllm_args for '%s': %s", spec.name, spec.vllm_args or {})
        try:
            await self._model.swap_to(spec.model_dir, spec.name, spec.vllm_args or None)
            self._current = spec
            logger.info("Hot-reload: successfully loaded '%s'", spec.name)
        except Exception:
            logger.exception("Hot-reload: failed to load '%s'", spec.name)
            default = self.get_default()
            if spec.name != default.name and not self._swapping_to_default:
                logger.warning("Hot-reload: falling back to default model '%s'", default.name)
                self._swapping_to_default = True
                try:
                    await self._swap_to(default)
                finally:
                    self._swapping_to_default = False
            else:
                raise

    async def _reload_once(self) -> None:
        try:
            new_config = load_hot_reload_config(self._config.path)
            for spec in new_config.models:
                if spec.name not in self._registry:
                    logger.info("Hot-reload: discovered new model '%s' in config", spec.name)
                    self._registry[spec.name] = spec
                    self._register_fn(spec.name)
        except Exception:  # noqa: BLE001
            logger.warning("Hot-reload: failed to reload config '%s'", self._config.path, exc_info=True)

    async def _watch_config(self) -> None:
        while True:
            await asyncio.sleep(60)
            await self._reload_once()
