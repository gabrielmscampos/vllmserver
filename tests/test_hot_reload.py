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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllmserver.hot_reload import HotReloadConfig, HotReloadManager, ModelSpec, load_hot_reload_config


def _make_config(*specs: tuple[str, str, bool]) -> HotReloadConfig:
    """Build a HotReloadConfig from (name, model_dir, default) tuples."""
    return HotReloadConfig(
        models=[ModelSpec(name=n, model_dir=d, default=df) for n, d, df in specs],
        path=Path("/fake/config.yaml"),
    )


def _make_manager(config: HotReloadConfig, model=None) -> HotReloadManager:
    model = model or MagicMock()
    register_fn = MagicMock()
    manager = HotReloadManager(config, model, register_fn)
    return manager


def test_load_config_ok(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "models:\n  - name: m1\n    model_dir: /tmp/m1\n    default: true\n  - name: m2\n    model_dir: /tmp/m2\n"
    )
    result = load_hot_reload_config(cfg)
    assert len(result.models) == 2
    assert result.models[0].name == "m1"
    assert result.models[0].default is True
    assert result.models[1].default is False


def test_load_config_empty_file_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("")
    with pytest.raises(ValueError, match="must contain a 'models' key"):
        load_hot_reload_config(cfg)


def test_load_config_missing_models_key_raises(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("other_key: value\n")
    with pytest.raises(ValueError, match="must contain a 'models' key"):
        load_hot_reload_config(cfg)


def test_validate_config_ok(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    config = _make_config(("m1", str(d), True))
    HotReloadManager.validate_config(config)  # should not raise


def test_validate_config_no_default_raises(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    config = _make_config(("m1", str(d), False))
    with pytest.raises(ValueError, match="exactly one model"):
        HotReloadManager.validate_config(config)


def test_validate_config_two_defaults_raises(tmp_path):
    d1, d2 = tmp_path / "m1", tmp_path / "m2"
    d1.mkdir()
    d2.mkdir()
    config = _make_config(("m1", str(d1), True), ("m2", str(d2), True))
    with pytest.raises(ValueError, match="exactly one model"):
        HotReloadManager.validate_config(config)


def test_validate_config_missing_dir_raises(tmp_path):
    config = _make_config(("m1", str(tmp_path / "nonexistent"), True))
    with pytest.raises(ValueError, match="does not exist"):
        HotReloadManager.validate_config(config)


@pytest.mark.asyncio
async def test_ensure_loaded_noop_when_already_current():
    config = _make_config(("m1", "/fake/m1", True))
    model = MagicMock()
    model.swap_to = AsyncMock()
    manager = _make_manager(config, model)
    manager._current = config.models[0]

    await manager.ensure_loaded("m1")

    model.swap_to.assert_not_called()


@pytest.mark.asyncio
async def test_ensure_loaded_triggers_swap():
    config = _make_config(("m1", "/fake/m1", True), ("m2", "/fake/m2", False))
    model = MagicMock()
    model.swap_to = AsyncMock()
    manager = _make_manager(config, model)
    manager._current = config.models[0]  # m1 is loaded

    await manager.ensure_loaded("m2")

    model.swap_to.assert_awaited_once_with("/fake/m2", "m2", None)
    assert manager._current.name == "m2"


@pytest.mark.asyncio
async def test_ensure_loaded_concurrent_waiters_only_swap_once():
    """Two coroutines waiting for the same unloaded model should only swap once."""
    import asyncio

    config = _make_config(("m1", "/fake/m1", True), ("m2", "/fake/m2", False))
    model = MagicMock()
    call_count = 0

    async def counted_swap(model_dir, model_name, vllm_args=None):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)  # yield

    model.swap_to = counted_swap
    manager = _make_manager(config, model)
    manager._current = config.models[0]

    await asyncio.gather(
        manager.ensure_loaded("m2"),
        manager.ensure_loaded("m2"),
    )

    assert call_count == 1
    assert manager._current.name == "m2"


@pytest.mark.asyncio
async def test_swap_failure_falls_back_to_default():
    config = _make_config(("m1", "/fake/m1", True), ("m2", "/fake/m2", False))
    model = MagicMock()
    call_args: list[str] = []

    async def swap_side_effect(model_dir, model_name, vllm_args=None):
        call_args.append(model_name)
        if model_name == "m2":
            raise RuntimeError("failed to load m2")

    model.swap_to = swap_side_effect
    manager = _make_manager(config, model)
    manager._current = config.models[0]

    await manager.ensure_loaded("m2")

    assert "m2" in call_args
    assert "m1" in call_args
    assert manager._current.name == "m1"


@pytest.mark.asyncio
async def test_swap_failure_default_also_fails_raises():
    config = _make_config(("m1", "/fake/m1", True))
    model = MagicMock()

    async def always_fail(model_dir, model_name, vllm_args=None):
        raise RuntimeError("always fails")

    model.swap_to = always_fail
    manager = _make_manager(config, model)
    manager._current = None

    with pytest.raises(RuntimeError, match="always fails"):
        await manager.ensure_loaded("m1")


@pytest.mark.asyncio
async def test_watch_config_adds_new_model(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models:\n  - name: m1\n    model_dir: /fake/m1\n    default: true\n")
    config = load_hot_reload_config(cfg)
    register_fn = MagicMock()
    model = MagicMock()
    manager = HotReloadManager(config, model, register_fn)

    # Add m2 to the file, then trigger one reload tick directly
    cfg.write_text(
        "models:\n  - name: m1\n    model_dir: /fake/m1\n    default: true\n  - name: m2\n    model_dir: /fake/m2\n"
    )
    await manager._reload_once()

    assert "m2" in manager._registry
    register_fn.assert_called_once_with("m2")


@pytest.mark.asyncio
async def test_watch_config_bad_yaml_does_not_crash(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("models:\n  - name: m1\n    model_dir: /fake/m1\n    default: true\n")
    config = load_hot_reload_config(cfg)
    model = MagicMock()
    register_fn = MagicMock()
    manager = HotReloadManager(config, model, register_fn)

    cfg.write_text(": invalid: yaml: [[[")
    await manager._reload_once()  # should not raise

    assert list(manager._registry.keys()) == ["m1"]
    register_fn.assert_not_called()
