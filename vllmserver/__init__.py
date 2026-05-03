# Copyright 2023 The KServe Authors.
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

from ._version import __version__


def _patch_vllm_score_compat() -> None:
    # kserve <=0.18 imports vllm.entrypoints.pooling.score.protocol, which was
    # renamed to vllm.entrypoints.pooling.scoring.protocol in vLLM 0.20.
    # Register the old path as an alias so kserve loads without modification.
    import importlib
    import sys
    import types

    _parent = "vllm.entrypoints.pooling.score"
    if _parent in sys.modules:
        return
    try:
        scoring = importlib.import_module("vllm.entrypoints.pooling.scoring.protocol")
    except ModuleNotFoundError:
        return
    sys.modules[_parent] = types.ModuleType(_parent)
    sys.modules[f"{_parent}.protocol"] = scoring


_patch_vllm_score_compat()
del _patch_vllm_score_compat


__all__ = ["__version__"]
