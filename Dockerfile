ARG CUDA_VERSION=13.0.2
ARG VENV_PATH=prod_venv
ARG WORKSPACE_DIR=/kserve-workspace

#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04 AS base

ARG WORKSPACE_DIR
ARG CUDA_VERSION=13.0.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo gcc python3 python3-venv python3-pip python-is-python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv and ensure it's in PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0 10.0 11.0 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real;120-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}

WORKDIR ${WORKSPACE_DIR}

#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build

ARG WORKSPACE_DIR
ARG LMCACHE_VERSION=0.4.4
ARG FLASHINFER_VERSION=0.6.8-1
ARG BUILD_VERSION=0.0.0+local

WORKDIR ${WORKSPACE_DIR}

ARG VENV_PATH
RUN python3 -m venv ${VENV_PATH}
# Activate virtual env by setting VIRTUAL_ENV
ENV VIRTUAL_ENV=${WORKSPACE_DIR}/${VENV_PATH}
ENV PATH="${WORKSPACE_DIR}/${VENV_PATH}/bin:$PATH"

# From this point, all Python packages will be installed in the virtual environment and copied to the final image

# Copy vllmserver implementation and metadata
COPY vllmserver vllmserver
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY README.md README.md

# Install dependencies
# SETUPTOOLS_SCM_PRETEND_VERSION is required because .git is not in the build context
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION=${BUILD_VERSION} uv sync --active --no-cache

# Install vllm addons
# https://docs.vllm.ai/en/latest/models/extensions/runai_model_streamer.html, https://docs.vllm.ai/en/latest/models/extensions/tensorizer.html
# https://docs.vllm.ai/en/latest/models/extensions/fastsafetensor.html
RUN --mount=type=cache,target=/root/.cache/pip pip install vllm[runai,tensorizer,fastsafetensors]==$(uv pip show vllm | grep 'Version: ' | awk '{ print $2 }')

# Install lmcache
RUN --mount=type=cache,target=/root/.cache/pip pip install lmcache==${LMCACHE_VERSION} \
    && pip uninstall -y nixl-cu12 cupy-cuda12x \
    && pip install nixl-cu13 cupy-cuda13x

# Use Bash with `-o pipefail` so we can leverage Bash-specific features (like `[[ … ]]` for glob tests)
# and ensure that failures in any part of a piped command cause the build to fail immediately.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install flashinfer
# https://docs.flashinfer.ai/installation.html
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install flashinfer-cubin==${FLASHINFER_VERSION} && \
    pip install flashinfer-jit-cache==${FLASHINFER_VERSION} \
        --extra-index-url https://flashinfer.ai/whl/cu$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr -d '.') && \
    flashinfer show-config

#################### WHEEL BUILD IMAGE ####################

#################### PROD IMAGE ####################
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu24.04 AS prod

ARG WORKSPACE_DIR
ARG CUDA_VERSION=13.0.2
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR ${WORKSPACE_DIR}

# Install Python and other dependencies
# cuda-nvcc is required for DeepGEMM JIT kernel compilation at runtime
RUN CUDA_PKG=$(echo ${CUDA_VERSION} | cut -d. -f1,2 | tr '.' '-') && \
    apt-get update -y \
    && apt-get upgrade -y \
    && apt-get install -y software-properties-common curl ffmpeg libsm6 libxext6 libgl1 gcc python3-dev libibverbs-dev cuda-nvcc-${CUDA_PKG} \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG VENV_PATH
# Activate virtual env by setting VIRTUAL_ENV
ENV VIRTUAL_ENV=${WORKSPACE_DIR}/${VENV_PATH}
ENV PATH="${WORKSPACE_DIR}/${VENV_PATH}/bin:$PATH"

# Create non-root user
RUN userdel -r ubuntu && useradd kserve -m -u 1000 -d /home/kserve

COPY --from=build --chown=kserve:kserve ${WORKSPACE_DIR}/$VENV_PATH $VENV_PATH
COPY --from=build ${WORKSPACE_DIR}/vllmserver vllmserver

# Set a writable Hugging Face home folder to avoid permission issue. See https://github.com/kserve/kserve/issues/3562
ENV HF_HOME="/tmp/huggingface"
# https://huggingface.co/docs/safetensors/en/speed#gpu-benchmark
ENV SAFETENSORS_FAST_GPU="1"
# https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhubdisabletelemetry
ENV HF_HUB_DISABLE_TELEMETRY="1"
# NCCL Lib path for vLLM. https://github.com/vllm-project/vllm/blob/ec784b2526219cd96159a52074ab8cd4e684410a/vllm/utils.py#L598-L602
ENV VLLM_NCCL_SO_PATH="/lib/x86_64-linux-gnu/libnccl.so.2"
# https://github.com/vllm-project/vllm/issues/6152
# Set the multiprocess method to spawn to avoid issues with cuda initialization for `mp` executor backend.
ENV VLLM_WORKER_MULTIPROC_METHOD="spawn"
ENV CUDA_HOME="/usr/local/cuda"

USER 1000
ENV PYTHONPATH=${WORKSPACE_DIR}/vllmserver
ENTRYPOINT ["python3", "-m", "vllmserver"]
#################### PROD IMAGE ####################
