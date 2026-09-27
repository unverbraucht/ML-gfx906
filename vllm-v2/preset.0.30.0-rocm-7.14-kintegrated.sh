#!/bin/bash

export VLLM_VERSION="0.30.0"
export VLLM_ROCM_VERSION="7.14"
export VLLM_PYTORCH_VERSION="v2.13.0"

# KIntegrated vLLM gfx906/v0.30.0: upstream v0.30.0 merged onto the gfx906 fork
# line (tagged gfx906/v0.30.0-final). Carries the 0.29 promotion work plus:
# - V2 model runner default for the models whose parity passed (dense 27B, MoE
#   35B, Nemotron 3.5 Lightning, Ornith); Gemma-4 and Muse-Glimmer stay pinned to
#   V1 (VLLM_USE_V2_MODEL_RUNNER=0) until their gate passes. Upstream removes V1
#   in 0.32.0.
# - VIT-1: the Qwen3.5/3.8 vision tower runs on the custom FA (default ON,
#   image-prompt TTFT -11.5 % @1024x1024, no per-boot Triton ViT JIT); kill
#   switches GFX906_FA_VIT=0 / GFX906_FA_VIT_AUTO=0.
# - MTP k=3 is the default spec config; the cudagraph capture ladder must be a
#   multiple of k+1 up to max_num_seqs x (k+1).
# - REL30-1 fixed: fused M=1 MoE align on V1 only.
# - CAT-1 corpus tooling (draft-vocab list, default OFF per deployment),
#   A3 fused-draft revived but inert (default OFF), 0.30 fused KV-cache layout.
# VLLM_VERSION is applied as a local git tag so setuptools-scm reports it
# (see AGENTS.md). The vllm short commit is kept in the image tag for build
# metadata while this is a prerelease.
export VLLM_REPO="https://github.com/KIntegrated/vllm-gfx906-mobydick.git"
export VLLM_BRANCH="gfx906/v0.30.0"
export VLLM_COMMIT="6ccbb03119"

# triton / flash-attention: the validated gfx906 forks as used by the 0.29.0
# image (triton 3.6.0+gfx906). NOTE (TRITON-1): the fork's docs now default
# gfx906 to *stock* upstream Triton 3.8.0 (gfx906 is upstream since aa53dba7455,
# ISAFamily::GCN5_1) — building that from source would be a follow-up image
# build with TRITON_CODEGEN_BACKENDS=amd and
# TRITON_APPEND_CMAKE_ARGS="-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
# (see README.md "TRITON 3.8.0" / docs/gfx906/RECON-triton-1.md).
export VLLM_TRITON_REPO="https://github.com/ai-infos/triton-gfx906.git"
export VLLM_TRITON_BRANCH="v3.6.0+gfx906"

export VLLM_FA_REPO="https://github.com/ai-infos/flash-attention-gfx906.git"
export VLLM_FA_BRANCH="gfx906/v2.8.3.x"
export VLLM_FA_COMMIT="0ac8e77"

export VLLM_PRESET_NAME="${VLLM_VERSION}-${VLLM_COMMIT}-rocm-${VLLM_ROCM_VERSION}-kintegrated"
export VLLM_EXTRA_REQUIREMENTS="KIntegrated_vllm-gfx906-mobydick/${VLLM_COMMIT}.txt"
