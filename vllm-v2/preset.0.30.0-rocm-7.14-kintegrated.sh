#!/bin/bash

export VLLM_VERSION="0.30.0"
export VLLM_ROCM_VERSION="7.14"
export VLLM_PYTORCH_VERSION="v2.13.0"

# KIntegrated vLLM gfx906/v0.30.0: upstream v0.30.0 merged onto the gfx906 fork
# line (tagged gfx906/v0.30.0-final). Carries the 0.29 promotion work plus:
# - V2 model runner default for the models whose parity passed (dense 27B, MoE
#   35B, Nemotron 3.5 Lightning, Ornith). CORRECTION (2026-10-20): Gemma-4 and
#   Muse-Glimmer were NOT left pinned to V1 -- both gates passed (templated
#   V1-vs-V2 comparison for Gemma-4 on 2026-09-15, V2 serving for Muse-Glimmer on
#   2026-09-16) and the pins were lifted. V1 also survived into 0.32.0 (the env
#   var is still honoured; only HiSparse refuses it), so the pin remains available
#   as a rollback rather than as a supported default.
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
export VLLM_COMMIT="1c4d1065ad"

# triton: *stock upstream* Triton 3.8.0 since the 0.30.0 build. gfx906 is an
# upstream target (aa53dba7455, ISAFamily::GCN5_1), so the old
# ai-infos/triton-gfx906 fork (v3.6.0 + a 7-line patch) is no longer used and is
# kept only as a rollback. Build settings live in the Dockerfile's build_triton
# stage (TRITON_CODEGEN_BACKENDS=amd, TRITON_APPEND_CMAKE_ARGS RPATH fix).
# See README.md "TRITON 3.8.0" and docs/gfx906/RECON-triton-1.md.
export VLLM_TRITON_REPO="https://github.com/triton-lang/triton.git"
export VLLM_TRITON_BRANCH="v3.8.0"

export VLLM_FA_REPO="https://github.com/ai-infos/flash-attention-gfx906.git"
export VLLM_FA_BRANCH="gfx906/v2.8.3.x"
export VLLM_FA_COMMIT="0ac8e77"

export VLLM_PRESET_NAME="${VLLM_VERSION}-${VLLM_COMMIT}-rocm-${VLLM_ROCM_VERSION}-kintegrated"
export VLLM_EXTRA_REQUIREMENTS="KIntegrated_vllm-gfx906-mobydick/${VLLM_COMMIT}.txt"
