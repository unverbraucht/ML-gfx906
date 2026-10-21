#!/bin/bash

export VLLM_VERSION="0.32.0"
export VLLM_ROCM_VERSION="7.14"
export VLLM_PYTORCH_VERSION="v2.13.0"

# KIntegrated vLLM gfx906/v0.32.0: upstream v0.32.0 merged onto the gfx906 fork
# line (merge commit 1e514c6fd0; 822 fork / 840 upstream commits, 59 conflicts),
# plus the post-merge fixes. Release snapshot: docs/gfx906/RELEASE-0.32.0-final.md.
# - csrc/libtorch_stable/* flattened into csrc/* by upstream (187 fork files);
#   the fork's 4 op registrations were re-ported into csrc/ops.h /
#   csrc/torch_bindings.cpp, and csrc/moe/moe_wna16.cu stays unconditional
#   (upstream made it CUDA-only; the gfx906 WNA16 path needs it on ROCm).
# - Five marker-less merge breakages fixed: a lost cuda_utils include, upstream
#   dropping cuda_utils::ceil_div + HOST_DEVICE_INLINE (restored with a
#   VLLM_CEIL_DIV_QUALIFIER), op-registration ordering inside the _C fragment,
#   and two imports of modules upstream deleted.
# - Nemotron-Lightning on 0.32 (96db9355c3): ckpt_*_proj_name=None no longer
#   breaks the "all MoE projections share a quantization scheme" check.
# - moe_wna16 qzeros loader is expert-parallel aware (5f41d480b9): EP reports
#   tp_size=1 while the checkpoint ids are global, so TP=2 + EP used to die with
#   IndexError at load. Qwen3.5-35B-A3B-AWQ now serves with EP (2/2 requests).
# - Packed KV `auto` now engages under MTP spec decode (7fcd4205b1): the hybrid
#   aligner left mamba_page_size_padded=None when no padding was needed, so the
#   gate compared (1072, None) with (1408, 1103872) and read +0.0 %. Served gate
#   at 32k/MTP: 890,070 -> 1,057,678 tokens (+18.8 %) against +28.4 % predicted.
#   Re-anchored: 35B 8k +33.6 % (no spec) / +32.6 % (MTP), 32k +28.4 %;
#   Lightning 8k +0.4 % (off), 16k +14.8 % (on).
# - QSA selection width is rejected at config time when it exceeds the device's
#   dynamic-LDS budget (3d5486abd6) instead of faulting in the kernel.
# - V2 model runner is upstream's default for every arch; V1 remains available
#   as an escape hatch (verified with Muse-Glimmer + the DFlash drafter).
# - transformers 5.18.0 (upstream 0.32 requires >= 5.16.1, < 5.20.0; the 0.31
#   pin of 5.15.0 is below that and predates Qwen4ExpTextConfig).
# Gates: build rc=0; PPL 35B 15.9555 / dense 27B 14.3750 / Lightning 26.9922;
# Lightning and 35B serving smoke green; graphs PIECEWISE 2/2 + FULL 2/2.
export VLLM_REPO="https://github.com/KIntegrated/vllm-gfx906-mobydick.git"
export VLLM_BRANCH="gfx906/v0.32.0"
# Pinned to the release commit (docs/release snapshot). Later commits on
# gfx906/v0.32.0 (e.g. the FA tuning-mirror guard, d5b53702e8) are
# post-release and do not affect the built artifact; bump this to the new
# tip when the next image is built (the Dockerfile clones --depth 1, so a
# commit that is not the branch tip cannot be checked out).
export VLLM_COMMIT="a1b8a71744"

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
