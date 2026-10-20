#!/bin/bash

export VLLM_VERSION="0.31.0"
export VLLM_ROCM_VERSION="7.14"
export VLLM_PYTORCH_VERSION="v2.13.0"

# KIntegrated vLLM gfx906/v0.31.0: upstream v0.31.0 merged onto the gfx906 fork
# line (merge commit db60e2cbd9), plus the whole post-merge dev train that has
# landed on the branch since. Carries the 0.30 promotion work plus:
# - upstream v0.31.0 with the merge's semantic fixes: the CT WNA16 N-first
#   expert repack (36bd5a82d0), REL30-1's fused M=1 align defaulting OFF
#   (427c445723), the V1/V2 runner + triton_matmul dtype fixes and the GDN
#   spec-decode update_block_table crash under MTP (d4a182edfb).
# - mi50-vllm-lab ports: skinny/W4 GEMMs (VLLM_GFX906_SKINNY_M16) and the TP=2
#   host all-reduce; the lab serving harnesses live in tools/gfx906_lab_bench/.
# - DFlash2/LiLiCorr drafters work: fp16 residual rescale (2048 divisor for
#   LiLiCorr, weight_scale path for quantized drafters) and the non-causal
#   sliding window kept in the custom FA. BOLT-1 closed at 6.77 accept/step.
# - BOLT-2 closed (6fce381638): the DFlash drafter stores context KV from row 0
#   on prefill steps — an upstream bug, report in
#   docs/gfx906/UPSTREAM-dflash-context-kv.md.
# - SYV-4 sort-free sampler runs before the triton top-k/top-p path
#   (afc9565da5; dead code before, 1434 -> 306 us at B=1 k=8, bit-identical).
# - Packed KV layout ("E2", K Q8_0 + V fp16 in one cell) is now default `auto`
#   (2026-10-19 train): the engine keeps the alias layout unless the predicted
#   *usable* KV capacity gain clears 5 %, computed per (model, max_model_len)
#   from the hybrid aligner's two dry runs. 35B 8k +33.6 % -> on, Lightning 8k
#   +0.4 % -> off, Lightning 16k +14.8 % -> on. `--gfx906-fa-packed on|off`
#   forces it; `GFX906_FA_PACKED` is the env override (the only form a TP worker
#   sees before the layout is resolved). Throughput-neutral (TP=1 B=4 +0.41 %,
#   TP=2 B=4 -0.5 %), so the win is capacity.
# - Packed KV at TP>1 is fixed (`ad69623e9c`): the layout is now derived from the
#   pool's cell width instead of a flag captured at impl construction, which in a
#   TP worker happens before the layout is resolved and made packed x TP=2 die
#   with "D must be multiple of 32". Also fixed: `--gfx906-fa-packed on` used to
#   be silently served as the alias layout (`d3d3a59040`).
# - `top_k_per_row_decode` fails closed when the selection width exceeds the
#   device's dynamic-LDS budget (`ef2155cb40`): on gfx906 that is topK <= 8192,
#   and exceeding it used to fault inside the kernel with a sticky HIP error.
# - Muse-Glimmer's spec method is the official DFlash assistant (`dflash`, k=7,
#   graphs), not MTP (that checkpoint has no MTP head): 39-43 t/s at B=1 and
#   60-62 t/s aggregate decode at B=4, acceptance 2.95, TP=2 only - TP=1 cannot
#   fit the drafter's 2.51 GiB embedding (`9eb7e9adbc`). The obsolete
#   TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0 workaround is gone (stock Triton 3.8.0).
# - MTP k=3 is the default spec config for the Qwen3.5/3.8 family; the cudagraph
#   capture ladder must be a multiple of k+1 up to max_num_seqs x (k+1).
# VLLM_VERSION is applied as a local git tag so setuptools-scm reports it
# (see AGENTS.md). The vllm short commit is kept in the image tag for build
# metadata while this is a prerelease.
export VLLM_REPO="https://github.com/KIntegrated/vllm-gfx906-mobydick.git"
export VLLM_BRANCH="gfx906/v0.31.0"
export VLLM_COMMIT="3b5f908a60"

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
