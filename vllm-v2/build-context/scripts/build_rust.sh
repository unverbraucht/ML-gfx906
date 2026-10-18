#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Kevin Read <me@kevin-read.com>
#
# Build the vLLM Rust artifacts (vllm-rs binary + _rust_* PyO3 extensions).
#
# Copied from vLLM's tools/build_rust.sh (v0.31.0, Apache-2.0) and kept in this
# build context on purpose: upstream moved the script from the repo root to
# tools/ in #58572 (v0.31.0), so referencing it with a --from=files_vllm COPY
# makes the Dockerfile layout-specific. Shipping it here is layout-agnostic
# (tools/build_rust.py, which does the actual work, exists in both layouts) and
# keeps build_rust's layer cache key to the Rust inputs only, so python-only
# vLLM commits do not trigger a cargo rebuild.
#
# Usage: tools/build_rust.sh [--debug]   (run from the repo root)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Read the required toolchain from rust-toolchain.toml.
TOOLCHAIN=$(grep '^channel' "$REPO_ROOT/rust-toolchain.toml" | sed 's/.*= *"\(.*\)"/\1/')

# Ensure rustup and the required toolchain are available.
if ! command -v rustup &>/dev/null; then
    echo "rustup not found, installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source "$HOME/.cargo/env"
fi

if ! rustup run "$TOOLCHAIN" rustc --version &>/dev/null; then
    echo "Installing Rust toolchain: $TOOLCHAIN"
    rustup toolchain install "$TOOLCHAIN"
fi

if [[ "${1:-}" == "--debug" ]]; then
    PROFILE_ARG="--debug"
else
    PROFILE_ARG="--release"
fi

python3 "$REPO_ROOT/tools/build_rust.py" "$PROFILE_ARG"
