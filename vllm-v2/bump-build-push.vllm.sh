#!/bin/bash
# bump-build-push.vllm.sh: bump the pinned vLLM commit to the branch tip,
# rebuild, re-tag (mixa3607 + unverbraucht) and push.
#
# Usage:  ./bump-build-push.vllm.sh preset.0.28.0-rocm-7.14-kintegrated.sh [--push]
#   --push  also push to the unverbraucht namespace (default: local build only)
#
# Requires: git (HTTPS remotes), docker buildx, the preset file, and the
# requirements pinset dir. Uses the LRZ APT mirror unless VLLM_APT_MIRROR set.
set -euo pipefail

PRESET="${1:?usage: bump-build-push.vllm.sh <preset> [--push]}"
PUSH=0
[ "${2:-}" = "--push" ] && PUSH=1

cd "$(dirname "$0")"

# --- load current preset values ---
. "./${PRESET}"
REQ_DIR="build-context/requirements/$(dirname "$VLLM_EXTRA_REQUIREMENTS")"

NEW_TIP=$(git ls-remote "$VLLM_REPO" "refs/heads/$VLLM_BRANCH" | awk '{print $1}')
if [ -z "$NEW_TIP" ]; then
  echo "ERROR: could not resolve $VLLM_REPO refs/heads/$VLLM_BRANCH" >&2; exit 1
fi
NEW_SHORT="${NEW_TIP:0:10}"

echo "== current: VLLM_BRANCH=$VLLM_BRANCH VLLM_COMMIT=$VLLM_COMMIT"
echo "== tip:     $NEW_TIP (short $NEW_SHORT)"

# --- update preset only if the tip actually differs (compare short forms) ---
if [ "$NEW_SHORT" != "$VLLM_COMMIT" ]; then
  echo "== bumping VLLM_COMMIT $VLLM_COMMIT -> $NEW_SHORT"
  sed -i "s/^export VLLM_COMMIT=.*/export VLLM_COMMIT=\"${NEW_SHORT}\"/" "$PRESET"
  # copy req pinset under the new commit name (content unchanged unless deps changed)
  if [ ! -f "${REQ_DIR}/${NEW_SHORT}.txt" ]; then
    cp "${REQ_DIR}/${VLLM_COMMIT}.txt" "${REQ_DIR}/${NEW_SHORT}.txt"
    echo "  (copied req pinset ${VLLM_COMMIT}.txt -> ${NEW_SHORT}.txt)"
  fi
  # re-source for the new value
  . "./$PRESET"
  echo "== now: VLLM_COMMIT=$VLLM_COMMIT"
else
  echo "== already at branch tip; no preset change"
fi

# --- build (local) ---
mkdir -p ./logs
echo "== building ${VLLM_IMAGE}:${VLLM_PRESET_NAME} ..."
export VLLM_APT_MIRROR="${VLLM_APT_MIRROR:-http://ubuntu.mirror.lrz.de/ubuntu/}"
./build.vllm.sh 2>&1 | tee ./logs/bump_$(date +%Y%m%d%H%M%S).log

# --- tag + push unverbraucht ---
SHORT=${VLLM_COMMIT}
for T in "unverbraucht/vllm-gfx906:${VLLM_VERSION}-rocm-${VLLM_ROCM_VERSION}" \
         "unverbraucht/vllm-gfx906:${VLLM_VERSION}-${SHORT}"; do
  docker tag "${VLLM_IMAGE}:${VLLM_PRESET_NAME}" "$T"
  echo "== tagged $T"
  if [ "$PUSH" = "1" ]; then
    docker push "$T" && echo "  pushed $T"
  else
    echo "  (not pushing; use --push)"
  fi
done

echo "== done. commit bump in: $PRESET"
