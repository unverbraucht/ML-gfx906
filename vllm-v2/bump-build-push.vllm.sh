#!/bin/bash
# bump-build-push.vllm.sh: bump the pinned vLLM commit to the branch tip,
# rebuild, re-tag (mixa3607 + unverbraucht) and push.
#
# Usage:  ./bump-build-push.vllm.sh <preset> [--branch NAME] [--push]
#
#   --branch NAME  build that branch's tip instead of the preset's
#                  VLLM_BRANCH. Ephemeral: the preset file is NOT rewritten
#                  (use for e.g. building main without changing the release
#                  line the preset tracks).
#   --push         push the unverbraucht tags (default: build + tag only)
#
# Requires: git (HTTPS remotes), docker buildx, the preset file, and the
# requirements pinset dir. Uses the LRZ APT mirror unless VLLM_APT_MIRROR set.
set -eo pipefail

PRESET=""
BRANCH_OVERRIDE=""
PUSH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --push)         PUSH=1 ;;
    --branch)       BRANCH_OVERRIDE="${2:?--branch needs a name}"; shift ;;
    -h|--help)      sed -n '2,14p' "$0"; exit 0 ;;
    -*)             echo "unknown option: $1" >&2; exit 2 ;;
    *)              PRESET="$1" ;;
  esac
  shift
done
[ -n "$PRESET" ] || { echo "usage: bump-build-push.vllm.sh <preset> [--branch NAME] [--push]" >&2; exit 2; }

cd "$(dirname "$0")"

# --- load current preset values ---
. "./${PRESET}"
# env.sh provides the image name and other defaults (same as build.vllm.sh).
# Note: env.sh is not set -u clean (reads REPO_GIT_REF/\$1 before setting),
# so this script uses set -e only, like build.vllm.sh.
source ../env.sh "vllm-v2" "pytorch"

if [ -n "$BRANCH_OVERRIDE" ]; then
  echo "== branch override: $VLLM_BRANCH -> $BRANCH_OVERRIDE (ephemeral, preset not rewritten)"
  VLLM_BRANCH="$BRANCH_OVERRIDE"
  EPHEMERAL=1
else
  EPHEMERAL=0
fi
export VLLM_BRANCH

REQ_DIR="build-context/requirements/$(dirname "$VLLM_EXTRA_REQUIREMENTS")"
PINSET_PREFIX="$(dirname "$VLLM_EXTRA_REQUIREMENTS")"
OLD_COMMIT="$VLLM_COMMIT"

NEW_TIP=$(git ls-remote "$VLLM_REPO" "refs/heads/$VLLM_BRANCH" | awk '{print $1}')
if [ -z "$NEW_TIP" ]; then
  echo "ERROR: could not resolve $VLLM_REPO refs/heads/$VLLM_BRANCH" >&2; exit 1
fi
NEW_SHORT="${NEW_TIP:0:10}"

echo "== current: VLLM_BRANCH=$VLLM_BRANCH VLLM_COMMIT=$VLLM_COMMIT"
echo "== tip:     $NEW_TIP (short $NEW_SHORT)"

# --- apply the bump (persistently for the preset's own branch, else just in-memory) ---
if [ "$NEW_SHORT" != "$VLLM_COMMIT" ]; then
  echo "== bumping VLLM_COMMIT $VLLM_COMMIT -> $NEW_SHORT"
  if [ "$EPHEMERAL" = "0" ]; then
    sed -i "s/^export VLLM_COMMIT=.*/export VLLM_COMMIT=\"${NEW_SHORT}\"/" "$PRESET"
    VLLM_COMMIT="$NEW_SHORT"
    export VLLM_COMMIT
    # re-source so VLLM_PRESET_NAME / VLLM_EXTRA_REQUIREMENTS pick up the new commit
    . "./$PRESET"
  else
    VLLM_COMMIT="$NEW_SHORT"
    export VLLM_COMMIT
    VLLM_PRESET_NAME="${VLLM_VERSION}-${VLLM_COMMIT}-rocm-${VLLM_ROCM_VERSION}-kintegrated"
    VLLM_EXTRA_REQUIREMENTS="${PINSET_PREFIX}/${VLLM_COMMIT}.txt"
    export VLLM_PRESET_NAME VLLM_EXTRA_REQUIREMENTS
  fi
  # copy req pinset under the new commit name (content unchanged unless deps changed)
  if [ ! -f "${REQ_DIR}/${NEW_SHORT}.txt" ]; then
    cp "${REQ_DIR}/${OLD_COMMIT}.txt" "${REQ_DIR}/${NEW_SHORT}.txt"
    echo "  (copied req pinset ${OLD_COMMIT}.txt -> ${NEW_SHORT}.txt)"
  fi
  echo "== now: VLLM_COMMIT=$VLLM_COMMIT  image tag: ${VLLM_PRESET_NAME}"
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
TAGS=(
  "unverbraucht/vllm-gfx906:${VLLM_VERSION}-rocm-${VLLM_ROCM_VERSION}"
  "unverbraucht/vllm-gfx906:${VLLM_VERSION}-${SHORT}"
)
for T in "${TAGS[@]}"; do
  docker tag "${VLLM_IMAGE}:${VLLM_PRESET_NAME}" "$T"
  echo "== tagged $T"
  if [ "$PUSH" = "1" ]; then
    docker push "$T" && echo "  pushed $T"
  else
    echo "  (not pushing; use --push)"
  fi
done

echo "== done (commit=${SHORT}, branch=${VLLM_BRANCH})"
[ "$EPHEMERAL" = "1" ] && echo "   note: --branch override was ephemeral; commit the bump manually if this line becomes canonical"
