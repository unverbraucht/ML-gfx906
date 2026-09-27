# AGENTS.md

Operating notes for agents (human or AI) working in this repo. This file
captures non-obvious rules that must be followed to keep builds correct and
consistent. Scope of this section: the vLLM images under `vllm-v2/`.

## Docker image builds: use the LRZ apt mirror

The default `archive.ubuntu.com` / `security.ubuntu.com` CDN is unreliable
from this host (transfers stall mid-index; we've seen a single `apt-get
update` crawl for 20+ minutes or hang a build entirely). The LRZ mirror is
~6x faster and stable:

```
http://ubuntu.mirror.lrz.de/ubuntu/
```

For vLLM image builds, always export this so the in-build apt steps use it
(`vllm-v2/vllm.Dockerfile` rewrites both Ubuntu hosts when `APT_MIRROR` is
set; it is a no-op when empty):

```bash
export VLLM_APT_MIRROR="http://ubuntu.mirror.lrz.de/ubuntu/"
. ./preset.<...>.sh && ./build.vllm.sh   # or build-and-push.vllm.sh
```

Keeping it set consistently matters for cache reuse: the mirror rewrite
sits in `rocm_base`, so flipping it on/off invalidates every layer above
that stage. Always build with it on.

## vLLM images: docker tags and the reported version must agree

### Background — there are two independent version surfaces

A built vLLM image carries its version in two unrelated places:

1. **The docker image tag** — built in `build-and-push.vllm.sh` /
   `build.vllm.sh` from `${VLLM_PRESET_NAME}` (set in each `preset.*.sh`),
   plus a git-ref suffix taken from *this* repo. Example:
   `0.26.0-rocm-7.2.1-kintegrated`.
2. **The version vLLM reports at runtime** (`vllm.__version__`) — baked into
   the wheel at build time by **setuptools-scm**, which derives it from **git
   tags** in the cloned vLLM source via `git describe`.

Because these are computed by completely different mechanisms, they drift
apart silently. The failure mode we already hit: the gfx906 forks carry **no
git tags**, and the Dockerfile clones with `--depth 1` (so no tags are fetched
anyway). setuptools-scm then falls back to `0.1.dev1+g<short-sha>`, even
though the docker tag says `0.26.0`. The image runs fine — it just lies about
its own version.

### Rule

The version component of the docker tag and `vllm.__version__` **must match**,
and **must come from a single source of truth**: a `VLLM_VERSION` declared once
in the preset. Do not hand-write the version in two places.

### Required mechanism — set the correct git tag at build time

Per the agreed approach: **the agent must ensure a git tag of the intended
version exists on the pinned commit before the wheel is built**, so that
setuptools-scm derives the right version. The robust way (no push access to
the fork required):

1. Declare the version once in the preset and derive the docker tag from it:
   ```bash
   # preset.0.26.0-rocm-7.2.1-kintegrated.sh
   export VLLM_VERSION="0.26.0"
   export VLLM_PRESET_NAME="${VLLM_VERSION}-rocm-${VLLM_ROCM_VERSION}-kintegrated"
   ```
2. Thread `VLLM_VERSION` as a build arg and tag the checkout in the
   `files_vllm` Dockerfile stage, right after the `git checkout`. The tagging
   is **autodetecting**: the stage first fetches `refs/tags/v${VLLM_VERSION}`
   and only applies the synthetic `git tag -f` when HEAD is not already
   exactly tagged. Since shallow `--branch` clones fetch no tags, untagged
   fork branches (e.g. `gfx906/main` snapshots) always get the synthetic
   tag (e.g. `v0.27.99rc0`), while a branch whose tip carries the real tag
   keeps its natural version. Setting `VLLM_VERSION=""` disables tagging
   entirely (setuptools-scm then derives whatever it can — usually
   `0.1.dev1+g…` on untagged forks).
   ```dockerfile
   ARG VLLM_VERSION=""
   ...
   RUN if [ -n "$VLLM_VERSION" ]; then \
         git fetch --quiet --depth 1 origin "refs/tags/v${VLLM_VERSION}:refs/tags/v${VLLM_VERSION}" 2>/dev/null || true; \
         if ! git describe --tags --exact-match HEAD >/dev/null 2>&1; then \
           git tag -f "v${VLLM_VERSION}"; \
         fi; \
       fi
   ```
   Pass it from the build scripts:
   `--build-arg VLLM_VERSION="${VLLM_VERSION}"`.

   The `.git` directory (tag included) is copied into `build_vllm`, so
   setuptools-scm sees the tag during `pip3 wheel`. With the tag sitting
   exactly on HEAD, setuptools-scm yields the clean version (`0.26.0`) even in
   a shallow clone.

If a tag genuinely cannot be set, the documented escape hatch is
`ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VLLM_VERSION}` in `build_vllm`. Prefer
the tag, though — it is what upstream/the forks actually use, keeps the
commit-hash semantics intact, and makes the version authoritative.

Even better if you have push access to the fork: push a real `vX.Y.Z` tag to
the fork at the pinned commit. Then the local-tag step becomes a harmless
no-op and every consumer gets the same version for free.

### Agent checklist — when adding or building a vLLM target

- [ ] `VLLM_VERSION` is set in the preset and `VLLM_PRESET_NAME` is derived
      from it (not hand-typed separately).
- [ ] A `v${VLLM_VERSION}` tag is present on the pinned `VLLM_COMMIT` (set
      locally in the build, or pushed to the fork).
- [ ] `VLLM_VERSION` is passed as a build arg and used to tag in the Dockerfile.
- [ ] The version actually matches the fork's state — don't label a checkout
      `0.26.0` if it is really an `-rc0` or a different release. Check the fork
      branch name / README before choosing `VLLM_VERSION`.

### Verification — after a successful build

```bash
docker run --rm <image> python3 -c "import vllm; print(vllm.__version__)"
# must print exactly VLLM_VERSION (e.g. 0.26.0)
```
and confirm the leading version component of the image tag equals the same
value. If these two ever disagree, the build is not finished.

### Current status (as of the 0.26.0 port)

The `0.26.0-rocm-7.2.1-kintegrated` image builds and runs correctly on gfx906
(MI60/MI50), but currently reports `0.1.dev1+g91bd31b9e` because the git-tag
mechanism above is **not yet wired** into `vllm-v2/`. Wiring it is a small
change following this note: add `VLLM_VERSION` to the presets, thread it as a
build arg, and add the `git tag` line in `files_vllm`.

## Build memory: the build can OOM-kill the host (2026-09-27)

Symptom: a `buildx` build dies and **takes the agent session with it** — the
kernel log shows a burst of `Out of memory: Killed process …`, including the
build's own `tee` (`journalctl -k | grep -i oom`). This box has 31 GiB RAM and
no swap headroom worth speaking of, so any unbounded parallelism in the build
is a host-level hazard, not just a slow build.

Two independent causes hit us on the same day, both now fixed in
`vllm-v2/vllm.Dockerfile`; keep both invariants if you touch it:

1. **`ARG` scoping silently voids the cap.** `ARG MAX_JOBS=4` declared
   *before* any `FROM` is **not** visible to stage instructions. Without a
   per-stage `ARG MAX_JOBS`, `MAX_JOBS=${MAX_JOBS:-$(nproc)}` expands to
   **`nproc` (16 here)** — so the cap looked present in every review and did
   nothing. Each wheel stage (`build_triton`, `build_fa`, `build_vllm`) must
   re-declare `ARG MAX_JOBS` after its `FROM`.
   *How to check:* the resolved value is printed in the log —
   `#NN [build_triton 5/7] RUN MAX_JOBS=4 pip3 wheel …`. If it shows
   `${MAX_JOBS:-$(nproc)}` or a bare `$(nproc)`, the ARG never arrived.
2. **buildx runs independent stages concurrently.** `build_rust` and
   `build_triton` are both `FROM build_base`, so their peaks *sum* (cargo ×
   `CARGO_BUILD_JOBS` + cmake × `MAX_JOBS`), and C++ TUs in triton's
   `TritonGPUToLLVM` are multi-GiB each. `build_triton` therefore carries a
   dependency-only bind mount
   (`RUN --mount=type=bind,from=build_rust,… true`) that serializes it behind
   `build_rust`; the mount adds no layer content, just the DAG edge.

Current defaults for this host: **`MAX_JOBS=4`**, **`CARGO_BUILD_JOBS=2`**.
Raising either is a memory decision — check `free -g` and expect the worst-case
sum, not the average. Note also that Triton's own build defaults `MAX_JOBS` to
`2 × cpu_count` when the variable is unset, so an unset/voided cap is *worse*
than no cap at all.

(Historical note: the pre-0.30.0 images were built with `MAX_JOBS` effectively
unset — the 3.6.0 fork's triton TUs were small enough to survive it. Stock
Triton 3.8.0 is not.)
