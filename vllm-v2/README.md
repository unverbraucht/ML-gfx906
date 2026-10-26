# vLLM GFX906

Used forks:

- https://github.com/ai-infos/vllm-gfx906-mobydick
- https://github.com/ai-infos/triton-gfx906
- https://github.com/ai-infos/flash-attention-gfx906

## Prebuilt images

Current line (built from
[KIntegrated/vllm-gfx906-mobydick](https://github.com/KIntegrated/vllm-gfx906-mobydick),
ROCm 7.14, torch 2.13.0+gfx906, transformers 5.18.0):

- [`docker.io/unverbraucht/vllm-gfx906:0.32.1-rocm-7.14`](https://hub.docker.com/r/unverbraucht/vllm-gfx906/tags) —
  upstream v0.32.0 base + the 0.32.1 patch set; Mamba2 prefill checkpoints
  default OFF (**+21 % KV capacity** on hybrid models)
- `docker.io/unverbraucht/vllm-gfx906:0.32.1-8c0888158d` and
  `:0.32.1-8c0888158d-rocm-7.14-kintegrated` — the same image, commit-pinned
- `:0.32.0-rocm-7.14`, `:0.31.0-rocm-7.14`, `:0.30.0-rocm-7.14` — earlier lines,
  still published

Older upstream images:

- [`docker.io/mixa3607/vllm-gfx906:0.20.1-rocm-6.3.3-aiinfos`](https://hub.docker.com/r/mixa3607/vllm-gfx906/tags)\*
- [`docker.io/mixa3607/vllm-gfx906:0.20.1-rocm-7.2.1-aiinfos`](https://hub.docker.com/r/mixa3607/vllm-gfx906/tags)\*

Details — what is inside, the backend/KV decision table, environment knobs and
per-model numbers — are in `README-docker.md` in this directory.

## Run

## Docker/Kubernetes

https://arkprojects.space/wiki/AMD_GFX906/vllm/run

## Build

See build vars in `./env.sh`. You also may use presets `./preset.*.sh`. Exec `./build-and-push.vllm.sh`:

```bash
$ . preset.0.20.1-rocm-6.3.3-aiinfos.sh
$ ./build-and-push.vllm.sh
```
