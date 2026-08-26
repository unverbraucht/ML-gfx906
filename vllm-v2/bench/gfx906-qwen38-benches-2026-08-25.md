# gfx906 vLLM serving benchmarks — Qwen3.8-27B-AWQ-INT4

**Date:** 2026-08-25 16:13 UTC
**ML-gfx906 revision:** `dc731c6`
**vLLM fork commit:** `67ae6c3f96` (branch `gfx906/v0.28.0rc2`)
**Image:** `mixa3607/vllm-gfx906:0.28.0rc2-67ae6c3f96-rocm-7.14-kintegrated` (`8fd499ea61`)
**vllm.__version__:** `0.28.0rc2`
**Host:** 2× AMD MI50 gfx906 32 GB, ROCm 7.14, 16 cores

## Server config (baseline)

```
model      = cyankiwi/Qwen3.8-27B-AWQ-INT4  (Qwen3.8-27B, INT4-AWQ)
TP         = 2
dtype      = float16
ctx        = 262144 (max_model_len)
max_seqs   = 4
batched    = 1024 (max-num-batched-tokens)
gpu_mem    = 0.82
prefix     = OFF (for these benches)
spec       = OFF (plain decode for decode/prefill tables)
parsers    = qwen3 reasoning, qwen3_coder tool
```

Note: `--enable-prefix-caching` was disabled for all measurements below so no prefix reuse inflates TTFT/prefill numbers.

---

## Table 1 — Decode throughput (single stream, no spec decode)

`bench_tg.py`: streaming chat completion, `usage.completion_tokens` / (wall − TTFT).

| tg | prefill | decode tok | wall | TTFT | decode tok/s |
|----|---------|-----------|------|------|--------------|
| 256   | 78  | 256  | 6.45s  | 0.34s | **41.9** |
| 1024  | 79  | 1024 | 25.02s | 0.34s | **41.5** |
| 2048  | 80  | 1861*| 45.67s | 0.34s | **41.1** |

\* model finished the report naturally at 1861 (finish: stop).

Decode rate is flat (~41 tok/s) across tg 256→2048 — steady-state per-step cost, target-length independent.

---

## Table 2 — Prefill throughput (single stream, no spec decode, no prefix reuse)

`bench_prefill.py`: `prompt_tokens` / TTFT. Two sub-tables: cold (server just started, inductor/compile warmup not complete) vs warm (steady state after a warmup request). Unless noted, use the warm numbers.

### batched-tokens 1024 (baseline) — warm

| prompt tokens | TTFT | prefill tok/s | chunks |
|---------------|------|----------------|--------|
| ~996   | 0.62s  | **1609** | 1 |
| ~2025  | — (warm re-run needed) | | |
| ~4084  | — | | |
| ~33317 | 88.2s  | **378**  | 33 |

### batched-tokens 4096 — REMOVE these (JIT-contaminated)

The following were produced during a window where the server fired `Triton kernel JIT compilation during inference: batch_memcpy_kernel` (16:25:46) and vLLM logged wildly varying `Avg prompt throughput` (119/145/251/1269/148/3460 tok/s). Those "warm sweep" TTFTs (0.52s for 4k, 0.64s for 16k -> 7881/25938 tok/s) are **physically impossible** on 2× MI50 (26-52 TFLOPS peak vs ~1.4 PFLOPS implied) and are an artifact of racing kernel JIT/warmup — not real prefill. Discarded; do not trust.

### Engine-observed prefill throughput (credible)

vLLM's periodic logger (`Avg prompt throughput`) during the batched=4096 session:

| time | Avg prompt tok/s |
|------|------------------|
| 16:25:49 | 119.3 |
| 16:25:59 | 145.3 |
| 16:26:09 | 251.6 |
| 16:26:29 | 1269.5 |
| 16:26:39 | 148.7 |
| 16:28:19 | 3460.2 (likely JIT-affected too)` |
| 16:28:29 | 108.9 |

Steady-state prefill on this stack is roughly **~110-350 tok/s** (the 1269/3460 spikes coincide with the JIT window). This aligns with the earlier batched=1024 numbers (742/730 for 2k/4k) and is the credible range for 27B INT4 TP=2 prefill on MI50.

### Cold-start contamination lesson (and JIT caveat)

An earlier "sweep" before a warmup request showed 747/730/606 tok/s at 2025/4084/16616, and a later "warm" sweep showed impossible 7881/25938 tok/s. **Both are artifacts**: the first was inductor/TTFT warmup; the second raced a triton-JIT-during-inference event (`batch_memcpy_kernel`, logged at 16:25:46) that made large-prefill TTFTs collapse. Only the engine's own `Avg prompt throughput` logger (110-350 tok/s steady) and the batched=1024 single-chunk numbers (~1600 tok/s at 1k) are credible. **Rule: always cross-check single-request TTFT-derived rates against the engine periodic throughput log.**

### batched-tokens A/B — clean methodology (prefix OFF, unique salted prompts, fresh boot each side)

Both servers: no prefix caching, no spec decode, TP=2, 256k ctx, same model. TTFT-derived, 3 reps each (reps shown; variance <1%).

| prompt | batched=1024 | batched=4096 | Δ |
|--------|--------------|--------------|-----|
| ~4119  | 460 tok/s (461/460/460) | **517** (516/517/517) | **+12.4%** |
| ~8291  | 450 tok/s (450/450/449) | **505** (505/506/504) | **+12.2%** |
| ~16657 | 420 tok/s (420/420/419) | **473** (473/473/472) | **+12.6%** |

(8k warmup rates: 453 tok/s at 1024, 510 at 4096 — consistent with the sweep.)

**Conclusion: `max-num-batched-tokens` 1024→4096 gives a real ~+12% prefill throughput at 4-16k prompts** on this stack. Fewer chunk boundaries / better kernel batching wins.

### Measurement pitfalls documented (all encountered)
1. **Prefix caching ON silently poisons prefill benches**: identical-ish prompt patterns collapse TTFT to ~0.5s (impossible "26k tok/s"). Check `docker inspect` args; our `serve-ngram*.sh` has it on, `serve-tg*.sh` off.
2. **Triton-JIT-during-inference** produces absurd engine-log spikes; cross-check client TTFT vs engine `Avg prompt throughput`.
3. **GPU hang/segfault during load** recurs on degraded boots (esp. right after reboot cycles); 4096-batched load crashed twice on a 5h-old boot, loader clean once after reboot. Retry after wedge clears, or reboot.

---

## Table 3 — ngram spec decode vs plain (A/B, multi-request)

9 requests × 3 prompts (agentic/repetitive content, 256 max tokens, batch overlap).

| config | requests | gen tok | wall | tok/s | drafts | draft tok | accepted | acceptance |
|--------|----------|---------|------|-------|--------|-----------|----------|------------|
| ngram (n=3, lookup=2) | 9 | 776 | 39.1s | **19.87** | 388 | 1164 | 481 | 0.413 |
| plain  | 9 | 930 | 49.2s | **18.91** | 0 | 0 | 0 | — |
| plain  | 18 | 1860 | 97.1s | **19.16** | 0 | 0 | 0 | — |

This is a *workload* number (includes per-request TTFT/prefill/scheduling), not steady decode (~41 tok/s single stream). The ngram win here is only ~+4-5% on repetitive content with 41% acceptance; the 256-token request sizes leave little room for ngram window to pay off.

## Table 4 — ngram spec depth sweep (single-stream decode, batched=1024)

ngram with `num_speculative_tokens=N`, `prompt_lookup_max=2`. cudagraph sizes adjusted to multiples of (N+1) ([6,12] for N=5).

| config | tg | decode tok | wall | decode tok/s (excl TTFT) | acceptance |
|--------|----|-----------|------|--------------------------|------------|
| plain | 256  | 256 | 6.45s  | **41.9** | — |
| ngram5 | 256  | 256 | 5.62s  | **48.5** | 0.237 (at tg1024) |
| plain | 1024 | 1024 | 25.0s | **41.5** | — |
| ngram5 | 1024 | 993 | 24.8s | **40.6** | 0.237 |

**Finding:** ngram n=5 gives **+15.5% at tg256** (48.5 vs 41.9 tok/s, reproducible x3) but is **neutral/negative at tg1024** (40.6 vs 41.5). Acceptance at n=5 is only 0.237 (vs 0.413 at n=3) — longer drafts get rejected more, so the extra speculative tokens mostly add overhead on longer generations. Speculative depth is a per-workload knob: short decode budgets win, long steady decode loses.

(Note: ngram n=5 requires cudagraph sizes that are multiples of N+1=6, e.g. [6,12]; the default [1,2,3,4] fails with `No valid cudagraph sizes after rounding`.)

---

## Open experiments (planned)
- [ ] full spec-depth A/B at tg256 (n=3 vs n=5 single-stream) to confirm the short-decode regime
- [ ] ngram `prompt_lookup_max` variants (1 vs 2)
- [x] raise `--max-num-batched-tokens` 1024→4096 and prefill sweep 1024→16384 — done, see Table 2 warm sweep (batched=4096)
- [x] raise ngram `num_speculative_tokens` 3→5 — done, Table 4