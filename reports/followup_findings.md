# Sanity follow-up (notebook 02, run 3): findings (2026-10-07)

Raw files: `results/followup/` (`progress.log` = one line per configuration). Earlier invalid attempt: `results/followup/_run1_invalid/`.
Every model configuration ran in its own process. CPU RAM peaked at ≤ 9.9 GB per configuration, with ~31 GB still available
after each one, so the earlier killed session was RAM accumulating inside one kernel, which is now solved.

## A. Gemma family, bitsandbytes with fp32 compute (24 sanity prompts, vs run-1 fp32 reference)

| model | format | valid | letter mass | argmax agree vs fp32 | 4-way KL | acc (n=24) | tok/s | peak GPU GB |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b | int8 (fp32 compute) | yes | 0.990 | 0.875 | 0.338 | 0.875 | 816 | 9.3 |
| gemma3-4b | NF4 (fp32 compute) | yes | 0.987 | 0.958 | 0.135 | 0.875 | 371 | 7.5 |
| nilechat-4b | int8 (fp32 compute) | yes | 0.942 | 1.000 | 0.006 | 0.958 | 780 | 8.8 |
| nilechat-4b | NF4 (fp32 compute) | yes | 0.962 | 1.000 | 0.046 | 0.958 | 316 | 7.2 |

All four are valid with coherent greedy text. This confirms the run-1 collapse came from **fp16 compute**, not from quantization.
**Decision:** quantized Gemma-family runs use fp32 compute. The agreement and KL values above come from n=24, so they are anecdotal.

## B. Prompt format for Qwen3.5-4B (fp16, FORMAT-DEV-300 = 100 uids × eng/arb/arz)

| format | letter mass (median) | p10 | acc all | eng | arb | arz | argmax agree with raw |
|---|---|---|---|---|---|---|---|
| **raw** | **0.246** | 0.161 | **0.840** | 0.93 | 0.84 | 0.75 | — |
| chat_nothink | 0.0025 | 1e-5 | 0.480 | 0.81 | 0.31 | 0.32 | 0.51 |
| raw_prefill_nothink | 0.0005 | 2e-5 | 0.553 | 0.87 | 0.40 | 0.39 | 0.60 |

With thinking disabled, Qwen3.5 does not answer with a letter. It starts an Arabic explanation ("بناءً على النص المقدم…"), so letter
mass falls ~100× and accuracy collapses, worst on Arabic. The pre-agreed criterion ("use it if mass rises substantially without hurting
accuracy") is not met. **Decision: keep the `raw` format (lm-eval style) for all models.** This also makes one protocol for all
three models, so the Gemma-family format runs (which hit the GPU-memory issue below) are no longer needed for that decision.

Not a result yet: the per-variety numbers on these 300 FORMAT-DEV items (eng 0.93 > arb 0.84 > arz 0.75) are a first glimpse of the
dialect gap on a small subset that was used for a protocol decision. Main results come only from the full runs.

**GPU out-of-memory (all fp32 runs in B).** Scoring computed logits for every position ([8 × ~300 × ~250k] fp32 ≈ 2 GB). Only the last
position is needed. Fixed with `logits_to_keep=1` (identical last-position logits: max |Δ| = 6e-8 on the test model).

## C. GGUF Q4_K_M (llama.cpp), CPU only, 8 English sanity prompts

| model | size | valid | letter mass | argmax agree vs fp32 | KL | sec/prompt (Kaggle CPU) | tok/s |
|---|---|---|---|---|---|---|---|
| gemma3-4b | 2.49 GB | yes | 0.993 | 1.000 | 0.0000 | 7.7 | 26 |
| nilechat-4b | 2.49 GB | yes | 0.922 | 1.000 | 0.012 | 7.5 | 27 |
| qwen3.5-4b | 2.78 GB | yes | 0.238 | 0.875 | 0.016 | 9.1 | 22 |

The fixed GGUF scoring is valid on all three real models, and llama.cpp handles Qwen3.5. **GPU offload did not run.** The prebuilt
CUDA 12.4 wheel installed fine, but its import check failed only because `diskcache` was installed after the check (fixed). Kaggle has
CUDA 12.8 and no driver stub, and the CUDA source build failed.
CPU speed (~25 tok/s) makes the full GGUF grid infeasible on CPU (4.66M tokens per model ≈ 50 h). GGUF needs GPU offload, or else a
subsample, with the CPU used only for edge-latency measurements.

## D. flash-linear-attention for Qwen3.5
Installed fine, but no speed-up: 657 tok/s vs ~850 tok/s without it (same 300 prompts, fp16, batch 8). `causal_conv1d` is still missing.
**Decision:** not used.

## Measured throughput so far (tok/s, Kaggle T4×2, `device_map="auto"`)

| model | reference | int8 | NF4 | GGUF CPU |
|---|---|---|---|---|
| gemma3-4b | fp32: 441 (run 1, n=24) | 816 (fp32 compute) | 371 (fp32 compute) | 26 |
| nilechat-4b | fp32: 369 (run 1, n=24) | 780 (fp32 compute) | 316 (fp32 compute) | 27 |
| qwen3.5-4b | fp16: ~850 (n=300) | 1107 (run 1, n=24) | 1080 (run 1, n=24) | 22 |

These come from small prompt sets with different batch sizes, so they are planning numbers only. `logits_to_keep=1` should make them somewhat higher.
