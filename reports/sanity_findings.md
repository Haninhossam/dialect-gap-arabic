# Precision sanity check: findings (run 1, Kaggle T4×2, 2026-10-06)

Raw files: `results/sanity/` (one JSON per model × precision, `summary.csv`, `embedding_summary.csv`).
Setup: 24 prompts = 8 Belebele questions (paired by uid) × {eng_Latn, arb_Arab, arz_Arab}. Letter log-likelihood scoring;
the fp32 run is the reference. Quantized runs used **fp16 compute**. The GGUF section did not run (old cell; fixed in `1592253`).

## 1. LLMs

| model | precision | finite logits | argmax agree vs fp32 | 4-way KL vs fp32 | letter mass (median) | acc eng / arb / arz (n=8 each) | tok/s | peak GB |
|---|---|---|---|---|---|---|---|---|
| gemma3-4b | fp32 (ref) | 24/24 | — | — | 0.991 | 1.00 / 1.00 / 0.75 | 441 (2 GPUs) | 20.1 |
| gemma3-4b | bf16 | 24/24 | 1.00 | 0.0002 | 0.990 | 1.00 / 1.00 / 0.75 | 285 | 9.5 |
| gemma3-4b | **fp16** | **0/24 (all NaN)** | — | — | — | — | — | 9.5 |
| gemma3-4b | int8 (fp16 compute) | 24/24 | 0.375 | 3.92 | **0.000** | 0.62 / 0.38 / 0.12 | 1547 | 6.1 |
| gemma3-4b | **nf4 (fp16 compute)** | **0/24 (all NaN)** | — | — | — | — | — | 4.3 |
| nilechat-4b | fp32 (ref) | 24/24 | — | — | 0.953 | 1.00 / 1.00 / 0.88 | 369 (2 GPUs) | 18.4 |
| nilechat-4b | bf16 | 24/24 | 1.00 | 0.0003 | 0.953 | 1.00 / 1.00 / 0.88 | 263 | 8.7 |
| nilechat-4b | **fp16** | **0/24 (all NaN)** | — | — | — | — | — | 8.7 |
| nilechat-4b | int8 (fp16 compute) | **4/24** | 0.50 | 1.29 | 0.008 | — | 1604 | 5.6 |
| nilechat-4b | **nf4 (fp16 compute)** | **0/24 (all NaN)** | — | — | — | — | — | 4.1 |
| qwen3.5-4b | fp32 (ref) | 24/24 | — | — | 0.219 | 1.00 / 1.00 / 1.00 | 307 (2 GPUs) | 19.5 |
| qwen3.5-4b | bf16 | 24/24 | 1.00 | 0.0002 | 0.223 | 1.00 / 1.00 / 1.00 | 229 | 9.5 |
| qwen3.5-4b | **fp16** | 24/24 | **1.00** | **0.000003** | 0.219 | 1.00 / 1.00 / 1.00 | 918 | 9.5 |
| qwen3.5-4b | int8 (fp16 compute) | 24/24 | 0.96 | 0.0047 | 0.185 | 1.00 / 1.00 / 0.88 | 1107 | 6.0 |
| qwen3.5-4b | nf4 (fp16 compute) | 24/24 | 0.96 | 0.036 | 0.363 | 1.00 / 1.00 / 0.88 | 1080 | 4.4 |

Greedy generations (2 Egyptian prompts): Gemma/Nile-Chat fp16 and nf4 return empty strings, and int8 returns multilingual gibberish.
fp32 and bf16 return a sensible answer plus explanation. All Qwen3.5 precisions are coherent.

### Interpretation
1. **The Gemma-3 family is numerically broken in fp16 on T4**, and so is anything that computes in fp16 (bnb NF4 and LLM.int8 with fp16 compute). These are overflow failures, **not quantization effects**, and they must never be reported as "quantization hurts dialects".
2. **bf16 matches fp32 exactly** for all three models (argmax 1.00, KL ≤ 0.0003). On T4 it is emulated and ~1.5× slower than fp32 split across 2 GPUs.
3. **Qwen3.5-4B is fp16-safe**: fp16 is indistinguishable from fp32 (KL 3e-6). Its int8/nf4 runs are valid. Their divergence from fp32 (1 of 24 flips; KL 0.005 / 0.036) is the quantity RQ3 measures.
4. **Rule refinement.** The pre-registered stability rule (agreement ≥ 95%, KL < 0.01) fits *reference/compute precisions* (fp16/bf16 vs fp32). Applied to quantized formats, it would label genuine quantization error as "instability" (Qwen nf4). From now on, quantized runs get a **validity** check instead: finite logits, non-collapsed letter mass, coherent greedy text. Their distance from the reference is the measured outcome. This is a definitional fix made after seeing data, and it is stated here openly.
5. **Letter mass.** Qwen3.5 puts only ~0.22 of next-token mass on " A–D", because it wants to open a `<think>` block. Gemma puts ~0.99 there. Argmax over the 4 letters is still the standard lm-eval protocol, but it is a protocol caveat to report.
6. **Batch invariance.** It holds for all fp32/bf16/fp16 references (1.00). For quantized runs it is not exact: Qwen int8 has 1 of 24 flips between batched and single scoring. Main runs will use a fixed, deterministic batching (sorted uid, fixed batch size), so results stay reproducible.

## 2. Embedding models (fp32 vs fp16, 100 aligned Belebele passages, arz → eng)

| model | non-finite | max abs cosine diff | retrieval argmax agree | R@1 fp32 |
|---|---|---|---|---|
| LaBSE | 0 | 0.00075 | 1.00 | 1.00 |
| BGE-M3 | 0 | 0.00082 | 1.00 | 1.00 |
| Qwen3-Embedding-0.6B | 0 | 0.00073 | 1.00 | 1.00 |

fp16 is safe for all three. **Warning:** R@1 = 1.00 on 100 passage-level candidates means this setting is at **ceiling**. The real FLORES evaluation must be harder to discriminate between varieties: sentence-level, all 1,012 devtest candidates, both directions, plus dialect ↔ MSA.

## 3. Throughput note
On T4×2 with `device_map="auto"`, models are split across GPUs (pipeline, not parallel). fp16 Gemma's 2,431 tok/s is meaningless (NaN output). Qwen3.5 is slower than Gemma at equal precision (918 vs 2,431 tok/s in fp16), probably because its Gated-DeltaNet layers fall back to plain PyTorch without the `flash-linear-attention` kernels (to verify).
