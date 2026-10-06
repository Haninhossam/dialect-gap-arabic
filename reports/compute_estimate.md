# Compute estimate (GPU hours, 1×T4 unless noted)

Token counts are **measured** (`results/token_stats.json`, Qwen3.5 tokenizer, full prompts).
Throughputs below are fp16: 2000 tok/s (assumed), int8: 500 tok/s (assumed), nf4: 1200 tok/s (assumed), gguf_q4km: 1500 tok/s (assumed).
**Assumed values are placeholders until the sanity notebook measures them; rerun this script after.**

- Belebele, 8 varieties × 900: 1.64M prompt tokens per (model, format)
- DialectalArabicMMLU, 7 × 3,135: 3.02M prompt tokens per (model, format)

## RQ1 + RQ3 grid: 4 LLMs × {fp16, int8, nf4, GGUF Q4_K_M} × all varieties

| DA-MMLU size | fp16 | int8 | nf4 | GGUF | grid total | + RQ2 | + other | **total** |
|---|---|---|---|---|---|---|---|---|
| Full DA-MMLU (3,135 items) | 2.9 | 10.7 | 4.7 | 3.8 | 22.6 | 7.2 | 1.8 | **31.6** |
| Stratified 50/domain (1,600 items) | 2.1 | 7.4 | 3.3 | 2.7 | 16.0 | 7.2 | 1.8 | **25.0** |
| Stratified 30/domain (960 items) | 1.8 | 6.0 | 2.7 | 2.2 | 13.2 | 7.2 | 1.8 | **22.2** |

## RQ2 and other items

- rewrite_generation (2 models): 2.2 h
- score_rewritten fp16+nf4 (2 models): 0.7 h
- few_shot_3 fp16+nf4 (2 models): 2.9 h
- lora_train 3 seeds (1 model): 1.4 h
- embeddings (3 models × fp16,int8 × FLORES devtest): 0.3 h
- sanity notebook (T4×2 wall-clock): 1.5 h

## Paired-CI precision vs sample size
95% CI half-width for a paired accuracy difference ≈ 1.96·sqrt(d/n), d = share of discordant items.

| n items | d=0.05 | d=0.10 | d=0.20 | single accuracy at p=0.5 |
|---|---|---|---|---|
| 900 | ±1.5 pt | ±2.1 pt | ±2.9 pt | ±3.3 pt |
| 960 | ±1.4 pt | ±2.0 pt | ±2.8 pt | ±3.2 pt |
| 1600 | ±1.1 pt | ±1.5 pt | ±2.2 pt | ±2.5 pt |
| 3135 | ±0.8 pt | ±1.1 pt | ±1.6 pt | ±1.8 pt |
