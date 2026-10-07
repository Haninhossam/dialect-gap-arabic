# Main grid: run order and estimated GPU hours

Week-1 budget: 21.6 h (23h38m quota left minus 2 h safety). Gates + session setup counted first: 0.35 h.

| block | content | cells | prompt tokens | est. hours | cumulative | fits week 1? |
|---|---|---|---|---|---|---|
| B1 | core Belebele: eng/arb/arz x {ref, nf4} | 18 | 3.27M | 2.3 | 2.6 | yes |
| B2 | core DA-MMLU s50: ENG/MSA/EGY x {ref, nf4} | 18 | 3.46M | 2.4 | 5.0 | yes |
| B3 | int8 on core varieties (Belebele + DA-MMLU s50) | 18 | 3.36M | 1.2 | 6.2 | yes |
| B4 | other varieties x {ref, nf4, int8} (Belebele + DA-MMLU s50) | 81 | 17.22M | 9.1 | 15.3 | yes |
| B5 | GGUF Q4_K_M: all varieties (Belebele + DA-MMLU s50) | 45 | 9.10M | 3.2 | 18.5 | yes |
| B6 | DA-MMLU remaining 1,535 items: all varieties x all formats | 84 | 16.31M | 7.9 | 26.4 | no |

Throughputs used (tok/s):

- gemma3-4b ref: 441 (measured fp32, run 1)
- gemma3-4b int8: 816 (measured, fp32 compute, run 3)
- gemma3-4b nf4: 371 (measured, fp32 compute, run 3)
- nilechat-4b ref: 369 (measured fp32, run 1)
- nilechat-4b int8: 780 (measured, fp32 compute, run 3)
- nilechat-4b nf4: 316 (measured, fp32 compute, run 3)
- qwen3.5-4b ref: 850 (measured fp16, run 3)
- qwen3.5-4b int8: 1107 (measured, run 1)
- qwen3.5-4b nf4: 1080 (measured, run 1)
- GGUF Q4_K_M on GPU: 1000 (ASSUMED (GPU offload not yet measured))

Token counts per cell are exact for the Qwen3.5 tokenizer; other tokenizers differ somewhat. Throughputs come from small prompt sets, and `logits_to_keep=1` (added after they were measured) should make them a little faster. Treat the hours as planning estimates; the notebook re-measures and stops at its budget.
