# Items that must appear in the write-up (agreed with the PI)

Collected as decisions are made, so nothing is lost when the paper is written. Each item points to its evidence.

1. **fp16 collapse of the Gemma-3 family on T4 (no bf16).** Gemma-3-4B and Nile-Chat-4B return NaN in fp16, and NF4/int8
   with fp16 compute collapse too (empty or garbage text). bf16 and fp32 are exact. With fp32 compute, int8 and NF4 are valid.
   Practical edge finding: on pre-Ampere GPUs, quantized Gemma-3 fails silently unless compute is fp32.
   Evidence: `reports/sanity_findings.md` (run 1), `reports/followup_findings.md` (part A).
2. **Stability-rule refinement (stated openly).** The pre-registered rule (argmax agreement ≥ 95% and KL < 0.01 vs fp32)
   applies to reference/compute precisions. Quantized formats get a validity check (finite logits, letter mass ≥ 0.05,
   non-empty text), and their distance from fp32 is the RQ3 outcome. The refinement was made after seeing run-1 data.
   Evidence: `reports/sanity_findings.md` §1.4.
3. **Negative finding: disabling Qwen3.5 "thinking" hurts MCQ scoring.** With the chat template + `enable_thinking=False`, or
   with a prefilled empty think block, letter mass fell ~100× (0.246 → 0.0025 / 0.0005) and accuracy fell (0.84 → 0.48 / 0.55),
   most on Arabic. The model starts an Arabic explanation instead of a letter. We kept the raw lm-eval-style prompt.
   Evidence: `reports/followup_findings.md` part B.
4. **Protocol-selection leakage control.** The prompt format was chosen on FORMAT-DEV = the first 100 sorted Belebele uids
   (× eng/arb/arz). All Belebele results are reported on all 900 items and on the 800 held-out items.
5. **Same-input guarantee for GGUF.** GGUF runs are fed the HF tokenizer's ids. A GGUF-f32 vs HF-f32 check gives max |Δ logprob| ≈ 4e-6,
   so HF-vs-GGUF differences reflect quantization and kernels, not tokenization.
6. **Reference precision differs by model, by necessity:** fp32 for the Gemma family, fp16 for Qwen3.5 (measured equal to fp32).
   Quantization drops are always measured against each model's own reference.
7. **Calibration-free quantizers only** (bnb int8, bnb NF4, GGUF Q4_K_M): GPTQ/AWQ would need calibration text whose language
   could itself bias which variety degrades.
8. **Scoring caveat:** Qwen3.5 puts only ~0.25 of next-token mass on " A"–" D" in raw format (Gemma ~0.99). Argmax over the four
   letters is the standard lm-eval protocol, but the low mass is reported.
