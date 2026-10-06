# Follow-up run 1: INVALID (kept for the record, do not use)

Kaggle run of `02_sanity_followup.ipynb` before the commit "Fix three bugs found in follow-up run 1" (see git log):

1. **A, B, D all failed** with `RuntimeError: Invalid device argument`. `torch.cuda.reset_peak_memory_stats()` was called before
   CUDA was initialised. Fixed in `dialectgap/sanity.py` (`torch.cuda.init()` first; memory stats can no longer abort a run).
2. **C (GGUF) numbers are meaningless.** llama-cpp-python 0.3.36 does not fill `llm.scores` when `logits_all=False`, so the
   code read all-zero logits: letter mass = 4/vocab (1.5e-5), argmax always "A", "accuracy" 0.375 = share of gold "A" in the
   8 prompts, identical for all three models. Fixed in `dialectgap/gguf.py` (last-token logits read from the context, plus
   a degenerate-logits guard). Verified: GGUF-f32 vs HF-f32 max |Δ logprob| = 4e-6 (`tests/test_gguf.py`).
3. **Tokenization**: llama.cpp tokenized " A" as 2 tokens for a SentencePiece vocab. GGUF scoring now uses the HF tokenizer's
   ids, so HF and GGUF see identical inputs.

The llama-cpp-python CUDA source build also failed in this run (it fell back to CPU). The rerun tries a prebuilt CUDA wheel
first and saves every failure message to `C_build_log.json`.

Still informative from this run: llama.cpp converted and loaded all three models (including Qwen3.5) as GGUF Q4_K_M
(2.49 / 2.49 / 2.78 GB), and CPU-only prompt processing ran at ~19–25 tok/s on Kaggle's CPU.
