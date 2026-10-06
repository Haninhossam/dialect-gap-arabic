"""Letter-log-likelihood scoring for GGUF models via llama-cpp-python (same method as scoring.py)."""
import time

import numpy as np

from .prompts import LETTERS


def load_gguf(path: str, n_gpu_layers: int = -1, n_ctx: int = 2048, n_threads=None):
    from llama_cpp import Llama
    return Llama(model_path=path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_threads=n_threads,
                 logits_all=False, verbose=False)


def score_mcq_gguf(llm, prompts: list[str]) -> dict:
    letter_ids = []
    for letter in LETTERS:
        toks = llm.tokenize((" " + letter).encode("utf-8"), add_bos=False, special=False)
        if len(toks) != 1:
            raise ValueError(f"' {letter}' is {len(toks)} tokens: {toks}")
        letter_ids.append(toks[0])
    out_lp, out_len = [], []
    t0 = time.perf_counter()
    for p in prompts:
        toks = llm.tokenize(p.encode("utf-8"), add_bos=True, special=False)
        llm.reset()
        llm.eval(toks)
        logits = np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float64)
        logz = logits.max() + np.log(np.exp(logits - logits.max()).sum())
        out_lp.append((logits[letter_ids] - logz).astype(np.float32))
        out_len.append(len(toks))
    lp = np.stack(out_lp)
    return {"logprobs": lp, "nonfinite": ~np.isfinite(lp).all(1),
            "n_tokens": np.asarray(out_len), "seconds": time.perf_counter() - t0}
