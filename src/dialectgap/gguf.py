"""Letter-log-likelihood scoring for GGUF models via llama-cpp-python (same method as scoring.py).

Token ids come from the model's Hugging Face tokenizer (GGUF conversion keeps the vocabulary ids), so the HF
and GGUF runs see exactly the same input ids and score exactly the same letter tokens. Any difference between
them is then due to the weights/kernels (what RQ3 measures), not to tokenizer differences.
"""
import ctypes
import time

import numpy as np

from .scoring import letter_token_ids


def load_gguf(path: str, n_gpu_layers: int = -1, n_ctx: int = 2048, n_threads=None):
    from llama_cpp import Llama
    return Llama(model_path=path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, n_threads=n_threads,
                 logits_all=False, verbose=False)


def _last_logits(llm) -> np.ndarray:
    """Logits of the last evaluated token.

    `llm.scores` is NOT filled when logits_all=False (recent llama-cpp-python), so read the context directly.
    """
    import llama_cpp
    ptr = llama_cpp.llama_get_logits_ith(llm.ctx, -1)
    return np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(llm.n_vocab(),)).astype(np.float64)


def check_vocab(llm, tokenizer, ids) -> None:
    """The GGUF vocabulary must map the letter ids to the same strings as the HF tokenizer."""
    for i in ids:
        a = llm.detokenize([i]).decode("utf-8", errors="replace").strip()
        b = tokenizer.decode([i]).strip()
        if a != b:
            raise ValueError(f"GGUF/HF vocab mismatch for id {i}: {a!r} vs {b!r}")


def score_mcq_gguf(llm, tokenizer, prompts: list[str], letter_prefix: str = " ",
                   add_special_tokens: bool = True) -> dict:
    ids = letter_token_ids(tokenizer, letter_prefix)
    check_vocab(llm, tokenizer, ids)
    out_lp, out_len = [], []
    t0 = time.perf_counter()
    for p in prompts:
        toks = tokenizer(p, add_special_tokens=add_special_tokens)["input_ids"]
        llm.reset()
        llm.eval(toks)
        logits = _last_logits(llm)
        if not np.isfinite(logits).all() or np.ptp(logits) == 0:
            raise RuntimeError("degenerate logits (non-finite or all equal): scoring would be meaningless")
        logz = logits.max() + np.log(np.exp(logits - logits.max()).sum())
        out_lp.append((logits[ids] - logz).astype(np.float32))
        out_len.append(len(toks))
    lp = np.stack(out_lp)
    return {"logprobs": lp, "nonfinite": ~np.isfinite(lp).all(1),
            "n_tokens": np.asarray(out_len), "seconds": time.perf_counter() - t0}
