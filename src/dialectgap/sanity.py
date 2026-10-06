"""Precision sanity check: does fp16 (and int8/nf4) agree with an fp32 reference on the same prompts?

Stability rule (fixed before running):
  stable  <=>  no non-finite logits  AND  argmax agreement with fp32 >= 95%  AND  mean 4-way KL < 0.01
"""
import time

import numpy as np
import torch

from .models import free, load_llm
from .scoring import compare_runs, score_mcq

STABLE_MIN_AGREEMENT = 0.95
STABLE_MAX_KL = 0.01


def _peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9


def _reset_peak():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


@torch.no_grad()
def _greedy(model, tok, prompt: str, n: int = 24) -> str:
    enc = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_precision(model_id: str, precision: str, prompts: list[str], gen_prompts: list[str],
                  compute_dtype: str = "fp16", batch_size: int = 4) -> dict:
    """Load once, score single + batched, time a batched pass, greedy-generate. Never raises."""
    rec = {"model": model_id, "precision": precision, "compute_dtype": compute_dtype}
    model = None
    try:
        _reset_peak()
        t0 = time.perf_counter()
        model, tok, cls = load_llm(model_id, precision, compute_dtype=compute_dtype)
        rec.update(loader=cls, load_seconds=time.perf_counter() - t0)
        single = score_mcq(model, tok, prompts, batch_size=1)
        score_mcq(model, tok, prompts[:batch_size], batch_size=batch_size)  # warm-up
        batched = score_mcq(model, tok, prompts, batch_size=batch_size)
        rec["single"] = single
        rec["batch_vs_single"] = compare_runs(single, batched)
        rec["tokens_per_second_batched"] = float(batched["n_tokens"].sum() / batched["seconds"])
        rec["generations"] = [_greedy(model, tok, p) for p in gen_prompts]
        rec["peak_mem_gb"] = _peak_mem_gb()
        rec["ok"] = True
    except Exception as e:  # record and continue with the next configuration
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        del model
        free()
    return rec


def verdict(ref: dict, test: dict) -> dict:
    """Compare a precision run against the fp32 reference run."""
    if not (ref.get("ok") and test.get("ok")):
        return {"stable": False, "reason": test.get("error") or ref.get("error")}
    cmp = compare_runs(ref["single"], test["single"])
    stable = (cmp["n_nonfinite_test"] == 0 and cmp["argmax_agreement"] >= STABLE_MIN_AGREEMENT
              and cmp["mean_kl_4way"] < STABLE_MAX_KL)
    same_text = [a == b for a, b in zip(ref["generations"], test["generations"])]
    return {**cmp, "stable": bool(stable), "greedy_text_identical": same_text}


def to_jsonable(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            out[k] = to_jsonable(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer, np.bool_)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def from_jsonable(rec: dict) -> dict:
    """Inverse of to_jsonable for the parts verdict() needs (so runs can resume from saved JSON)."""
    rec = dict(rec)
    if "single" in rec:
        s = dict(rec["single"])
        s["logprobs"] = np.asarray(s["logprobs"], dtype=np.float32)
        s["nonfinite"] = np.asarray(s["nonfinite"], dtype=bool)
        s["n_tokens"] = np.asarray(s["n_tokens"])
        rec["single"] = s
    return rec
