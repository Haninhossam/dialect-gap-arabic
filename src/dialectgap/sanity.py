"""Precision sanity check against an fp32 reference on the same prompts.

Two different checks (refined after run 1, see reports/sanity_findings.md):
  * Reference / compute precisions (fp16, bf16 vs fp32): STABLE  <=>  no non-finite logits
      AND argmax agreement with fp32 >= 95%  AND  mean 4-way KL < 0.01   (pre-registered rule)
  * Quantized formats (int8, nf4, GGUF): VALID  <=>  all logits finite  AND  median letter mass >= 0.05
      AND every greedy generation non-empty. Their distance from fp32 is the measured outcome (RQ3),
      not an instability.
"""
import time

import numpy as np
import torch

from .models import free, load_llm
from .prompts import FORMATS, format_prompt
from .scoring import compare_runs, letter_mass, score_mcq

STABLE_MIN_AGREEMENT = 0.95
STABLE_MAX_KL = 0.01
VALID_MIN_LETTER_MASS = 0.05


def _peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return float("nan")
    return sum(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9


def _reset_peak():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


@torch.no_grad()
def _greedy(model, tok, prompt: str, n: int = 24, add_special_tokens: bool = True) -> str:
    enc = tok(prompt, return_tensors="pt", add_special_tokens=add_special_tokens).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_precision(model_id: str, precision: str, prompts: list[str], gen_prompts: list[str],
                  compute_dtype: str = "fp16", batch_size: int = 4, fmt: str = "raw",
                  check_batching: bool = True) -> dict:
    """Load once, score single + batched, time a batched pass, greedy-generate. Never raises.
    `prompts` are raw MCQ prompts; they are wrapped with prompt format `fmt` after the tokenizer loads."""
    rec = {"model": model_id, "precision": precision, "compute_dtype": compute_dtype, "format": fmt}
    kw = FORMATS[fmt]
    model = None
    try:
        _reset_peak()
        t0 = time.perf_counter()
        model, tok, cls = load_llm(model_id, precision, compute_dtype=compute_dtype)
        rec.update(loader=cls, load_seconds=time.perf_counter() - t0)
        fp = [format_prompt(p, tok, fmt) for p in prompts]
        gp = [format_prompt(p, tok, fmt) for p in gen_prompts]
        score_mcq(model, tok, fp[:batch_size], batch_size=batch_size, **kw)  # warm-up
        batched = score_mcq(model, tok, fp, batch_size=batch_size, **kw)
        if check_batching:  # one-at-a-time scoring is the reference; batched must match it
            single = score_mcq(model, tok, fp, batch_size=1, **kw)
            rec["batch_vs_single"] = compare_runs(single, batched)
        else:
            single = batched
        rec["single"] = single
        rec["tokens_per_second_batched"] = float(batched["n_tokens"].sum() / batched["seconds"])
        rec["generations"] = [_greedy(model, tok, p, add_special_tokens=kw["add_special_tokens"]) for p in gp]
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


def validity(rec: dict) -> dict:
    """Validity of a quantized run (does not compare with fp32; that distance is the RQ3 outcome)."""
    if not rec.get("ok"):
        return {"valid": False, "reason": rec.get("error")}
    lp = np.asarray(rec["single"]["logprobs"], dtype=np.float64)
    finite = bool(np.isfinite(lp).all() and not np.asarray(rec["single"]["nonfinite"]).any())
    mass = float(np.median(letter_mass(lp))) if finite else float("nan")
    gens_ok = all(g.strip() for g in rec.get("generations", []))
    valid = finite and mass >= VALID_MIN_LETTER_MASS and gens_ok
    return {"valid": bool(valid), "finite": finite, "median_letter_mass": mass, "generations_nonempty": gens_ok}


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
        for k in ("logprobs", "logprobs_space", "logprobs_nospace"):
            if k in s:
                s[k] = np.asarray(s[k], dtype=np.float32)
        s["nonfinite"] = np.asarray(s["nonfinite"], dtype=bool)
        s["n_tokens"] = np.asarray(s["n_tokens"])
        rec["single"] = s
    return rec
