"""Zero-shot multiple-choice scoring by next-token log-likelihood of the answer letter.

For a prompt ending in "Answer:", we take log-softmax over the full vocabulary at the last
position and read the log-probabilities of " A", " B", " C", " D". Prediction = argmax.
"""
import time

import numpy as np
import torch

from .prompts import LETTERS


def letter_token_ids(tokenizer, prefix: str = " ") -> list[int]:
    ids = []
    for letter in LETTERS:
        toks = tokenizer.encode(prefix + letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"{prefix + letter!r} is {len(toks)} tokens for this tokenizer: {toks}")
        ids.append(toks[0])
    return ids


def _input_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def score_mcq(model, tokenizer, prompts: list[str], batch_size: int = 4, max_length: int = 2048,
              letter_prefix: str = " ", add_special_tokens: bool = True) -> dict:
    """Returns {"logprobs": [n,4], "logprobs_space": [n,4], "logprobs_nospace": [n,4], "nonfinite": [n],
    "n_tokens": [n], "seconds": float}. `logprobs` uses the letter variant given by `letter_prefix`;
    both variants are kept as diagnostics (probability mass on the answer letters)."""
    space_ids = torch.tensor(letter_token_ids(tokenizer, " "))
    nospace_ids = torch.tensor(letter_token_ids(tokenizer, ""))
    tokenizer.padding_side = "left"  # last position = last real token for every row
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _input_device(model)
    out_sp, out_ns, out_bad, out_len = [], [], [], []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=add_special_tokens).to(device)
        # explicit positions so left padding does not shift them
        position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=position_ids).logits[:, -1, :].float()
        bad = ~torch.isfinite(logits).all(dim=-1)
        lsm = torch.log_softmax(logits, dim=-1)
        out_sp.append(lsm[:, space_ids.to(logits.device)].cpu().numpy())
        out_ns.append(lsm[:, nospace_ids.to(logits.device)].cpu().numpy())
        out_bad.append(bad.cpu().numpy())
        out_len.append(enc["attention_mask"].sum(dim=1).cpu().numpy())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sp = np.concatenate(out_sp).astype(np.float32)
    ns = np.concatenate(out_ns).astype(np.float32)
    return {
        "logprobs": sp if letter_prefix == " " else ns,
        "logprobs_space": sp,
        "logprobs_nospace": ns,
        "nonfinite": np.concatenate(out_bad),
        "n_tokens": np.concatenate(out_len),
        "seconds": time.perf_counter() - t0,
    }


def compare_runs(ref: dict, test: dict) -> dict:
    """Agreement of a test precision against a reference (e.g. fp16 vs fp32) on the same prompts."""
    a, b = ref["logprobs"], test["logprobs"]
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    pa, pb = np.exp(a[ok]), np.exp(b[ok])
    pa, pb = pa / pa.sum(1, keepdims=True), pb / pb.sum(1, keepdims=True)  # renormalise over 4 letters
    kl = (pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum(1)
    return {
        "n": int(len(a)),
        "n_nonfinite_test": int((~np.isfinite(b).all(1) | test["nonfinite"]).sum()),
        "argmax_agreement": float((a[ok].argmax(1) == b[ok].argmax(1)).mean()) if ok.any() else float("nan"),
        "max_abs_logprob_diff": float(np.abs(a[ok] - b[ok]).max()) if ok.any() else float("nan"),
        "mean_kl_4way": float(kl.mean()) if ok.any() else float("nan"),
    }


def letter_mass(logprobs) -> np.ndarray:
    """Total next-token probability on the 4 answer letters, per item."""
    return np.exp(np.asarray(logprobs, dtype=np.float64)).sum(1)
