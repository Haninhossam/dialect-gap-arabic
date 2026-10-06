"""Zero-shot multiple-choice scoring by next-token log-likelihood of the answer letter.

For a prompt ending in "Answer:", we take log-softmax over the full vocabulary at the last
position and read the log-probabilities of " A", " B", " C", " D". Prediction = argmax.
"""
import time

import numpy as np
import torch

from .prompts import LETTERS


def letter_token_ids(tokenizer) -> list[int]:
    ids = []
    for letter in LETTERS:
        toks = tokenizer.encode(" " + letter, add_special_tokens=False)
        if len(toks) != 1:
            raise ValueError(f"' {letter}' is {len(toks)} tokens for this tokenizer: {toks}")
        ids.append(toks[0])
    return ids


def _input_device(model):
    return next(model.parameters()).device


@torch.no_grad()
def score_mcq(model, tokenizer, prompts: list[str], batch_size: int = 4, max_length: int = 2048) -> dict:
    """Returns {"logprobs": [n,4] float32, "nonfinite": [n] bool, "n_tokens": [n] int, "seconds": float}."""
    letter_ids = torch.tensor(letter_token_ids(tokenizer))
    tokenizer.padding_side = "left"  # last position = last real token for every row
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = _input_device(model)
    out_lp, out_bad, out_len = [], [], []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        # explicit positions so left padding does not shift them
        position_ids = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        logits = model(**enc, position_ids=position_ids).logits[:, -1, :].float()
        bad = ~torch.isfinite(logits).all(dim=-1)
        lp = torch.log_softmax(logits, dim=-1)[:, letter_ids.to(logits.device)]
        out_lp.append(lp.cpu().numpy())
        out_bad.append(bad.cpu().numpy())
        out_len.append(enc["attention_mask"].sum(dim=1).cpu().numpy())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "logprobs": np.concatenate(out_lp).astype(np.float32),
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
