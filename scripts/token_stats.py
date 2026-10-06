"""Prompt-length statistics per benchmark and variety (input for the compute estimate).

Usage: python scripts/token_stats.py [--tokenizer Qwen/Qwen3.5-4B]
"""
import argparse
import json
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from dialectgap import paths
from dialectgap.data import BELEBELE_VARIETIES, load_belebele, load_dialectal_mmlu
from dialectgap.prompts import build_mcq_prompt


def summarize(lengths):
    a = np.asarray(lengths)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": int(a.max()), "total": int(a.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cache = str(paths.RAW)
    out = {"tokenizer": args.tokenizer, "belebele": {}, "dialectal_mmlu": {}}

    for v in BELEBELE_VARIETIES:
        items = load_belebele(v, cache_dir=cache)
        out["belebele"][v] = summarize([len(tok.encode(build_mcq_prompt(it))) for it in items])

    ds = load_dialectal_mmlu(cache_dir=cache)
    out["dialectal_mmlu_columns"] = ds.column_names
    out["dialectal_mmlu_first_row"] = {k: str(v)[:120] for k, v in ds[0].items()}
    dialect_col = next(c for c in ds.column_names if c.lower() in {"dialect", "language", "lang"})
    out["dialectal_mmlu_counts"] = dict(Counter(ds[dialect_col]))
    # Rough prompt length: all text columns joined (exact template is fixed in Phase 2 after inspection).
    text_cols = [c for c in ds.column_names if isinstance(ds[0][c], (str, list)) and c != dialect_col]
    by_var = {}
    for r in ds:
        text = " ".join(" ".join(map(str, r[c])) if isinstance(r[c], list) else str(r[c]) for c in text_cols)
        by_var.setdefault(r[dialect_col], []).append(len(tok.encode(text)) + 40)  # +40 ≈ instruction/letters
    out["dialectal_mmlu"] = {k: summarize(v) for k, v in by_var.items()}

    dest = paths.RESULTS_DIR / "token_stats.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
