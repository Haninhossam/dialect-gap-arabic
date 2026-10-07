"""Planned run order and estimated GPU hours per block (writes reports/grid_plan.md).

Token counts: exact per cell (real prompts, Qwen3.5 tokenizer; the Gemma tokenizer is gated locally, so it is used as a proxy).
Throughputs: measured in sanity runs 1 and 3 (reports/followup_findings.md); GGUF on GPU is ASSUMED until measured.
Usage: python scripts/plan_grid.py [--budget-hours 21.6]
"""
import argparse
import json
from collections import defaultdict

from transformers import AutoTokenizer

from dialectgap import paths
from dialectgap.grid import ItemStore, blocks
from dialectgap.prompts import build_mcq_prompt

# tok/s on Kaggle T4x2 (device_map="auto"); (value, source)
THROUGHPUT = {
    ("gemma3-4b", "ref"): (441, "measured fp32, run 1"),
    ("gemma3-4b", "int8"): (816, "measured, fp32 compute, run 3"),
    ("gemma3-4b", "nf4"): (371, "measured, fp32 compute, run 3"),
    ("nilechat-4b", "ref"): (369, "measured fp32, run 1"),
    ("nilechat-4b", "int8"): (780, "measured, fp32 compute, run 3"),
    ("nilechat-4b", "nf4"): (316, "measured, fp32 compute, run 3"),
    ("qwen3.5-4b", "ref"): (850, "measured fp16, run 3"),
    ("qwen3.5-4b", "int8"): (1107, "measured, run 1"),
    ("qwen3.5-4b", "nf4"): (1080, "measured, run 1"),
}
GGUF_GPU_TOK_S = (1000, "ASSUMED (GPU offload not yet measured)")
JOB_OVERHEAD_H = 3 / 60          # per (model, fmt) per block: process start, model load, item load
GGUF_CONVERT_H = 10 / 60         # per model per session: download + convert + quantize
GATES_H = 0.25                   # gate 1 (Gemma fp32, 300 prompts) + gate 2 (llama-cpp CUDA wheel, tiny model)
SESSION_SETUP_H = 0.1            # clone + pip per Kaggle session


def token_counts(store, tok):
    counts = {}
    for _, _, cells in blocks():
        for c in cells:
            k = (c.bench, c.variety, c.tier)
            if k not in counts:
                counts[k] = sum(len(tok.encode(build_mcq_prompt(it))) for it in store.items(c))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-hours", type=float, default=23 + 38 / 60 - 2)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    store = ItemStore(cache_dir=str(paths.RAW))
    counts = token_counts(store, tok)

    rows, cum = [], GATES_H + SESSION_SETUP_H
    for bid, desc, cells in blocks():
        hours, jobs, toks = 0.0, set(), 0
        for c in cells:
            n = counts[(c.bench, c.variety, c.tier)]
            tps = GGUF_GPU_TOK_S[0] if c.fmt == "gguf_q4km" else THROUGHPUT[(c.model, c.fmt)][0]
            hours += n / tps / 3600
            toks += n
            jobs.add((c.model, c.fmt))
        hours += len(jobs) * JOB_OVERHEAD_H + (GGUF_CONVERT_H * 3 if bid == "B5" else 0)
        cum += hours
        rows.append((bid, desc, len(cells), toks, hours, cum))

    lines = ["# Main grid: run order and estimated GPU hours", "",
             f"Week-1 budget: {a.budget_hours:.1f} h (23h38m quota left minus 2 h safety). "
             f"Gates + session setup counted first: {GATES_H + SESSION_SETUP_H:.2f} h.", "",
             "| block | content | cells | prompt tokens | est. hours | cumulative | fits week 1? |",
             "|---|---|---|---|---|---|---|"]
    for bid, desc, n, toks, h, c in rows:
        lines.append(f"| {bid} | {desc} | {n} | {toks/1e6:.2f}M | {h:.1f} | {c:.1f} | {'yes' if c <= a.budget_hours else 'no'} |")
    lines += ["", "Throughputs used (tok/s):", ""]
    lines += [f"- {m} {f}: {v} ({src})" for (m, f), (v, src) in THROUGHPUT.items()]
    lines += [f"- GGUF Q4_K_M on GPU: {GGUF_GPU_TOK_S[0]} ({GGUF_GPU_TOK_S[1]})", "",
              "Token counts per cell are exact for the Qwen3.5 tokenizer; other tokenizers differ somewhat. "
              "Throughputs come from small prompt sets, and `logits_to_keep=1` (added after they were measured) should make "
              "them a little faster. Treat the hours as planning estimates; the notebook re-measures and stops at its budget."]
    per_cell = defaultdict(int)
    for (b, v, t), n in counts.items():
        per_cell[f"{b}/{v}/{t}"] = n
    (paths.RESULTS_DIR / "grid_token_counts.json").write_text(json.dumps(per_cell, indent=1), encoding="utf-8")
    out = paths.REPORTS / "grid_plan.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
