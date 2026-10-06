"""GPU-hour estimate for the experiment grid.

Token counts are MEASURED (results/token_stats.json). Throughputs are ASSUMED until the sanity notebook
measures them; if results/sanity/summary.csv exists, measured medians per precision replace the assumptions.
Usage: python scripts/estimate_compute.py  -> writes reports/compute_estimate.md
"""
import csv
import json
import math
from statistics import median

from dialectgap import paths

N_LLMS = 3
QUANT_FORMATS = ["fp16", "int8", "nf4", "gguf_q4km"]
ASSUMED_TOK_PER_S = {"fp16": 2000, "int8": 500, "nf4": 1200, "gguf_q4km": 1500}  # batched prefill, 1×T4
LOAD_OVERHEAD_H = 5 / 60          # download + load per (model, format, session)
GGUF_SETUP_H = 0.5                # build llama.cpp once + convert/quantize 4 models
GEN_TOK_PER_S = 200               # batched greedy generation, fp16, 1×T4 (assumed)
DAMMLU_DOMAINS = 32
DAMMLU_VARIETIES = 7
WEEKLY_QUOTA_H = 30
SAFETY = 0.5   # +50%: throughputs are assumed until measured, plus reruns


def throughputs():
    src = {k: ("assumed", v) for k, v in ASSUMED_TOK_PER_S.items()}
    f = paths.RESULTS_DIR / "sanity" / "summary.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        for prec in ("fp16", "int8", "nf4"):
            vals = [float(r["tok_per_s"]) for r in rows if r["precision"] == prec and r["tok_per_s"] not in ("", "nan")]
            if vals:
                src[prec] = ("measured", median(vals))
    return src


def hours(tokens, tps):
    return tokens / tps / 3600


def main():
    ts = json.loads((paths.RESULTS_DIR / "token_stats.json").read_text(encoding="utf-8"))
    bele_tokens = sum(v["total"] for v in ts["belebele"].values())
    dm = ts["dialectal_mmlu"]
    dm_tokens_full = sum(v["total"] for v in dm.values())
    dm_items_full = dm["ENG"]["n"]
    dm_tok_per_item = dm_tokens_full / (dm_items_full * DAMMLU_VARIETIES)
    tp = throughputs()

    def grid_hours(dm_items):
        dm_tokens = dm_tok_per_item * dm_items * DAMMLU_VARIETIES
        per_fmt = {f: N_LLMS * (hours(bele_tokens + dm_tokens, tp[f][1]) + LOAD_OVERHEAD_H) for f in QUANT_FORMATS}
        return per_fmt, sum(per_fmt.values()) + GGUF_SETUP_H

    # Belebele dialect varieties (excluding eng, arb_Arab) used by RQ2 fixes
    dial = [v for v in ts["belebele"] if v not in ("eng_Latn", "arb_Arab", "arb_Latn")]
    dial_tokens = sum(ts["belebele"][v]["total"] for v in dial)
    rq2 = {
        # (a) rewrite dialect passages+questions+options to MSA: ~ as many output tokens as input content
        "rewrite_generation (2 models)": 2 * (0.8 * dial_tokens) / GEN_TOK_PER_S / 3600,
        "score_rewritten fp16+nf4 (2 models)": 2 * (hours(dial_tokens, tp["fp16"][1]) + hours(dial_tokens, tp["nf4"][1])),
        # (b) 3-shot prompts ≈ 4× tokens
        "few_shot_3 fp16+nf4 (2 models)": 2 * (hours(4 * dial_tokens, tp["fp16"][1]) + hours(4 * dial_tokens, tp["nf4"][1])),
        # (c) QLoRA on Dial2MSA-Verified EGY train (9,099 pairs × ~120 tok), 3 seeds, 1 model, ~800 tok/s
        "lora_train 3 seeds (1 model)": 3 * (9099 * 120) / 800 / 3600 + 0.25,
    }
    other = {"embeddings (3 models × fp16,int8 × FLORES devtest)": 0.3,
             "sanity notebook (T4×2 wall-clock)": 1.5}
    rq2_total, other_total = sum(rq2.values()), sum(other.values())

    scenarios = {"Full DA-MMLU (3,135 items)": dm_items_full,
                 "Stratified 50/domain (1,600 items)": 50 * DAMMLU_DOMAINS,
                 "Stratified 30/domain (960 items)": 30 * DAMMLU_DOMAINS}
    lines = ["# Compute estimate (GPU hours, 1×T4 unless noted)", "",
             "Token counts are **measured** (`results/token_stats.json`, Qwen3.5 tokenizer, full prompts).",
             "Throughputs below are " + ", ".join(f"{k}: {v[1]:.0f} tok/s ({v[0]})" for k, v in tp.items()) + ".",
             "**Assumed values are placeholders until the sanity notebook measures them; rerun this script after.**", "",
             f"- Belebele, 8 varieties × 900: {bele_tokens/1e6:.2f}M prompt tokens per (model, format)",
             f"- DialectalArabicMMLU, 7 × 3,135: {dm_tokens_full/1e6:.2f}M prompt tokens per (model, format)", "",
             "## RQ1 + RQ3 grid: 3 LLMs × {fp16, int8, nf4, GGUF Q4_K_M} × all varieties", "",
             "| DA-MMLU size | fp16 | int8 | nf4 | GGUF | grid total | + RQ2 | + other | **total** |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, n in scenarios.items():
        per, tot = grid_hours(n)
        lines.append(f"| {name} | " + " | ".join(f"{per[f]:.1f}" for f in QUANT_FORMATS)
                     + f" | {tot:.1f} | {rq2_total:.1f} | {other_total:.1f} | **{tot + rq2_total + other_total:.1f}** |")
    lines += ["", "## RQ2 and other items", ""] + [f"- {k}: {v:.1f} h" for k, v in {**rq2, **other}.items()]

    lines += ["", "## Paired-CI precision vs sample size",
              "95% CI half-width for a paired accuracy difference ≈ 1.96·sqrt(d/n), d = share of discordant items.", "",
              "| n items | d=0.05 | d=0.10 | d=0.20 | single accuracy at p=0.5 |", "|---|---|---|---|---|"]
    for n in (900, 960, 1600, 3135):
        lines.append(f"| {n} | ±{196*math.sqrt(.05/n):.1f} pt | ±{196*math.sqrt(.10/n):.1f} pt | "
                     f"±{196*math.sqrt(.20/n):.1f} pt | ±{196*math.sqrt(.25/n):.1f} pt |")
    # Quota fit: 30 GPU-h/week (user's Kaggle quota). Safety margin covers throughput uncertainty + reruns.
    lines += ["", f"## Quota fit ({WEEKLY_QUOTA_H:.0f} GPU-h/week, +{int(SAFETY*100)}% safety margin)",
              "Everything except the sanity notebook runs on a single T4. The sanity notebook uses T4×2; "
              "if T4×2 is billed double it costs 2× its wall-clock (shown as the second number).", "",
              "| DA-MMLU size | total (T4×2 billed 1×) | total (billed 2×) | with margin | weeks needed |", "|---|---|---|---|---|"]
    sanity_h = other["sanity notebook (T4×2 wall-clock)"]
    for name, n in scenarios.items():
        _, tot = grid_hours(n)
        t1 = tot + rq2_total + other_total
        t2 = t1 + sanity_h
        m = t2 * (1 + SAFETY)
        lines.append(f"| {name} | {t1:.1f} | {t2:.1f} | {m:.1f} | {math.ceil(m / WEEKLY_QUOTA_H)} |")
    out = paths.REPORTS / "compute_estimate.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
