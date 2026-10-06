"""Run ONE model × precision × prompt-format configuration in a fresh process and save its record as JSON.

Each configuration gets its own process so that all CPU/GPU memory is returned to the OS when it ends.
Loading many models in one notebook kernel leaked RAM until Kaggle killed the session (follow-up run 2).

Usage:
  python scripts/run_one.py --model ID --precision {fp32,fp16,bf16,int8,nf4} --prompts P.json --out REC.json
                            [--gen G.json] [--compute-dtype fp16] [--fmt raw] [--batch-size 4] [--no-check-batching]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialectgap.prompts import FORMATS  # noqa: E402


def peak_rss_gb() -> float:
    """Peak resident memory of this process (Linux/macOS); NaN where unsupported (Windows)."""
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1e6 if sys.platform != "darwin" else kb / 1e9
    except ImportError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "bf16", "int8", "nf4"])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--gen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compute-dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--fmt", default="raw", choices=sorted(FORMATS))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-check-batching", action="store_true")
    a = ap.parse_args()

    from dialectgap.sanity import run_precision, to_jsonable

    prompts = json.loads(Path(a.prompts).read_text(encoding="utf-8"))
    gen = json.loads(Path(a.gen).read_text(encoding="utf-8")) if a.gen else []
    rec = run_precision(a.model, a.precision, prompts, gen, compute_dtype=a.compute_dtype,
                        batch_size=a.batch_size, fmt=a.fmt, check_batching=not a.no_check_batching)
    rec["peak_rss_gb"] = peak_rss_gb()
    tmp = a.out + ".tmp"
    Path(tmp).write_text(json.dumps(to_jsonable(rec), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, a.out)  # atomic: a half-written file never looks like a finished run
    print(json.dumps({"ok": rec["ok"], "error": rec.get("error", "")[:300], "peak_rss_gb": rec["peak_rss_gb"]}))


if __name__ == "__main__":
    main()
