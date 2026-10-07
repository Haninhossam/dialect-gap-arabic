"""Run grid cells: the parent builds a job spec (prompts included); a child process loads the model once and scores
the job's cells one by one, writing each cell's result atomically as soon as it is done.

Cell result file: <out_dir>/<model>/<fmt>/<bench>__<variety>__<tier>.json with item-level log-probabilities, so every
statistic (accuracy, bootstrap CIs, paired tests, held-out subsets) can be recomputed offline.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import paths
from .prompts import FORMATS, build_mcq_prompt, format_prompt
from .scoring import letter_mass

RUN_JOB = paths.ROOT / "scripts" / "run_grid_job.py"
VALID_MIN_LETTER_MASS = 0.05


def cell_path(out_dir, cell_key: str) -> Path:
    return Path(out_dir) / f"{cell_key}.json"


def is_done(out_dirs, cell_key: str) -> bool:
    """A cell is done if any results directory holds a successful record for it."""
    for d in out_dirs:
        p = cell_path(d, cell_key)
        if p.exists():
            try:
                if json.loads(p.read_text(encoding="utf-8")).get("ok"):
                    return True
            except (OSError, ValueError):
                pass
    return False


def make_spec(model_id: str, cell_objs, store, out_dir, prompt_format: str, gguf_path=None,
              deadline_epoch=None, batch_size: int = 8) -> dict:
    """All cells of one (model, fmt): items are resolved here so the child needs no dataset access."""
    first = cell_objs[0]
    cells = []
    for c in cell_objs:
        items = store.items(c)
        cells.append({"key": c.key, "cell": c.to_dict(), "uids": [it["uid"] for it in items],
                      "gold": [it["answer"] for it in items], "prompts": [build_mcq_prompt(it) for it in items]})
    return {"model_id": model_id, "fmt": first.fmt, "precision": first.precision(),
            "compute_dtype": first.compute_dtype(), "prompt_format": prompt_format, "gguf_path": gguf_path,
            "out_dir": str(out_dir), "deadline_epoch": deadline_epoch, "batch_size": batch_size, "cells": cells}


def _versions() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "bitsandbytes", "llama_cpp"):
        try:
            v[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        v["git_commit"] = subprocess.run(["git", "-C", str(paths.ROOT), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return v


def _write_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _score_hf(model, tok, prompts, fmt_kw, batch_size):
    """Score in length-sorted batches (less padding, deterministic order) and return results in input order."""
    from .scoring import score_mcq
    lengths = [len(tok(p, add_special_tokens=fmt_kw["add_special_tokens"])["input_ids"]) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: (lengths[i], i))
    r = score_mcq(model, tok, [prompts[i] for i in order], batch_size=batch_size, **fmt_kw)
    inv = np.empty(len(order), dtype=int)
    inv[order] = np.arange(len(order))
    return {"logprobs": r["logprobs"][inv], "nonfinite": r["nonfinite"][inv], "n_tokens": r["n_tokens"][inv],
            "seconds": r["seconds"]}


def run_job(spec: dict) -> dict:
    """Child-process entry point. Returns a short summary; per-cell files are written as each cell finishes."""
    from transformers import AutoTokenizer
    fmt_kw = FORMATS[spec["prompt_format"]]
    summary = {"done": [], "failed": [], "stopped_at_deadline": False}
    t_load = time.perf_counter()
    versions = _versions()
    try:
        if spec["fmt"] == "gguf_q4km":
            from .gguf import load_gguf, score_mcq_gguf
            tok = AutoTokenizer.from_pretrained(spec["model_id"])
            model = load_gguf(spec["gguf_path"], n_gpu_layers=-1, n_ctx=4096, n_threads=os.cpu_count())
            import llama_cpp
            backend = "gguf-gpu" if llama_cpp.llama_supports_gpu_offload() else "gguf-cpu"
        else:
            from .models import load_llm
            model, tok, _ = load_llm(spec["model_id"], spec["precision"], compute_dtype=spec["compute_dtype"])
            backend = "hf"
    except Exception as e:  # record the load failure on every cell of the job (retried next session)
        err = f"model load failed: {type(e).__name__}: {e}"[:2000]
        for c in spec["cells"]:
            _write_atomic(cell_path(spec["out_dir"], c["key"]),
                          {"cell": c["cell"], "key": c["key"], "model_id": spec["model_id"], "precision": spec["precision"],
                           "ok": False, "error": err, "versions": versions,
                           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            summary["failed"].append(c["key"])
        return summary
    load_seconds = time.perf_counter() - t_load
    for c in spec["cells"]:
        if spec.get("deadline_epoch") and time.time() > spec["deadline_epoch"]:
            summary["stopped_at_deadline"] = True
            break
        rec = {"cell": c["cell"], "key": c["key"], "model_id": spec["model_id"], "precision": spec["precision"],
               "compute_dtype": spec["compute_dtype"], "prompt_format": spec["prompt_format"], "backend": backend,
               "uids": c["uids"], "gold": c["gold"], "versions": versions, "load_seconds": load_seconds,
               "finished_at": None}
        try:
            prompts = [format_prompt(p, tok, spec["prompt_format"]) for p in c["prompts"]]
            if backend == "hf":
                r = _score_hf(model, tok, prompts, fmt_kw, spec["batch_size"])
            else:
                r = score_mcq_gguf(model, tok, prompts, **fmt_kw)
            lp = np.asarray(r["logprobs"], dtype=np.float64)
            finite = bool(np.isfinite(lp).all())
            mass = float(np.median(letter_mass(lp))) if finite else float("nan")
            gold = np.asarray(c["gold"])
            rec.update(ok=True, logprobs=lp.round(6).tolist(), n_tokens=np.asarray(r["n_tokens"]).tolist(),
                       seconds=float(r["seconds"]), tok_per_s=float(np.sum(r["n_tokens"]) / r["seconds"]),
                       finite=finite, median_letter_mass=mass,
                       valid=bool(finite and mass >= VALID_MIN_LETTER_MASS),
                       accuracy=float((lp.argmax(1) == gold).mean()) if finite else float("nan"))
            summary["done"].append(c["key"])
        except Exception as e:
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:2000])
            summary["failed"].append(c["key"])
        rec["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _write_atomic(cell_path(spec["out_dir"], c["key"]), rec)
    return summary


def launch_job(spec: dict, spec_path, timeout: int, log_path=None) -> dict:
    """Run one job in a child process; a killed or timed-out child is reported, never raised."""
    Path(spec_path).write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, str(RUN_JOB), str(spec_path)], capture_output=True, text=True,
                           timeout=timeout)
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        code, out, err = None, "", str(e)
    res = {"returncode": code, "wall_seconds": time.perf_counter() - t0, "stderr_tail": err[-3000:]}
    try:
        res.update(json.loads(out.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        pass
    Path(spec_path).unlink(missing_ok=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {spec['model_id']} {spec['fmt']} rc={code} "
                     f"wall={res['wall_seconds']:.0f}s done={len(res.get('done', []))} failed={len(res.get('failed', []))} "
                     f"deadline_stop={res.get('stopped_at_deadline')} peak_rss={res.get('peak_rss_gb', float('nan')):.1f}GB\n")
    return res
