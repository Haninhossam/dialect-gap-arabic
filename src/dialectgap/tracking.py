"""Append every run to RESULTS.md (human view) and results/runs.jsonl (full record)."""
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from . import paths
from .config import config_hash


def data_version(files) -> str:
    """sha256 over the contents of the given split files (order-independent by name)."""
    h = hashlib.sha256()
    for f in sorted(Path(p) for p in files):
        h.update(f.name.encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:10]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.ROOT,
                             capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=paths.ROOT,
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _pip_freeze() -> list[str]:
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.splitlines()
    except Exception:
        return []


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def log_run(name: str, cfg: dict, metrics: dict, data_files=(), notes: str = "",
            results_md=None, runs_jsonl=None) -> dict:
    results_md = Path(results_md or paths.RESULTS_MD)
    runs_jsonl = Path(runs_jsonl or paths.RUNS_JSONL)
    record = {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "run": name,
        "config": cfg.get("_config_path", "?"),
        "config_hash": config_hash(cfg),
        "seed": cfg.get("seed"),
        "data_version": data_version(data_files) if data_files else "n/a",
        "git": _git_commit(),
        "metrics": metrics,
        "notes": notes,
        "resolved_config": cfg,
        "pip_freeze": _pip_freeze(),
    }
    runs_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    metric_str = ", ".join(f"{k}={_fmt(v)}" for k, v in metrics.items())
    row = (f"| {record['date'][:10]} | {name} | `{record['config']}` ({record['config_hash']}) "
           f"| {record['seed']} | {record['data_version']} | {record['git']} | {metric_str} | {notes} |\n")
    with open(results_md, "a", encoding="utf-8") as f:
        f.write(row)
    return record
