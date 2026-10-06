"""Run each configuration in its own process (scripts/run_one.py) so memory cannot accumulate in the notebook kernel.

If a child is killed (e.g. out of CPU RAM) or times out, the failure is recorded and the notebook continues.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from . import paths
from .sanity import from_jsonable

RUN_ONE = paths.ROOT / "scripts" / "run_one.py"


def mem_available_gb() -> float:
    """MemAvailable from /proc/meminfo (Linux); NaN elsewhere."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def _explain(code: int) -> str:
    if code in (-9, 137):
        return " (killed by the OS: most likely out of CPU RAM)"
    if code in (-11, 139):
        return " (segmentation fault)"
    return ""


def run_isolated(model_id: str, precision: str, prompts: list[str], gen_prompts: list[str], out_path,
                 compute_dtype: str = "fp16", fmt: str = "raw", batch_size: int = 4,
                 check_batching: bool = True, timeout: int = 3 * 3600, log_path=None) -> dict:
    """Run one configuration in a child process; always writes `out_path` and returns the (numpy-restored) record."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    p_file, g_file = out_path.with_suffix(".prompts.json"), out_path.with_suffix(".gen.json")
    p_file.write_text(json.dumps(prompts, ensure_ascii=False), encoding="utf-8")
    g_file.write_text(json.dumps(gen_prompts, ensure_ascii=False), encoding="utf-8")
    cmd = [sys.executable, str(RUN_ONE), "--model", model_id, "--precision", precision, "--prompts", str(p_file),
           "--gen", str(g_file), "--out", str(out_path), "--compute-dtype", compute_dtype, "--fmt", fmt,
           "--batch-size", str(batch_size)] + ([] if check_batching else ["--no-check-batching"])
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        code, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as e:
        code, stderr = None, (e.stderr or b"").decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
    wall = time.perf_counter() - t0
    if code == 0 and out_path.exists():
        rec = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        why = f"timeout after {timeout}s" if code is None else f"child process exited with code {code}{_explain(code)}"
        rec = {"model": model_id, "precision": precision, "compute_dtype": compute_dtype, "format": fmt,
               "ok": False, "error": why, "stderr_tail": stderr[-3000:]}
    rec["wall_seconds"] = wall
    rec["mem_available_gb_after"] = mem_available_gb()
    out_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    for f in (p_file, g_file):
        f.unlink(missing_ok=True)
    if log_path:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {out_path.stem} ok={rec['ok']} wall={wall:.0f}s "
                     f"peak_rss={rec.get('peak_rss_gb', float('nan')):.1f}GB "
                     f"mem_avail_after={rec['mem_available_gb_after']:.1f}GB {rec.get('error', '')[:120]}\n")
    return from_jsonable(rec)
