"""Child process for one grid job (one model x format, many cells). Usage: python scripts/run_grid_job.py SPEC.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dialectgap.gridrun import run_job  # noqa: E402


def peak_rss_gb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except ImportError:
        return float("nan")


if __name__ == "__main__":
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    summary = run_job(spec)
    summary["peak_rss_gb"] = peak_rss_gb()
    print(json.dumps(summary))
