"""Repo-relative paths that resolve identically on Windows, Kaggle and Colab."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CONFIGS = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS_MD = ROOT / "RESULTS.md"
RUNS_JSONL = RESULTS_DIR / "runs.jsonl"
