# The Arabic Dialect Gap in Small Language Models

**Status: work in progress (Phase 0, repo skeleton).** No results yet.

## Research questions
- **RQ1:** How much do small open LLMs and multilingual embedding models degrade from English → MSA → Arabic dialects on human-translated parallel benchmarks?
- **RQ2:** Which cheap fix recovers most of the gap: rewriting dialect input into MSA, few-shot dialect prompting, or light LoRA on licensed dialect data?
- **RQ3:** Does 8-bit / 4-bit quantization hurt dialects more than MSA and English (accuracy, memory, latency)?

## Repository layout
```
data/{raw,interim,processed}   data (not committed; provenance in data/README.md)
src/dialectgap/                library code (seeding, configs, run tracking, ...)
configs/                       one YAML per experiment (inherits configs/base.yaml)
notebooks/                     Kaggle/Colab-ready notebooks
results/                       runs.jsonl (full run records), per-item predictions
reports/                       write-up and figures
RESULTS.md                     human-readable log of every run
```

## Setup
```bash
pip install -r requirements.txt
pip install -e .
python -m pytest
```
On Kaggle/Colab, see `notebooks/00_kaggle_setup.ipynb`.

## Reproducibility
- Fixed seeds (`dialectgap.seed.set_seed`); every experiment is driven by a config file.
- Every run is logged to `RESULTS.md` with date, config hash, data version hash and git commit.
- Per-item predictions are saved so all paired statistics can be recomputed.

## License
Code: MIT. Datasets keep their original licenses (see `data/README.md`).
