"""The main experiment grid: cells, run order (blocks), and item loading.

A cell = (model, fmt, bench, variety, tier). Every cell is scored and saved on its own, so runs are resumable and
partial results are usable. Blocks follow the agreed priority order (2026-10-07):
  B1 core Belebele   : eng/MSA/Egyptian × {ref, nf4} × 3 models
  B2 core DA-MMLU s50: ENG/MSA/EGY      × {ref, nf4} × 3 models
  B3 int8 core       : core varieties of both benchmarks (DA-MMLU s50) × int8
  B4 other varieties : remaining Belebele + DA-MMLU s50 varieties × {ref, nf4, int8}
  B5 GGUF            : all Belebele + DA-MMLU s50 varieties × gguf_q4km
  B6 DA-MMLU rest    : the other 1,535 DA-MMLU items × all varieties × all formats
RQ2 (fixes) and embeddings run in separate notebooks after the grid.
"""
from dataclasses import asdict, dataclass

from .data import (BELEBELE_VARIETIES, DAMMLU_VARIETIES, load_belebele_aligned, load_dialectal_mmlu_aligned,
                   stratified_subset)

# Reference precision and compute dtype per model, fixed by the sanity runs (reports/sanity_findings.md,
# reports/followup_findings.md): the Gemma family is NaN in fp16 on T4; Qwen3.5 fp16 == fp32.
MODEL_SETUP = {
    "gemma3-4b": {"ref": "fp32", "compute": "fp32"},
    "nilechat-4b": {"ref": "fp32", "compute": "fp32"},
    "qwen3.5-4b": {"ref": "fp16", "compute": "fp16"},
}
MODELS = list(MODEL_SETUP)
CORE_BELEBELE = ["eng_Latn", "arb_Arab", "arz_Arab"]
CORE_DAMMLU = ["ENG", "MSA", "EGY"]
OTHER_BELEBELE = [v for v in BELEBELE_VARIETIES if v not in CORE_BELEBELE]
OTHER_DAMMLU = [v for v in DAMMLU_VARIETIES if v not in CORE_DAMMLU]
PROMPT_FORMAT = "raw"            # decided in follow-up 02 (thinking-off formats lowered letter mass ~100x)
DAMMLU_SUBSET = {"per_domain": 50, "seed": 13}
FORMAT_DEV_N = 100               # first 100 sorted Belebele uids were used to choose the prompt format


@dataclass(frozen=True)
class Cell:
    model: str
    fmt: str        # ref | int8 | nf4 | gguf_q4km
    bench: str      # belebele | dammlu
    variety: str
    tier: str       # all (belebele) | s50 | rest (dammlu)

    @property
    def key(self) -> str:
        return f"{self.model}/{self.fmt}/{self.bench}__{self.variety}__{self.tier}"

    def precision(self) -> str:
        return MODEL_SETUP[self.model]["ref"] if self.fmt == "ref" else self.fmt

    def compute_dtype(self) -> str:
        return MODEL_SETUP[self.model]["compute"]

    def to_dict(self) -> dict:
        return asdict(self)


def _cells(fmts, bench, varieties, tier):
    return [Cell(m, f, bench, v, tier) for m in MODELS for f in fmts for v in varieties]


def blocks() -> list[tuple[str, str, list[Cell]]]:
    """(block id, description, cells) in run order. Cells are grouped by (model, fmt) so each model loads once per block."""
    return [
        ("B1", "core Belebele: eng/arb/arz x {ref, nf4}", _cells(["ref", "nf4"], "belebele", CORE_BELEBELE, "all")),
        ("B2", "core DA-MMLU s50: ENG/MSA/EGY x {ref, nf4}", _cells(["ref", "nf4"], "dammlu", CORE_DAMMLU, "s50")),
        ("B3", "int8 on core varieties (Belebele + DA-MMLU s50)",
         _cells(["int8"], "belebele", CORE_BELEBELE, "all") + _cells(["int8"], "dammlu", CORE_DAMMLU, "s50")),
        ("B4", "other varieties x {ref, nf4, int8} (Belebele + DA-MMLU s50)",
         _cells(["ref", "nf4", "int8"], "belebele", OTHER_BELEBELE, "all")
         + _cells(["ref", "nf4", "int8"], "dammlu", OTHER_DAMMLU, "s50")),
        ("B5", "GGUF Q4_K_M: all varieties (Belebele + DA-MMLU s50)",
         _cells(["gguf_q4km"], "belebele", BELEBELE_VARIETIES, "all")
         + _cells(["gguf_q4km"], "dammlu", DAMMLU_VARIETIES, "s50")),
        ("B6", "DA-MMLU remaining 1,535 items: all varieties x all formats",
         _cells(["ref", "nf4", "int8", "gguf_q4km"], "dammlu", DAMMLU_VARIETIES, "rest")),
    ]


def all_cells() -> list[Cell]:
    return [c for _, _, cs in blocks() for c in cs]


class ItemStore:
    """Loads each benchmark once; returns the aligned items of a cell (same uids, same order, every variety)."""

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self._bele = self._dm = self._s50 = None

    def belebele(self):
        if self._bele is None:
            self._bele = load_belebele_aligned(BELEBELE_VARIETIES, cache_dir=self.cache_dir)
        return self._bele

    def dammlu(self):
        if self._dm is None:
            self._dm = load_dialectal_mmlu_aligned(DAMMLU_VARIETIES, cache_dir=self.cache_dir)
            self._s50 = stratified_subset(self._dm["ENG"], **DAMMLU_SUBSET)
        return self._dm

    def format_dev_uids(self) -> set[str]:
        return {it["uid"] for it in self.belebele()["eng_Latn"][:FORMAT_DEV_N]}

    def items(self, cell: Cell) -> list[dict]:
        if cell.bench == "belebele":
            return self.belebele()[cell.variety]
        rows = self.dammlu()[cell.variety]
        if cell.tier == "s50":
            return [it for it in rows if it["uid"] in self._s50]
        if cell.tier == "rest":
            return [it for it in rows if it["uid"] not in self._s50]
        raise ValueError(f"unknown DA-MMLU tier: {cell.tier}")
