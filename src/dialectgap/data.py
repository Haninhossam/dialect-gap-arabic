"""Load benchmarks into one common multiple-choice item format.

Item: {"uid", "variety", "passage" (may be ""), "question", "options" [4], "answer" (0-3)}.
`uid` is shared by the same item across varieties, so all comparisons can be paired.
"""
from datasets import load_dataset

# Belebele config names (verified on the HF dataset, 2026-10-06).
BELEBELE_VARIETIES = [
    "eng_Latn", "arb_Arab", "arz_Arab", "apc_Arab", "ary_Arab", "ars_Arab", "acm_Arab", "arb_Latn",
]


def load_belebele(variety: str, cache_dir=None) -> list[dict]:
    ds = load_dataset("facebook/belebele", variety, split="test", cache_dir=cache_dir)
    items = []
    for r in ds:
        items.append({
            # link + question_number identifies the same question across all varieties
            "uid": f"{r['link']}#{r['question_number']}",
            "variety": variety,
            "passage": r["flores_passage"],
            "question": r["question"],
            "options": [r[f"mc_answer{i}"] for i in range(1, 5)],
            "answer": int(r["correct_answer_num"]) - 1,
        })
    return items


def load_belebele_aligned(varieties=BELEBELE_VARIETIES, cache_dir=None) -> dict[str, list[dict]]:
    """Same items in the same (sorted uid) order for every variety.

    Row order differs between Belebele configs, so pairing must go through `uid`, never position.
    Raises if any variety has a different item set or a different gold answer for the same uid.
    """
    by_var = {v: {it["uid"]: it for it in load_belebele(v, cache_dir=cache_dir)} for v in varieties}
    ref_v = varieties[0]
    ref = by_var[ref_v]
    for v, d in by_var.items():
        if set(d) != set(ref):
            raise ValueError(f"{v}: item set differs from {ref_v}")
        bad = [u for u in ref if d[u]["answer"] != ref[u]["answer"]]
        if bad:
            raise ValueError(f"{v}: {len(bad)} gold answers differ from {ref_v}, e.g. {bad[:3]}")
    order = sorted(ref)
    return {v: [by_var[v][u] for u in order] for v in varieties}


def load_dialectal_mmlu(cache_dir=None):
    """Raw DialectalArabicMMLU test split; column mapping is fixed in Phase 2 after inspection."""
    return load_dataset("MBZUAI/Dialectal-Arabic-MMLU", split="test", cache_dir=cache_dir)
