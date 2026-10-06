import numpy as np
import pytest

from dialectgap.prompts import build_mcq_prompt

ITEM = {"uid": "x#1", "variety": "arz_Arab", "passage": "النص", "question": "السؤال؟",
        "options": ["أ", "ب", "ج", "د"], "answer": 0}


def test_prompt_format():
    p = build_mcq_prompt(ITEM)
    assert p.startswith("The following is a multiple-choice reading comprehension question.")
    assert "\nA. أ\nB. ب\nC. ج\nD. د\nAnswer:" in p
    assert p.endswith("Answer:")
    assert "Passage:" not in build_mcq_prompt({**ITEM, "passage": ""})


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        tok = transformers.AutoTokenizer.from_pretrained(name)
        model = transformers.AutoModelForCausalLM.from_pretrained(name).eval()
    except OSError:
        pytest.skip("no network / model unavailable")
    return model, tok


def test_left_padded_batch_matches_single(tiny):
    from dialectgap.scoring import compare_runs, score_mcq
    model, tok = tiny
    prompts = [build_mcq_prompt(ITEM),
               build_mcq_prompt({**ITEM, "passage": "نص أطول بكتير من الأول " * 5})]
    batched = score_mcq(model, tok, prompts, batch_size=2)
    single = score_mcq(model, tok, prompts, batch_size=1)
    assert batched["logprobs"].shape == (2, 4)
    np.testing.assert_allclose(batched["logprobs"], single["logprobs"], atol=1e-4)
    cmp = compare_runs(single, batched)
    assert cmp["argmax_agreement"] == 1.0 and cmp["n_nonfinite_test"] == 0
    assert cmp["mean_kl_4way"] < 1e-6


def test_sanity_run_and_verdict(tiny):
    from dialectgap.sanity import run_precision, to_jsonable, verdict
    import json
    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    prompts = [build_mcq_prompt(ITEM)] * 3
    ref = run_precision(name, "fp32", prompts, prompts[:1], batch_size=2)
    test = run_precision(name, "fp32", prompts, prompts[:1], batch_size=2)
    assert ref["ok"], ref.get("error")
    v = verdict(ref, test)
    assert v["stable"] and v["greedy_text_identical"] == [True]
    from dialectgap.sanity import from_jsonable
    restored = from_jsonable(json.loads(json.dumps(to_jsonable(ref))))
    assert verdict(restored, test)["stable"]
    bad = run_precision("does-not/exist", "fp16", prompts, [])
    assert bad["ok"] is False and "error" in bad


def test_belebele_aligned_pairs_by_uid():
    pytest.importorskip("datasets")
    from dialectgap import paths
    from dialectgap.data import load_belebele_aligned
    try:
        al = load_belebele_aligned(["eng_Latn", "arz_Arab"], cache_dir=str(paths.RAW))
    except Exception as e:  # no network
        pytest.skip(str(e))
    assert [it["uid"] for it in al["eng_Latn"]] == [it["uid"] for it in al["arz_Arab"]]
    assert len(al["arz_Arab"]) == 900
