"""GGUF scoring must match HF scoring on an f32 conversion of the same model.

Regression test for the zero-logits bug (llama-cpp-python no longer fills `llm.scores` when logits_all=False).
Needs llama-cpp-python and a llama.cpp checkout (for convert_hf_to_gguf.py): set DIALECTGAP_LLAMACPP to its path.
"""
import numpy as np
import pytest

from dialectgap.prompts import build_mcq_prompt


def test_gguf_f32_matches_hf(tiny_gguf):
    from dialectgap.gguf import load_gguf, score_mcq_gguf
    from dialectgap.scoring import score_mcq
    path, tok, model = tiny_gguf
    items = [{"passage": "The cat sat on the mat. " * k, "question": "Where did the cat sit?",
              "options": ["mat", "roof", "car", "tree"]} for k in (1, 3)]
    prompts = [build_mcq_prompt(it) for it in items]
    ref = score_mcq(model, tok, prompts, batch_size=1)
    llm = load_gguf(str(path), n_gpu_layers=0, n_ctx=512)
    got = score_mcq_gguf(llm, tok, prompts)
    np.testing.assert_allclose(got["logprobs"], ref["logprobs"], atol=1e-4)
    # guard against the zero-logits bug: uniform logits would give exactly -log(vocab) for every letter
    assert np.ptp(got["logprobs"]) > 1e-3
