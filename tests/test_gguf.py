"""GGUF scoring must match HF scoring on an f32 conversion of the same model.

Regression test for the zero-logits bug (llama-cpp-python no longer fills `llm.scores` when logits_all=False).
Needs llama-cpp-python and a llama.cpp checkout (for convert_hf_to_gguf.py): set DIALECTGAP_LLAMACPP to its path.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dialectgap.prompts import build_mcq_prompt

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny_gguf(tmp_path_factory):
    pytest.importorskip("llama_cpp")
    transformers = pytest.importorskip("transformers")
    lc = os.environ.get("DIALECTGAP_LLAMACPP")
    if not lc or not Path(lc, "convert_hf_to_gguf.py").exists():
        pytest.skip("set DIALECTGAP_LLAMACPP to a llama.cpp checkout")
    from huggingface_hub import snapshot_download
    try:
        snap = Path(snapshot_download(TINY))
    except Exception:
        pytest.skip("no network")
    d = tmp_path_factory.mktemp("tiny")
    hf_dir = d / "hf"
    hf_dir.mkdir()
    for f in snap.iterdir():
        if f.is_file():
            shutil.copy(f, hf_dir / f.name)
    cfg = json.loads((hf_dir / "config.json").read_text())
    cfg["pad_token_id"] = None  # the test model ships pad_token_id=-1, which the converter rejects
    (hf_dir / "config.json").write_text(json.dumps(cfg))
    out = d / "tiny-f32.gguf"
    subprocess.run([sys.executable, str(Path(lc, "convert_hf_to_gguf.py")), str(hf_dir), "--outtype", "f32",
                    "--outfile", str(out)], check=True, capture_output=True)
    tok = transformers.AutoTokenizer.from_pretrained(hf_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(hf_dir).eval()
    return out, tok, model


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
