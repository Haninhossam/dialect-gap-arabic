"""Shared fixtures. `tiny_gguf` converts the tiny HF test model to GGUF f32 (needs llama-cpp-python and
DIALECTGAP_LLAMACPP pointing at a llama.cpp checkout; skipped otherwise)."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="session")
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
