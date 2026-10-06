import json

import pytest

from dialectgap.prompts import build_mcq_prompt

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
ITEM = {"passage": "The cat sat on the mat.", "question": "Where?", "options": ["mat", "roof", "car", "tree"]}


def test_run_isolated_success_and_log(tmp_path):
    pytest.importorskip("transformers")
    from dialectgap.isolate import run_isolated
    p = build_mcq_prompt(ITEM)
    log = tmp_path / "progress.log"
    rec = run_isolated(TINY, "fp32", [p, p], [p], tmp_path / "rec.json", compute_dtype="fp32",
                       batch_size=2, log_path=log)
    if not rec["ok"] and "Can't load" in rec.get("error", ""):
        pytest.skip("no network")
    assert rec["ok"], rec.get("error") or rec.get("stderr_tail")
    assert rec["single"]["logprobs"].shape == (2, 4)          # numpy restored
    assert json.loads((tmp_path / "rec.json").read_text())["ok"]
    assert "rec ok=True" in log.read_text()
    assert not (tmp_path / "rec.prompts.json").exists()      # temp files cleaned up


def test_run_isolated_records_child_crash(tmp_path):
    from dialectgap.isolate import run_isolated
    # an invalid precision makes the child exit non-zero (argparse) -> must be recorded, not raised
    rec = run_isolated(TINY, "int3", ["x"], [], tmp_path / "bad.json")
    assert rec["ok"] is False
    assert "exited with code" in rec["error"]
    assert "invalid choice" in rec["stderr_tail"]
    assert json.loads((tmp_path / "bad.json").read_text())["ok"] is False
