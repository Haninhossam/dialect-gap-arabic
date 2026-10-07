import json
import time

import numpy as np
import pytest

from dialectgap.prompts import build_mcq_prompt

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _items(n, k=1):
    return [{"uid": f"u{i}", "passage": "The cat sat on the mat. " * (1 + (i * k) % 4), "question": f"Q{i}?",
             "options": ["mat", "roof", "car", "tree"], "answer": i % 4} for i in range(n)]


def _spec(out_dir, cells, deadline=None, fmt="ref", gguf_path=None):
    return {"model_id": TINY, "fmt": fmt, "precision": "fp32", "compute_dtype": "fp32", "prompt_format": "raw",
            "gguf_path": gguf_path, "out_dir": str(out_dir), "deadline_epoch": deadline, "batch_size": 3,
            "cells": [{"key": key, "cell": {"name": key}, "uids": [it["uid"] for it in items],
                       "gold": [it["answer"] for it in items], "prompts": [build_mcq_prompt(it) for it in items]}
                      for key, items in cells]}


@pytest.fixture(scope="module")
def tiny():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(TINY)
        model = transformers.AutoModelForCausalLM.from_pretrained(TINY).eval()
    except OSError:
        pytest.skip("no network")
    return model, tok


def test_run_job_writes_cells_in_input_order(tmp_path, tiny):
    from dialectgap.gridrun import cell_path, is_done, run_job
    from dialectgap.scoring import score_mcq
    model, tok = tiny
    items = _items(7)
    summary = run_job(_spec(tmp_path, [("m/ref/belebele__x__all", items), ("m/ref/belebele__y__all", _items(5, 3))]))
    assert summary["done"] == ["m/ref/belebele__x__all", "m/ref/belebele__y__all"] and not summary["failed"]
    rec = json.loads(cell_path(tmp_path, "m/ref/belebele__x__all").read_text())
    assert rec["ok"] and rec["uids"] == [f"u{i}" for i in range(7)] and rec["valid"] in (True, False)
    # length-sorted batching must return results in the original item order, equal to one-at-a-time scoring
    ref = score_mcq(model, tok, [build_mcq_prompt(it) for it in items], batch_size=1)
    np.testing.assert_allclose(np.array(rec["logprobs"]), ref["logprobs"], atol=1e-4)
    assert is_done([tmp_path], "m/ref/belebele__x__all") and not is_done([tmp_path], "m/ref/other")


def test_run_job_stops_at_deadline(tmp_path, tiny):
    from dialectgap.gridrun import cell_path, run_job
    summary = run_job(_spec(tmp_path, [("a/ref/b__c__all", _items(2))], deadline=time.time() - 1))
    assert summary["stopped_at_deadline"] and not summary["done"]
    assert not cell_path(tmp_path, "a/ref/b__c__all").exists()


def test_launch_job_subprocess(tmp_path, tiny):
    from dialectgap.gridrun import launch_job
    log = tmp_path / "progress.log"
    res = launch_job(_spec(tmp_path / "out", [("m/ref/d__v__s50", _items(3))]), tmp_path / "spec.json",
                     timeout=600, log_path=log)
    assert res["returncode"] == 0, res["stderr_tail"]
    assert res["done"] == ["m/ref/d__v__s50"]
    assert "done=1 failed=0" in log.read_text()
    assert not (tmp_path / "spec.json").exists()


def test_gguf_job_matches_hf(tmp_path, tiny_gguf):
    """The full GGUF job path (gridrun.run_job, backend gguf) must reproduce HF log-probs on an f32 conversion."""
    from dialectgap.gridrun import cell_path, run_job
    from dialectgap.scoring import score_mcq
    path, tok, model = tiny_gguf
    items = _items(4)
    spec = _spec(tmp_path, [("m/gguf_q4km/belebele__x__all", items)], fmt="gguf_q4km", gguf_path=str(path))
    spec["model_id"] = str(path.parent / "hf")   # tokenizer of the converted model
    summary = run_job(spec)
    assert summary["done"] == ["m/gguf_q4km/belebele__x__all"], summary
    rec = json.loads(cell_path(tmp_path, "m/gguf_q4km/belebele__x__all").read_text())
    assert rec["backend"] in ("gguf-cpu", "gguf-gpu") and rec["valid"] in (True, False)
    ref = score_mcq(model, tok, [build_mcq_prompt(it) for it in items], batch_size=1)
    np.testing.assert_allclose(np.array(rec["logprobs"]), ref["logprobs"], atol=1e-4)


def test_model_load_failure_is_recorded_per_cell(tmp_path):
    from dialectgap.gridrun import cell_path, is_done, run_job
    spec = _spec(tmp_path, [("bad/ref/b__v1__all", _items(2)), ("bad/ref/b__v2__all", _items(2))])
    spec["model_id"] = "does-not/exist"
    summary = run_job(spec)
    assert summary["failed"] == ["bad/ref/b__v1__all", "bad/ref/b__v2__all"] and not summary["done"]
    rec = json.loads(cell_path(tmp_path, "bad/ref/b__v1__all").read_text())
    assert rec["ok"] is False and rec["error"].startswith("model load failed")
    assert not is_done([tmp_path], "bad/ref/b__v1__all")      # failed cells are retried
