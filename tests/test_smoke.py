import random

import numpy as np

from dialectgap import paths
from dialectgap.config import config_hash, load_config
from dialectgap.seed import set_seed
from dialectgap.tracking import data_version, log_run


def test_seed_is_deterministic():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_config_inheritance_and_overrides(tmp_path):
    (tmp_path / "exp.yaml").write_text(
        f"base: {(paths.CONFIGS / 'base.yaml').as_posix()}\neval:\n  batch_size: 4\n", encoding="utf-8")
    cfg = load_config(tmp_path / "exp.yaml", overrides=["seed=7"])
    assert cfg["eval"]["batch_size"] == 4
    assert cfg["eval"]["ci"] == 0.95        # inherited
    assert cfg["seed"] == 7                  # overridden
    assert config_hash(cfg) == config_hash(dict(cfg))


def test_log_run_writes_both_logs(tmp_path):
    split = tmp_path / "train.jsonl"
    split.write_text('{"id": 1, "text": "ازيك"}\n', encoding="utf-8")
    md, jl = tmp_path / "RESULTS.md", tmp_path / "runs.jsonl"
    rec = log_run("smoke", {"seed": 1}, {"cer": 0.1}, data_files=[split],
                  results_md=md, runs_jsonl=jl)
    assert rec["data_version"] == data_version([split])
    assert "smoke" in md.read_text(encoding="utf-8")
    assert jl.read_text(encoding="utf-8").count("\n") == 1


def test_load_hf_token_rejects_non_hf_value(monkeypatch):
    import pytest
    from dialectgap.env import load_hf_token
    monkeypatch.setenv("HF_TOKEN", "not-a-token")
    with pytest.raises(ValueError):
        load_hf_token()
    monkeypatch.delenv("HF_TOKEN")
    assert load_hf_token() is None  # no Kaggle client locally, no env var -> None, no crash
