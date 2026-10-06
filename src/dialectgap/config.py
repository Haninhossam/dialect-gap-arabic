"""YAML configs with optional `base:` inheritance and dotted CLI overrides."""
import copy
import hashlib
import json
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_dotted(cfg: dict, key: str, value) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = value


def load_config(path, overrides: list[str] | None = None) -> dict:
    """Load `path`; a `base:` key (relative path) is loaded first and merged under it.

    overrides: ["train.lr=3e-4", "seed=13"]; values are parsed as YAML scalars.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_cfg = load_config(path.parent / cfg.pop("base"))
        cfg = _deep_merge(base_cfg, cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        _set_dotted(cfg, key, yaml.safe_load(value))
    cfg.setdefault("_config_path", str(path.as_posix()))
    return cfg


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]
