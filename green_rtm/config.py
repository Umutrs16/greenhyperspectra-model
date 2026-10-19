"""YAML config loader with deep-merge overrides."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# Windows: numpy / MKL / PyTorch can each ship OpenMP. Allow a single process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_update(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(*names: str | Path) -> dict[str, Any]:
    """Load `configs/base.yaml` then apply named overrides in order.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg = load_config("emit_free")
    >>> cfg = load_config("emit_free", "ablations/fixed_nuisance")
    """
    cfg_dir = PROJECT_ROOT / "configs"
    cfg = yaml.safe_load((cfg_dir / "base.yaml").read_text(encoding="utf-8"))
    for name in names:
        path = Path(name)
        if not path.suffix:
            path = cfg_dir / f"{name}.yaml"
        elif not path.is_absolute():
            path = cfg_dir / path
        extra = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cfg = _deep_update(cfg, extra)
    cfg["_root"] = PROJECT_ROOT
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    rel = Path(cfg["paths"][key])
    if rel.is_absolute():
        return rel
    return PROJECT_ROOT / rel


def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
