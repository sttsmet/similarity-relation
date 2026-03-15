from __future__ import annotations

from pathlib import Path
from typing import Any
import copy

import yaml



def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result



def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.pop("extends", None)
    if extends:
        parent_path = (path.parent / extends).resolve() if not Path(extends).is_absolute() else Path(extends)
        base_cfg = load_yaml_config(parent_path)
        return _deep_merge(base_cfg, cfg)
    return cfg
