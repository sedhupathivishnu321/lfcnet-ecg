"""
Typed YAML config loader.

Every hyperparameter used anywhere in src/ must come from config.yaml through
this module - no hardcoded constants inside the pipeline code itself.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigNode:
    """Dict -> attribute-accessible, recursively, read-only-by-convention."""

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigNode(value)
            elif isinstance(value, list):
                value = [ConfigNode(v) if isinstance(v, dict) else v for v in value]
            setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, ConfigNode):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, ConfigNode) else x for x in v]
            else:
                out[k] = v
        return out

    def __repr__(self):
        return f"ConfigNode({self.to_dict()})"


def load_config(path: str | Path = "config.yaml") -> ConfigNode:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ConfigNode(raw)


def enabled_datasets(cfg: ConfigNode):
    """Return the list of dataset config nodes that are enabled (default True)."""
    return [d for d in cfg.data.datasets if getattr(d, "enabled", True)]
