"""Name registry, config loading and recursive instantiation.

Every configurable component registers itself under a dotted name
(``model.gat_encoder``); a config node ``{"class": "model.gat_encoder", ...}``
is turned into an object by :func:`instantiate`, recursively, so nested
components are built before their parents.
"""
from __future__ import annotations

import importlib
import os
import re
from typing import Any, Callable, Dict, Mapping, MutableMapping

import yaml

#: Environment variable naming the directory that holds the fermentation data
#: files and the flux corpus. Defaults to the synthetic demo data.
DATA_DIR_ENV = "MNM_DATA_DIR"
DEFAULT_DATA_DIR = os.path.join("data", "synthetic")


class Registry:
    """Flat name -> object table."""

    table: Dict[str, Any] = {}

    def __new__(cls):  # pragma: no cover
        raise TypeError("Registry is not instantiable")

    @classmethod
    def register(cls, name: str) -> Callable[[Any], Any]:
        def wrapper(obj: Any) -> Any:
            if name in cls.table:
                raise KeyError(f"`{name}` is already registered by {cls.table[name]}")
            cls.table[name] = obj
            return obj

        return wrapper

    @classmethod
    def get(cls, name: str) -> Any:
        if name not in cls.table:
            raise KeyError(f"`{name}` is not registered (known: {sorted(cls.table)})")
        return cls.table[name]


def _resolve_class(class_path: str) -> Any:
    if class_path in Registry.table:
        return Registry.table[class_path]
    module_name, attr = class_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attr)


def instantiate(config: Any) -> Any:
    """Build objects from a config tree.

    A mapping with a ``class`` key becomes ``cls(**kwargs)`` with every kwarg
    instantiated first; lists and plain mappings are traversed; anything else
    is returned unchanged.
    """
    if isinstance(config, Mapping) and "class" in config:
        cls = _resolve_class(config["class"])
        kwargs = {k: instantiate(v) for k, v in config.items() if k != "class"}
        return cls(**kwargs)
    if isinstance(config, list):
        return [instantiate(v) for v in config]
    if isinstance(config, tuple):
        return tuple(instantiate(v) for v in config)
    if isinstance(config, MutableMapping):
        return {k: instantiate(v) for k, v in config.items()}
    return config


def build_from_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Instantiate every top-level entry (``dataset``, ``task``, ...)."""
    return {k: instantiate(v) for k, v in config.items()}


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _variable(name: str) -> str:
    if name == DATA_DIR_ENV:
        return os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR)
    value = os.environ.get(name)
    if value is None:
        raise KeyError(f"Config variable ${{{name}}} is not set")
    return value


def resolve_variables(config: Any) -> Any:
    """Replace ``${VAR}`` placeholders in every string of a config tree.

    ``${MNM_DATA_DIR}`` falls back to ``data/synthetic`` when unset; any other
    variable must be present in the environment.
    """
    if isinstance(config, str):
        return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: _variable(m.group(1)), config)
    if isinstance(config, dict):
        return {k: resolve_variables(v) for k, v in config.items()}
    if isinstance(config, list):
        return [resolve_variables(v) for v in config]
    return config


def set_by_dotted_key(cfg: Dict[str, Any], key: str, value: Any) -> None:
    """``set_by_dotted_key(cfg, "task.model.use_film", True)``."""
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value
