"""GEM-graph surrogate for fermentation dynamics with metabolic graph pretraining."""
from . import dataset, layers, model, task, train  # noqa: F401  (register components)
from .registry import Registry, build_from_config, instantiate, load_config, resolve_variables  # noqa: F401

__version__ = "1.0.0"
