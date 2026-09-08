"""Classical and neural AFQ models.

The deep models need PyTorch, which is an optional extra
(``pip install -e '.[torch]'``).  Importing this package must not require
torch: the classical path, the CLI and the config loader all work without it,
so the torch-dependent names are loaded lazily through ``__getattr__`` and
raise an actionable error only when actually used.
"""

from importlib import import_module
from typing import Any

# Accepted domain-alignment values.  Defined here, torch-free, so experiment
# configuration can validate the flag without importing the training loop.
ALIGNMENTS: tuple[str, ...] = ("none", "coral", "mmd", "dann")

# name -> (module, attribute).  Every target module imports torch at its top.
_LAZY: dict[str, tuple[str, str]] = {
    "DomainAlignedClassifier": ("dit.models.domain_train", "DomainAlignedClassifier"),
    "DomainTrainConfig": ("dit.models.domain_train", "DomainTrainConfig"),
    "search_domain_classifier": ("dit.models.domain_train", "search_domain_classifier"),
    "TractTransformer": ("dit.models.tract_transformer", "TractTransformer"),
    "DomainDiscriminator": ("dit.models.domain_adaptation", "DomainDiscriminator"),
    "coral_loss": ("dit.models.domain_adaptation", "coral_loss"),
    "gradient_reverse": ("dit.models.domain_adaptation", "gradient_reverse"),
    "mmd_rbf_loss": ("dit.models.domain_adaptation", "mmd_rbf_loss"),
}

__all__ = [
    "ALIGNMENTS",
    "DomainAlignedClassifier",
    "DomainDiscriminator",
    "DomainTrainConfig",
    "TractTransformer",
    "coral_loss",
    "gradient_reverse",
    "mmd_rbf_loss",
    "search_domain_classifier",
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"{name} requires PyTorch, which is an optional extra. "
            "Install it with: pip install -e '.[torch]'"
        ) from exc
    value = getattr(module, attribute)
    globals()[name] = value  # cache so the lookup happens once
    return value
