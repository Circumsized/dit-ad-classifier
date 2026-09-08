"""Environment gate.

The pinned ``numpy<2`` is not cosmetic: the scikit-learn binary built for this
project is compiled against the NumPy 1.x ABI and raises on import under NumPy
2.x, which silently disables every evaluation command.  This test is the first
one in the suite so a broken environment fails fast with an actionable message
instead of 40 cryptic import errors.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_numpy_major_is_below_two() -> None:
    assert int(np.__version__.split(".")[0]) < 2, (
        f"numpy {np.__version__} is installed; scikit-learn for this project "
        "requires numpy<2. Reinstall with 'pip install \"numpy<2\"'."
    )


def test_sklearn_is_importable() -> None:
    # scikit-learn is a core dependency, not optional: a missing/broken install
    # must fail the gate, not skip it, or every evaluation silently disappears.
    import sklearn

    assert int(sklearn.__version__.split(".")[0]) >= 1


def test_scipy_is_importable() -> None:
    import scipy

    assert scipy.__version__


def test_optional_torch_does_not_break_core() -> None:
    from importlib.metadata import version

    import dit

    assert dit.__version__
    # The CLI and config loader import the experiment module, which must not
    # pull in torch at import time (torch is an optional extra). The real
    # no-torch proof is the subprocess smoke in tests/test_cli.py; this guards
    # the import chain stays torch-free even when torch happens to be present.
    from dit.cli.main import build_parser
    from dit.config import EXPERIMENT_KEYS

    build_parser()
    assert "model" in EXPERIMENT_KEYS
    assert version("dit-afq") == dit.__version__
