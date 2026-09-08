"""scikit-learn API shims.

The project pins ``scikit-learn>=1.3`` so it also runs on installs that predate
the 1.8 regularisation rewrite, in which ``penalty=`` was deprecated and is
removed in 1.10.  Rather than repeating a version check at every call site, the
LogisticRegression kwargs are built here once.
"""

from __future__ import annotations


def l1_ratio_kwargs(l1_ratio: float, solver: str) -> dict[str, object]:
    """Return LogisticRegression regularisation kwargs for the installed release.

    Newer releases select the regulariser with ``l1_ratio`` alone.  Older
    releases require an explicit ``penalty`` and reject ``l1_ratio`` at the
    pure L1 and L2 ends for some solvers.
    """

    from sklearn import __version__ as sklearn_version

    parts = tuple(int(part) for part in sklearn_version.split(".")[:2])
    if parts >= (1, 8):
        return {"l1_ratio": l1_ratio, "solver": solver}
    if l1_ratio >= 1.0:
        penalty = "elasticnet" if solver == "saga" else "l1"
    elif l1_ratio <= 0.0:
        penalty = "l2"
    else:
        penalty = "elasticnet"
    return {"penalty": penalty, "l1_ratio": l1_ratio, "solver": solver}


def has_new_logistic_api() -> bool:
    """True on sklearn releases where ``penalty`` is deprecated."""

    from sklearn import __version__ as sklearn_version

    parts = tuple(int(part) for part in sklearn_version.split(".")[:2])
    return parts >= (1, 8)
