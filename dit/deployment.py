"""Final-model deployment: fit on all labeled data, save, predict unlabeled rows.

Cross-validation scores tell you how a procedure performs; they do not give
you a model to submit. This module closes that loop with the same pipeline
machinery the evaluation path uses, so the deployed model differs from the
evaluated one only in *which rows* it fit on — never in structure:

1. ``fit_deployment_model`` freezes one :class:`ExperimentConfig`, runs the
   usual nested search over **all** labeled rows of the task view, and keeps
   the refit-on-everything pipeline together with its provenance.
2. ``save_deployment_artifact`` / ``load_deployment_artifact`` persist that
   bundle with the label map, feature metadata, config snapshot and data
   digest needed to audit a prediction later.
3. ``predict_unlabeled`` rebuilds the feature matrix with the artifact's
   frozen view settings, validates the column contract against the stored
   metadata, and returns calibrated probabilities plus argmax predictions.

The deep Transformer is refused here for now: its artifact needs torch
weights and mask semantics that this classical-first closure does not carry.

Trust boundary: a pickle is executable data. Loading is hardened two ways —
:class:`_ArtifactUnpickler` only resolves classes from the modules the
pipeline can legitimately reference, and every save writes a SHA-256 sidecar
that the load verifies before unpickling. Neither is a security boundary:
allowlisted roots contain code-executing globals a REDUCE opcode can invoke
(verified: ``numpy.testing._private.utils.runstring`` executes arbitrary
Python at load time), and anyone able to write the artifact can recompute the
unkeyed sidecar. ponytail: module-root allowlist as a raise-the-bar measure;
upgrade to exact ``(module, name)`` pairs or the skops format (no code
execution on load) if artifacts must ever be accepted from other parties.
Treat artifacts like data files and keep them under the same access control
as the dataset they were fitted on.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dit.data.schema import DatasetBundle
from dit.evaluation.experiment import DEEP_MODELS, ExperimentConfig, prepare_matrix
from dit.models.classical import make_search_estimator

ARTIFACT_VERSION = 1


@dataclass(frozen=True)
class DeploymentModel:
    """A fitted pipeline plus everything needed to audit its predictions."""

    pipeline: Any
    config: ExperimentConfig
    label_map: dict[int, str]
    feature_names: tuple[str, ...]
    data_digest: str
    best_params: dict[str, Any]
    classes_: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "pipeline": self.pipeline,
            "config": self.config.describe(),
            "label_map": {str(key): value for key, value in self.label_map.items()},
            "feature_names": list(self.feature_names),
            "data_digest": self.data_digest,
            "best_params": {key: _plain(value) for key, value in self.best_params.items()},
            "classes": self.classes_.tolist(),
        }


def _plain(value: Any) -> Any:
    """Convert numpy scalars left inside best_params into JSON-safe natives."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def fit_deployment_model(
    dataset: DatasetBundle,
    config: ExperimentConfig,
) -> DeploymentModel:
    """Tune and refit the frozen configuration on every labeled row.

    The search here is the same inner machinery the evaluation path uses —
    fold-local preprocessing, the same scorer, the same grid — but there is no
    outer split: every labeled row of the task view takes part in the final
    refit. Hyperparameters are therefore tuned *for deployment* on all
    available labels, which is standard practice precisely because their
    selection score is no longer an unbiased performance estimate; that
    estimate comes from the cross-validation reports, not from this fit.
    """

    if config.model.lower() in DEEP_MODELS:
        raise ValueError(
            "deployment artifacts currently support classical models only; "
            f"{config.model!r} is a deep model"
        )
    if config.model.lower() == "ensemble":
        raise ValueError("deploy a concrete base model rather than the ensemble wrapper")
    if config.control_view is not None:
        raise ValueError("control views are diagnostic inputs, not deployable models")

    view = dataset.task_view(config.task)
    if view.y is None:
        raise ValueError("fitting a deployment model requires labels")

    features, _, meta = prepare_matrix(view, config)
    layout = meta["model_layout"]
    control_names = meta.get("control_names")

    selector = None
    if config.enable_selection:
        from dit.data.selection import SparseBlockSelector

        selector = SparseBlockSelector(
            layout,
            top_blocks=config.top_blocks,
            keep_fraction=config.keep_fraction,
            max_features=config.max_features,
            seed=config.seed,
        )

    from sklearn.model_selection import StratifiedKFold

    counts = np.unique(view.y, return_counts=True)[1]
    usable = min(config.inner_splits, int(np.min(counts)))
    if usable < 2:
        raise ValueError("not enough samples per class for the deployment inner search")
    cv = StratifiedKFold(n_splits=usable, shuffle=True, random_state=config.seed)
    search = make_search_estimator(
        config.model,
        cv=cv,
        seed=config.seed,
        scoring=config.selection_metric,
        n_jobs=config.n_jobs,
        selector=selector,
        residualize=config.covariate_strategy == "residualize",
    )
    search.fit(features, view.y)

    names = (
        control_names
        if control_names is not None
        else tuple(layout.feature_names())
    )
    return DeploymentModel(
        pipeline=search.best_estimator_,
        config=config,
        label_map=view.label_map,
        feature_names=tuple(names),
        data_digest=view.data_digest,
        best_params=dict(search.best_params_),
        classes_=np.asarray(search.classes_, dtype=int),
    )


# Module roots a deployment artifact may reference while unpickling: the
# sklearn pipeline and its numpy/scipy building blocks, the package's own
# pipeline steps, and the registration helpers pickle itself uses. A global
# reference outside this list is refused before the class is instantiated,
# which blocks the "pickle runs arbitrary code" route that a plain
# ``pickle.load`` leaves open.
_ALLOWED_MODULE_ROOTS = frozenset({"numpy", "scipy", "sklearn", "dit", "copyreg", "collections"})


class _ArtifactUnpickler(pickle.Unpickler):
    """Unpickler restricted to the modules a fitted pipeline may reference."""

    def find_class(self, module: str, name: str) -> Any:
        if module.split(".", 1)[0] not in _ALLOWED_MODULE_ROOTS:
            raise pickle.UnpicklingError(
                f"deployment artifact references module {module!r}, which is outside "
                "the allowlist; the file may be corrupted or malicious"
            )
        return super().find_class(module, name)


def _digest_path(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _section(payload_dict: dict, key: str, source: Path):
    """Fetch one required artifact section as a ValueError, never a KeyError."""

    try:
        value = payload_dict[key]
    except KeyError as exc:
        raise ValueError(
            f"{source} has no {key!r} section; it is not a complete "
            f"version-{ARTIFACT_VERSION} deployment artifact"
        ) from exc
    if value is None:
        # A null section is the same corruption with the key present: refuse
        # at load (with the artifact path) instead of deferring an
        # AttributeError/TypeError to predict time -- or worse, silently
        # accepting it.
        raise ValueError(
            f"{source} has a null {key!r} section; it is not a complete "
            f"version-{ARTIFACT_VERSION} deployment artifact"
        )
    return value


def save_deployment_artifact(model: DeploymentModel, path: str | Path) -> Path:
    """Pickle the deployment bundle to ``path`` (parent directories created).

    Also writes a SHA-256 sidecar next to it (``<path>.sha256``); loading
    refuses any artifact whose sidecar is missing or does not match, so a
    truncated or swapped file is rejected before it can execute.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = pickle.dumps(model.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
    with target.open("wb") as handle:
        handle.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    _digest_path(target).write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return target


def load_deployment_artifact(path: str | Path) -> DeploymentModel:
    """Load an artifact written by :func:`save_deployment_artifact`.

    The checksum sidecar is verified before unpickling and class resolution
    is confined to :data:`_ALLOWED_MODULE_ROOTS`; loading still trusts whoever
    produced the file (see the module docstring).
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"deployment artifact not found: {source}")
    sidecar = _digest_path(source)
    if not sidecar.is_file():
        raise FileNotFoundError(
            f"deployment artifact checksum sidecar not found: {sidecar}; "
            "artifacts are always saved together with it, so refuse to load "
            "an unverifiable pickle and re-fit instead"
        )
    payload = source.read_bytes()
    recorded_tokens = sidecar.read_text(encoding="utf-8").split()
    if not recorded_tokens:
        raise ValueError(
            f"deployment artifact checksum sidecar is empty: {sidecar}; "
            "refuse to load an unverifiable pickle and re-fit instead"
        )
    recorded = recorded_tokens[0].strip().lower()
    # ``hmac.compare_digest`` raises TypeError on non-ASCII input, so a
    # sidecar written with a BOM or by an editor that mangled the digest must
    # be rejected here as malformed instead of crashing with an obscure error.
    if len(recorded) != 64 or any(char not in "0123456789abcdef" for char in recorded):
        raise ValueError(
            f"deployment artifact checksum sidecar is malformed: {sidecar.name} must "
            "start with a 64-character hexadecimal SHA-256 digest; refuse to load an "
            "unverifiable pickle and re-fit instead"
        )
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual.encode("ascii"), recorded.encode("ascii")):
        raise ValueError(
            f"deployment artifact checksum mismatch: {source} does not match {sidecar.name}; "
            "the file was truncated, modified, or paired with the wrong sidecar"
        )
    payload_dict = _ArtifactUnpickler(io.BytesIO(payload)).load()
    if not isinstance(payload_dict, dict) or payload_dict.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"{source} is not a version-{ARTIFACT_VERSION} deployment artifact")
    # Every required section surfaces as a ValueError, because the CLI
    # contract (cli/main.py) only catches ValueError: a truncated or
    # foreign artifact must print one actionable line, not a KeyError
    # traceback.
    config_values = _section(payload_dict, "config", source)
    try:
        config = ExperimentConfig(**dict(config_values))
    except (TypeError, ValueError) as exc:
        # dict() on a non-mapping raises ValueError with a bare dict-builder
        # message; ExperimentConfig raises ValueError for a bad value and
        # TypeError for an unknown key. All three get the artifact path.
        raise ValueError(
            f"{source} has a malformed or incompatible 'config' section: {exc}"
        ) from exc
    if not isinstance(config_values, dict):
        raise ValueError(
            f"{source} has a malformed 'config' section; it is not a complete "
            f"version-{ARTIFACT_VERSION} deployment artifact"
        )
    raw_label_map = _section(payload_dict, "label_map", source)
    # to_dict() stringifies the keys for JSON friendliness, so a legitimate
    # artifact carries digit strings; int() below accepts both. What must be
    # refused is anything int() cannot read (None keys would escape as a bare
    # TypeError) and booleans (True would read as class 1).
    def _is_label_key(key: object) -> bool:
        if isinstance(key, bool):
            return False
        if isinstance(key, int):
            return True
        if isinstance(key, str) and key.strip().lstrip("-").isdigit():
            return True
        return False

    if not isinstance(raw_label_map, dict) or any(
        not _is_label_key(key) for key in raw_label_map
    ):
        raise ValueError(
            f"{source} has a malformed 'label_map' section; its keys must be "
            f"integers (or digit strings), and it is not a complete "
            f"version-{ARTIFACT_VERSION} deployment artifact"
        )
    raw_classes = _section(payload_dict, "classes", source)
    if not isinstance(raw_classes, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_classes
    ):
        # np.asarray(dtype=int) would silently truncate floats ("[1.9, 2.9]"
        # became classes [1, 2], a class no subject belongs to) or merge a
        # string's digits; classes are the probability column order, so the
        # section must be a plain list of integers.
        raise ValueError(
            f"{source} has a malformed 'classes' section; it must be a list of "
            f"integers, got {type(raw_classes).__name__}"
        )
    classes_ = np.asarray(raw_classes, dtype=int)
    if set(classes_.tolist()) != set(int(key) for key in raw_label_map):
        raise ValueError(
            f"{source}: classes {sorted(classes_.tolist())} and label_map keys "
            f"{sorted(int(key) for key in raw_label_map)} disagree; the artifact "
            "is inconsistent and must be re-fitted"
        )
    raw_feature_names = _section(payload_dict, "feature_names", source)
    if not isinstance(raw_feature_names, (list, tuple)):
        raise ValueError(
            f"{source} has a malformed 'feature_names' section; it is not a "
            f"complete version-{ARTIFACT_VERSION} deployment artifact"
        )
    raw_best_params = _section(payload_dict, "best_params", source)
    if not isinstance(raw_best_params, dict):
        raise ValueError(
            f"{source} has a malformed 'best_params' section; it is not a "
            f"complete version-{ARTIFACT_VERSION} deployment artifact"
        )
    return DeploymentModel(
        pipeline=_section(payload_dict, "pipeline", source),
        config=config,
        label_map={int(key): str(value) for key, value in raw_label_map.items()},
        feature_names=tuple(raw_feature_names),
        data_digest=str(_section(payload_dict, "data_digest", source)),
        best_params=dict(raw_best_params),
        classes_=classes_,
    )


def predict_unlabeled(
    dataset: DatasetBundle,
    model: DeploymentModel,
) -> dict[str, Any]:
    """Score rows with a saved artifact and validate the column contract.

    Works on labeled or unlabeled data: labels, when present, are never read
    by the pipeline. Returns per-class probabilities, the argmax prediction in
    canonical encoding, and the display name of each predicted class.
    """

    view_for_features = dataset
    features, _, meta = prepare_matrix(view_for_features, model.config)
    expected = len(model.feature_names)
    if model.config.covariate_strategy == "residualize":
        # The pipeline strips the trailing covariate columns itself; the raw
        # matrix is two columns wider than the residualized model input.
        expected = len(model.feature_names) + 2
    if features.shape[1] != expected:
        raise ValueError(
            f"artifact expects {expected} input columns but the dataset's "
            f"{model.config.feature_view!r} view produced {features.shape[1]}; "
            "check that --view / --missing-pattern match the fitting run"
        )
    if meta.get("control_names") is not None and model.config.control_view is None:
        raise ValueError("artifact was not fitted on a control view but the dataset map is one")

    probabilities = model.pipeline.predict_proba(features)
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("deployment model produced non-finite probabilities")
    totals = probabilities.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6):
        raise RuntimeError(
            "deployment model produced rows whose probabilities do not sum to 1 "
            f"(min {float(totals.min()):.6f}, max {float(totals.max()):.6f})"
        )
    # predict_proba columns follow the fitted classes_, which is not
    # necessarily 0..n-1: a multiclass fit whose training rows miss the middle
    # class has classes_=[0, 2], and argmax alone would report column 1 as
    # label 1 -- a class no subject belongs to.
    predictions = np.asarray(model.classes_, dtype=int)[np.argmax(probabilities, axis=1)]
    # load_deployment_artifact guarantees every class has a label_map entry,
    # so index instead of .get: a fallback would smuggle an unlabelled class
    # into the submission file as a bare number.
    named = [model.label_map[int(label)] for label in predictions]
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "predicted_names": named,
        "class_names": [model.label_map[int(c)] for c in sorted(model.label_map)],
        "data_digest": model.data_digest,
    }
