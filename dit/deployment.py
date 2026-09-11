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
"""

from __future__ import annotations

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


def save_deployment_artifact(model: DeploymentModel, path: str | Path) -> Path:
    """Pickle the deployment bundle to ``path`` (parent directories created)."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(model.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return target


def load_deployment_artifact(path: str | Path) -> DeploymentModel:
    """Load an artifact written by :func:`save_deployment_artifact`."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"deployment artifact not found: {source}")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(f"{source} is not a version-{ARTIFACT_VERSION} deployment artifact")
    config_values = dict(payload["config"])
    config = ExperimentConfig(**config_values)
    return DeploymentModel(
        pipeline=payload["pipeline"],
        config=config,
        label_map={int(key): str(value) for key, value in payload["label_map"].items()},
        feature_names=tuple(payload["feature_names"]),
        data_digest=str(payload["data_digest"]),
        best_params=dict(payload["best_params"]),
        classes_=np.asarray(payload["classes"], dtype=int),
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
    predictions = np.argmax(probabilities, axis=1)
    named = [model.label_map.get(int(label), str(int(label))) for label in predictions]
    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "predicted_names": named,
        "class_names": [model.label_map[int(c)] for c in sorted(model.label_map)],
        "data_digest": model.data_digest,
    }
