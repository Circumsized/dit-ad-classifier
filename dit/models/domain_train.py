"""Training loop for the tract Transformer with optional domain alignment.

Combines the network in :mod:`dit.models.tract_transformer` with the alignment
primitives in :mod:`dit.models.domain_adaptation` under the same leakage rules
as the classical path: every normalisation statistic and every early-stopping
decision is made on outer-fold training rows only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from dit.models import ALIGNMENTS  # torch-free constant, single source of truth
from dit.models.calibration import (
    CALIBRATIONS,
    apply_platt_scalars,
    fit_platt_scalars,
    temperature_saturated,
    temperature_scale,
)
from dit.models.domain_adaptation import (
    DomainDiscriminator,
    coral_loss,
    mmd_rbf_loss,
)
from dit.models.tract_transformer import TractTransformer

__all__ = [
    "ALIGNMENTS",
    "CALIBRATIONS",
    "DomainAlignedClassifier",
    "DomainTrainConfig",
    "search_domain_classifier",
]

# Two settings only: this loop is expensive, and a wider grid multiplies the
# outer-fold cost by the grid size. The search varies the learning rate only;
# ``epochs`` is a hard budget owned by the caller's config, so the epochs a
# report claims are the epochs that actually ran instead of a hidden grid value.
_SEARCH_GRID: tuple[dict[str, Any], ...] = (
    {"learning_rate": 2e-3},
    {"learning_rate": 1e-3},
)


@dataclass
class DomainTrainConfig:
    """Hyper-parameters for one :class:`DomainAlignedClassifier` fit."""

    d_model: int = 48
    n_heads: int = 4
    n_layers: int = 2
    feedforward_dim: int = 144
    dropout: float = 0.15
    pooling: str = "cls"
    epochs: int = 90
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 12
    validation_fraction: float = 0.15
    alignment: str = "none"
    alignment_weight: float = 0.1
    alignment_ramp: int = 20
    alignment_batch: int = 64
    class_weights: bool = True
    calibration: str = "none"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.alignment not in ALIGNMENTS:
            raise ValueError(f"unknown alignment {self.alignment!r}; use one of {ALIGNMENTS}")
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"unknown calibration {self.calibration!r}; use one of {CALIBRATIONS}")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        # 0.0 disables the hold-out, which the tuning pass inside
        # :func:`search_domain_classifier` needs: it supplies its own outer
        # validation slice, so a second nested split would starve the fit.
        if not 0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be inside [0, 0.5)")
        if self.batch_size < 2:
            raise ValueError("batch_size must be >= 2 so domain groups can be compared")
        if self.alignment_weight < 0:
            raise ValueError("alignment_weight must be >= 0")
        if self.alignment_batch < 2:
            raise ValueError("alignment_batch must be >= 2")
        if self.alignment_ramp < 1:
            raise ValueError("alignment_ramp must be >= 1")


def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy shared with the sklearn path's scoring rule."""

    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def _class_weights(y: np.ndarray, n_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights, skipping classes absent from the fold."""

    counts = np.bincount(y, minlength=n_classes).astype(float)
    weights = np.where(counts > 0, counts.sum() / (counts.size * counts), 0.0)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _alignment_loss(
    alignment: str,
    features: torch.Tensor,
    groups: list[int],
    discriminator: nn.Module | None,
    strength: float,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, bool]:
    """Mean alignment penalty across the sites present in the batch.

    Returns ``(loss, used)``. ``used`` is False when fewer than two sites have
    two or more members each; the loss stays zero because the alignment
    statistic would be degenerate. Rows with a negative group id are excluded
    from the alignment term.
    """

    group_array = np.asarray(groups)
    known_positions = [index for index, group in enumerate(groups) if int(group) >= 0]
    if len(known_positions) < 2:
        return features.new_zeros(()), False
    known_index = torch.as_tensor(known_positions, device=features.device, dtype=torch.long)

    present = sorted({int(group_array[position]) for position in known_positions})
    members = {
        label: torch.as_tensor(
            [position for position in known_positions if int(group_array[position]) == label],
            device=features.device,
            dtype=torch.long,
        )
        for label in present
    }
    usable = [label for label in present if members[label].shape[0] >= 2]
    if len(usable) < 2:
        return features.new_zeros(()), False

    if alignment == "dann":
        if discriminator is None:
            raise RuntimeError("dann alignment requires a discriminator")
        # One multi-class pass over every known-site row, using its actual site
        # id as the target, so sites beyond the first two are positive classes
        # rather than an unused discriminator head.
        selected = features.index_select(0, known_index)
        target = torch.as_tensor(
            [int(group_array[position]) for position in known_positions],
            device=features.device,
            dtype=torch.long,
        )
        logits = discriminator(selected, strength=strength)
        return nn.functional.cross_entropy(logits, target), True

    chunks = [features.index_select(0, members[label]) for label in usable]
    pairs: list[tuple[int, int]] = []
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            pairs.append((i, j))
    if len(pairs) > n_pairs:
        picked = rng.choice(len(pairs), size=n_pairs, replace=False)
        pairs = [pairs[int(index)] for index in picked]

    loss = features.new_zeros(())
    for i, j in pairs:
        if alignment == "coral":
            loss = loss + coral_loss(chunks[i], chunks[j])
        elif alignment == "mmd":
            loss = loss + mmd_rbf_loss(chunks[i], chunks[j])
    return loss / len(pairs), True


class DomainAlignedClassifier:
    """Fits a :class:`TractTransformer` and reports class probabilities.

    ``fit`` consumes raw ``[N, T, P, M]`` profiles, so this class sits beside
    rather than inside the sklearn search: the classical path flattens profiles
    into a table first, and that flattening is not reversible.
    """

    def __init__(self, config: DomainTrainConfig | None = None, device: str = "cpu") -> None:
        self.config = config or DomainTrainConfig()
        if self.config.alignment not in ALIGNMENTS:
            raise ValueError(f"unknown alignment {self.config.alignment!r}")
        if self.config.calibration not in CALIBRATIONS:
            raise ValueError(f"unknown calibration {self.config.calibration!r}")
        self.device = torch.device(device)
        self.model: TractTransformer | None = None
        self.discriminator: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.covariate_median_: np.ndarray | None = None
        self.history_: dict[str, list[float]] = {}
        self.alignment_active_: bool = False
        self.temperature_: float = 1.0
        self.temperature_saturated_: bool = False
        self.platt_: list[Any] | None = None
        self.calibration_applied_: bool = False
        self.class_weights_: list[float] | None = None

    def fit(
        self,
        profiles: np.ndarray,
        y: np.ndarray,
        *,
        covariates: np.ndarray | None = None,
        site: np.ndarray | None = None,
    ) -> "DomainAlignedClassifier":
        # Re-seed inside fit rather than at construction time, so every outer
        # fold re-enters the constructor with a fresh random state and the
        # report is reproducible.
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        profiles = np.asarray(profiles, dtype=np.float32)
        y = np.asarray(y).reshape(-1).astype(int)
        if profiles.ndim != 4 or profiles.shape[0] != y.shape[0]:
            raise ValueError("profiles must be [N,T,P,M] with one row per label")
        if covariates is None:
            covariates = np.full((y.shape[0], 2), np.nan, dtype=np.float32)
        covariates = np.asarray(covariates, dtype=np.float32).reshape(y.shape[0], 2)
        if covariates.shape[0] != y.shape[0]:
            raise ValueError("covariates must have one row per subject")

        classes = np.unique(y)
        if classes.size < 2:
            raise ValueError("at least two classes are required to fit")
        relabel = {int(label): position for position, label in enumerate(classes)}
        labels = np.asarray([relabel[int(value)] for value in y], dtype=int)
        n_classes = classes.size

        grouped = self.config.alignment != "none" and site is not None
        if grouped:
            site_values = np.asarray(site).reshape(-1)
            if site_values.shape[0] != y.shape[0]:
                raise ValueError("site must have one value per subject")
            # -1 marks "site unknown" in the dataset contract; keep it out of
            # the domain groups by moving it to -2, which the relabeller skips.
            site_values = np.where(site_values == -1, -2, site_values)
            groups = _relabel(site_values)
            if np.unique(groups[groups >= 0]).size < 2:
                grouped = False

        include_covariates = bool(np.any(np.isfinite(covariates)))
        self.model = TractTransformer(
            n_tracts=profiles.shape[1],
            n_points=profiles.shape[2],
            n_metrics=profiles.shape[3],
            n_classes=n_classes,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            feedforward_dim=self.config.feedforward_dim,
            dropout=self.config.dropout,
            pooling=self.config.pooling,
            include_covariates=include_covariates,
        ).to(self.device)
        n_domains = int(groups.max() + 1) if grouped else 0
        # The features handed to the alignment loss are the classifier's
        # read-out, which grows when covariates are concatenated, so the
        # discriminator width must match that rather than the bare d_model.
        feature_dim = self.config.d_model + (16 if include_covariates else 0)
        self.discriminator = (
            DomainDiscriminator(feature_dim, n_domains).to(self.device) if grouped else None
        )

        train_local, validation_local, calibration_local = self._split_indices(labels)
        no_holdout = validation_local.size == 0

        # Fit centring on the training slice. Validation rows stay outside the
        # classifier loss, and calibration rows stay outside the entire fit.
        training_profiles = profiles[train_local]
        column_mean = np.nanmean(training_profiles, axis=0)
        column_scale = np.nanstd(training_profiles, axis=0)
        self.center_ = np.where(np.isfinite(column_mean), column_mean, 0.0)
        self.scale_ = np.where(column_scale > 1e-8, column_scale, 1.0)
        finite = np.isfinite(profiles)
        normalised = np.where(finite, (profiles - self.center_) / self.scale_, 0.0).astype(
            np.float32
        )
        # Impute covariate NaNs with training-row medians so no NaN reaches the
        # network; a real cohort has missing age/sex for some subjects. The
        # medians come from the training slice only and are stored so
        # ``predict_proba`` fills held-out rows identically; a wholly missing
        # column falls back to zero.
        train_cov = covariates[train_local]
        column_median = np.zeros(train_cov.shape[1], dtype=np.float64)
        for column in range(train_cov.shape[1]):
            finite_values = train_cov[np.isfinite(train_cov[:, column]), column]
            if finite_values.size:
                column_median[column] = float(np.median(finite_values))
        self.covariate_median_ = column_median.astype(np.float32)
        covariates = np.where(
            np.isfinite(covariates), covariates, self.covariate_median_
        ).astype(np.float32)

        optimiser = torch.optim.AdamW(
            list(self.model.parameters())
            + (list(self.discriminator.parameters()) if self.discriminator is not None else []),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # Inverse-frequency weights from the *training rows only*: the
        # validation and calibration slices are held out, so their labels must
        # not shape the loss — otherwise changing a held-out label would change
        # how the model trains on data it never sees.
        weights = (
            _class_weights(labels[train_local], n_classes, self.device)
            if self.config.class_weights
            else None
        )
        self.class_weights_ = (
            None if weights is None else weights.detach().cpu().numpy().tolist()
        )

        history: dict[str, list[float]] = {"validation": []}
        best_score = -np.inf
        best_state: dict[str, torch.Tensor] | None = None
        stall = 0
        generator = np.random.default_rng(self.config.seed)
        parameters = list(self.model.parameters()) + (
            list(self.discriminator.parameters()) if self.discriminator is not None else []
        )

        for epoch in range(self.config.epochs):
            self.model.train()
            order = generator.permutation(train_local)
            optimiser.zero_grad()
            epoch_loss = 0.0
            batches = 0
            for start in range(0, order.shape[0], self.config.batch_size):
                index = order[start : start + self.config.batch_size]
                logits = self.model(
                    torch.as_tensor(normalised[index], device=self.device),
                    valid_mask=torch.as_tensor(finite[index], device=self.device),
                    covariates=torch.as_tensor(covariates[index], device=self.device),
                )
                target = torch.as_tensor(labels[index], device=self.device, dtype=torch.long)
                loss = nn.functional.cross_entropy(logits, target, weight=weights)
                # Gradients accumulate across the batches below so the alignment
                # term, which needs several sites at once, can be added before a
                # single step.
                loss.backward()
                epoch_loss += float(loss)
                batches += 1

            if grouped:
                # A mini-batch of a handful of subjects rarely holds two sites
                # with two members each, so aligning per batch is a no-op.  The
                # statistic is computed over an epoch-level subsample of the
                # training rows instead; the ramp keeps the first epochs pure
                # classification so the features are meaningful when aligned.
                strength = min(1.0, (epoch + 1) / self.config.alignment_ramp)
                window = order[: min(self.config.alignment_batch, order.shape[0])]
                if window.shape[0] >= 2:
                    _, features = self.model(
                        torch.as_tensor(normalised[window], device=self.device),
                        valid_mask=torch.as_tensor(finite[window], device=self.device),
                        covariates=torch.as_tensor(covariates[window], device=self.device),
                        return_features=True,
                    )
                    penalty, used = _alignment_loss(
                        self.config.alignment,
                        features,
                        groups[window].tolist(),
                        self.discriminator,
                        strength,
                        n_pairs=min(6, max(1, window.shape[0] // 4)),
                        rng=generator,
                    )
                    if used:
                        self.alignment_active_ = True
                        (self.config.alignment_weight * penalty).backward()
                        epoch_loss += float(self.config.alignment_weight * penalty)

            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            history.setdefault("training", []).append(epoch_loss / max(batches, 1))
            if no_holdout:
                continue

            self.model.eval()
            with torch.no_grad():
                validation_logits = self.model(
                    torch.as_tensor(normalised[validation_local], device=self.device),
                    valid_mask=torch.as_tensor(finite[validation_local], device=self.device),
                    covariates=torch.as_tensor(covariates[validation_local], device=self.device),
                )
            score = _balanced_accuracy(
                labels[validation_local],
                validation_logits.argmax(dim=1).cpu().numpy(),
            )
            history["validation"].append(score)
            if score > best_score + 1e-6:
                best_score = score
                best_state = {key: value.detach().clone() for key, value in self.model.state_dict().items()}
                stall = 0
            else:
                stall += 1
                if stall >= self.config.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.classes_ = classes
        self.history_ = history
        self.best_validation_score_ = None if no_holdout else float(best_score)
        self._fit_calibration(labels, normalised, covariates, finite, calibration_local)
        return self

    def _forward_logits(
        self,
        normalised: np.ndarray,
        covariates: np.ndarray,
        finite: np.ndarray,
    ) -> np.ndarray:
        """Raw class logits for every row, in evaluation mode.

        ``finite`` is the missing-value mask of the *raw* profiles these rows
        came from: normalisation replaces NaN with 0, so the mask cannot be
        recovered downstream and must travel with the data.
        """

        if self.model is None:
            raise RuntimeError("classifier must be fit before scoring")
        logits = np.zeros((normalised.shape[0], self.model.n_classes), dtype=float)
        with torch.no_grad():
            for start in range(0, normalised.shape[0], self.config.batch_size):
                index = slice(start, min(start + self.config.batch_size, normalised.shape[0]))
                logits[index] = self.model(
                    torch.as_tensor(normalised[index], device=self.device),
                    valid_mask=torch.as_tensor(finite[index], device=self.device),
                    covariates=torch.as_tensor(covariates[index], device=self.device),
                ).cpu().numpy()
        return logits

    def _fit_calibration(
        self,
        labels: np.ndarray,
        normalised: np.ndarray,
        covariates: np.ndarray,
        finite: np.ndarray,
        calibration_rows: np.ndarray,
    ) -> None:
        """Fit the probability map on rows outside training and early stopping.

        These rows come from the holdout that drives early stopping and do not
        enter the classifier loss. Calibration requires a non-empty slice; an
        empty slice raises instead of reporting an unfitted map as calibrated.
        """

        self.temperature_ = 1.0
        self.temperature_saturated_ = False
        self.platt_ = None
        self.calibration_applied_ = False
        if self.config.calibration == "none":
            return
        if calibration_rows.size < 2:
            raise ValueError(
                f"{self.config.calibration} calibration needs at least two held-out "
                "rows; increase validation_fraction"
            )

        logits = self._forward_logits(
            normalised[calibration_rows],
            covariates[calibration_rows],
            finite[calibration_rows],
        )
        targets = labels[calibration_rows]
        if self.config.calibration == "temperature":
            self.temperature_ = temperature_scale(logits, targets)
            # Published next to the value: a clipped answer and a fitted one can
            # print identically, and that difference decides whether the reported
            # confidence is trustworthy.
            self.temperature_saturated_ = temperature_saturated(self.temperature_)
        else:
            probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
            self.platt_ = fit_platt_scalars(probabilities, targets)
        self.calibration_applied_ = True

    def predict_proba(self, profiles: np.ndarray, covariates: np.ndarray | None = None) -> np.ndarray:
        """Class probabilities, with rows ordered by :attr:`classes_`."""

        if self.model is None or self.classes_ is None or self.center_ is None or self.scale_ is None:
            raise RuntimeError("classifier must be fit before predict_proba")
        profiles = np.asarray(profiles, dtype=np.float32)
        if profiles.ndim != 4:
            raise ValueError(f"profiles must be [N,T,P,M], got {profiles.shape}")
        finite = np.isfinite(profiles)
        normalised = np.where(
            finite,
            (profiles - self.center_) / self.scale_,
            0.0,
        ).astype(np.float32)
        if covariates is None:
            covariates = np.full((profiles.shape[0], 2), np.nan, dtype=np.float32)
        covariates = np.asarray(covariates, dtype=np.float32).reshape(profiles.shape[0], 2)
        if self.covariate_median_ is not None:
            covariates = np.where(
                np.isfinite(covariates), covariates, self.covariate_median_
            ).astype(np.float32)
        logits = self._forward_logits(normalised, covariates, finite)
        if self.calibration_applied_ and self.config.calibration == "temperature":
            logits = logits / self.temperature_
        probabilities = torch.softmax(torch.as_tensor(logits), dim=1).numpy()
        if self.platt_ is not None:
            probabilities = apply_platt_scalars(probabilities, self.platt_)
        return np.asarray(probabilities, dtype=float)

    def parameters(self) -> dict[str, Any]:
        """Reportable fit summary; keys are stable so reports stay diffable."""

        return {
            "alignment": self.config.alignment,
            "alignment_weight": self.config.alignment_weight,
            "alignment_active": bool(self.alignment_active_),
            # Counted from the training series rather than the validation
            # series: with no hold-out there is no validation entry at all.
            "epochs_requested": int(self.config.epochs),
            "epochs_run": len(self.history_.get("training", [])),
            "best_validation_score": None
            if getattr(self, "best_validation_score_", None) is None
            else float(self.best_validation_score_),
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "pooling": self.config.pooling,
            "class_weights": self.config.class_weights,
            "class_weights_values": self.class_weights_,
            "calibration": self.config.calibration,
            "calibration_applied": bool(self.calibration_applied_),
            "temperature": float(self.temperature_),
            "temperature_saturated": bool(self.temperature_saturated_),
            "seed": self.config.seed,
        }

    def _split_indices(
        self, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hold out validation and calibration slices for early stopping.

        Stratified by class rather than a plain random cut, because the score
        it selects on is balanced accuracy: a slice that happens to miss a
        class would report a meaningless number and steer early stopping the
        wrong way.  A fraction of zero returns every row to training and empty
        hold-outs, which is what the tuning pass needs.

        The hold-out is divided *per class* when calibration is requested,
        because a calibrator fitted on the same rows that chose the stopping
        point is fitting on its own answer key.  A class-level cut is what
        keeps both halves usable: splitting the class-ordered hold-out list in
        half by position can leave an entire class on one side, and a
        calibrator or early-stopping score that never sees that class is not
        fitting or selecting anything meaningful.
        """

        if self.config.validation_fraction == 0:
            return np.arange(labels.shape[0]), np.arange(0, dtype=int), np.arange(0, dtype=int)
        train, held_out = _stratified_holdout(
            labels, self.config.validation_fraction, self.config.seed + 1
        )
        if self.config.calibration == "none":
            validation_rows, calibration_rows = held_out, np.arange(0, dtype=int)
        else:
            validation_rows, calibration_rows = _stratified_halves(
                held_out, labels, self.config.seed + 3
            )
        validation = np.asarray(validation_rows, dtype=int)
        calibration = np.asarray(calibration_rows, dtype=int)
        if train.size < 2 or validation.size < 1:
            raise ValueError("not enough rows to hold out a validation slice")
        return train, validation, calibration


def _stratified_holdout(
    labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into ``(train, held_out)`` with at least one held-out row per class.

    Stratified by class rather than a plain random cut, because every score
    that consumes the hold-out is balanced accuracy: a slice that happens to
    miss a class would report a meaningless number.  A singleton class keeps
    its only member for training, since there is nothing to compare.
    """

    generator = np.random.default_rng(seed)
    train_rows: list[int] = []
    held_out: list[int] = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        generator.shuffle(members)
        cut = int(round(members.size * fraction))
        cut = min(max(cut, 1), members.size - 1)
        held_out.extend(members[:cut].tolist())
        train_rows.extend(members[cut:].tolist())
    return np.asarray(train_rows, dtype=int), np.asarray(held_out, dtype=int)


def _stratified_halves(
    held_out: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[list[int], list[int]]:
    """Divide a held-out slice into early-stop and calibration halves per class.

    Every class with two or more held-out rows contributes at least one row to
    each half; a class with a single held-out row cannot serve both roles, so
    the fit fails with the remedy in the message instead of producing a
    calibrator or stopping score that never saw that class.
    """

    generator = np.random.default_rng(seed)
    held_out = np.asarray(held_out, dtype=int)
    validation: list[int] = []
    calibration: list[int] = []
    for label in np.unique(labels[held_out]):
        members = held_out[labels[held_out] == label]
        generator.shuffle(members)
        if members.size < 2:
            raise ValueError(
                f"class {int(label)} holds out only one row, which cannot serve "
                "both early stopping and calibration; increase validation_fraction, "
                "reduce the fold's class imbalance, or set calibration='none'"
            )
        half = members.size // 2
        validation.extend(members[:half].tolist())
        calibration.extend(members[half:].tolist())
    return validation, calibration


def _relabel(values: np.ndarray) -> np.ndarray:
    """Map arbitrary site codes onto contiguous integers, keeping -2 as missing."""

    known = np.unique(values)
    known = known[known >= 0]
    mapping = {int(value): position for position, value in enumerate(known)}
    return np.asarray(
        [mapping[int(value)] if int(value) in mapping else -2 for value in values], dtype=int
    )


def search_domain_classifier(
    profiles: np.ndarray,
    y: np.ndarray,
    *,
    config: DomainTrainConfig | None = None,
    covariates: np.ndarray | None = None,
    site: np.ndarray | None = None,
    device: str = "cpu",
) -> tuple[DomainAlignedClassifier, dict[str, Any]]:
    """Pick between the grid settings on a validation split, then refit.

    Returns the winning classifier together with the grid report so the outer
    experiment can publish which setting was chosen, matching the
    ``best_params`` field of the sklearn path.
    """

    base = config or DomainTrainConfig()
    labels = np.asarray(y).reshape(-1)
    if profiles.shape[0] != labels.shape[0]:
        raise ValueError("profiles and labels must have the same row count")
    site_values = None if site is None else np.asarray(site).reshape(-1)

    # Stratified, exactly like the in-fit hold-out: a candidate judged on a
    # validation slice that misses a class would be scored on a partial
    # balanced accuracy, and the grid would pick a winner on that noise.
    tuning, validation = _stratified_holdout(
        labels, base.validation_fraction, base.seed + 2
    )
    if tuning.size < 2 or validation.size < 1:
        raise ValueError("not enough rows to hold out a validation slice")
    if set(np.unique(labels[validation]).tolist()) != set(np.unique(labels).tolist()):
        raise ValueError("validation slice does not cover every class")
    # Checked before any network is trained: the winner's own hold-out is split
    # between early stopping and calibration, so four rows are the smallest
    # useful amount.
    if base.calibration != "none" and validation.size < 4:
        raise ValueError(
            f"{base.calibration} calibration needs two rows each for the validation and "
            f"calibration slices, but this fold holds out only {validation.size}; use more data per "
            f"fold or set calibration='none'"
        )

    candidates: list[tuple[float, dict[str, Any]]] = []
    for overrides in _SEARCH_GRID:
        # The candidate trains on the tuning slice only, and with the hold-out
        # disabled: this function is the hold-out.  Calibration is dropped with
        # it, because a candidate is judged on balanced accuracy, so its
        # probability scale is never used, and a calibration fit would fail on
        # an empty slice.
        candidate = DomainAlignedClassifier(
            _with_overrides(base, overrides, validation_fraction=0.0, calibration="none"),
            device=device,
        )
        candidate.fit(
            profiles[tuning],
            labels[tuning],
            covariates=None if covariates is None else covariates[tuning],
            site=None if site_values is None else site_values[tuning],
        )
        tuned = candidate.predict_proba(
            profiles[validation], None if covariates is None else covariates[validation]
        )
        candidates.append((_balanced_accuracy(labels[validation], tuned.argmax(axis=1)), dict(overrides)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best_overrides = candidates[0]
    winner = DomainAlignedClassifier(_with_overrides(base, best_overrides), device=device)
    winner.fit(profiles, labels, covariates=covariates, site=site_values)
    return winner, {
        "best_params": best_overrides,
        "tuning_score": float(best_score),
        "epochs_requested": int(base.epochs),
        "grid": [{"params": overrides, "score": float(score)} for score, overrides in candidates],
    }


def _with_overrides(base: DomainTrainConfig, overrides: dict[str, Any], **extra: Any) -> DomainTrainConfig:
    """Return a copy of ``base`` with grid overrides applied."""

    values = {**base.__dict__, **overrides, **extra}
    return DomainTrainConfig(**values)
