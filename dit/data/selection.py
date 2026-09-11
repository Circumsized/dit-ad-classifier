"""Nested feature selection for p >> n AFQ matrices.

The dataset has 14,400 features per subject against 700 subjects, so a
classifier fed every node of every metric fits noise. Selection is two staged
and always fits inside a fold:

1. **block ranking** — a regularized L1 model is fit on the full matrix and its
   coefficient magnitudes are aggregated per (tract, metric) block. Blocks are
   ranked by mean magnitude, which suppresses the fact that MD has three
   tensor-identity duplicates among the eight metrics.
2. **within-block sparsification** — surviving blocks keep only columns whose
   coefficient is non-negligible, so the final model names individual nodes.
   A sparse solution enumerates anatomical regions and makes the post-hoc
   "consistency of estimated predictors" analysis possible; a dense one does
   not.

Covariate and missing-pattern columns are left untouched; pruning them would
either leak demographics into the disease signal or discard the missingness
indicator.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dit.data.layout import FeatureLayout
from dit.data.sklearn_compat import l1_ratio_kwargs as _l1_ratio_kwargs


class SparseBlockSelector:
    """Select tract/metric blocks and prune weak nodes, fit per fold."""

    def __init__(
        self,
        layout: FeatureLayout,
        *,
        top_blocks: int | None = None,
        min_blocks: int = 1,
        C: float = 0.05,
        l1_ratio: float = 1.0,
        keep_fraction: float = 0.05,
        max_features: int | None = 300,
        seed: int = 42,
    ) -> None:
        """Configure selection.

        Parameters
        ----------
        layout:
            Column map describing which anatomical unit each column belongs to.
        top_blocks:
            Number of (tract, metric) blocks to retain. ``None`` keeps every
            block and only applies within-block pruning.
        min_blocks:
            Lower bound on retained blocks, so a degenerate search cannot drop
            the matrix to a single column.
        C, l1_ratio:
            Elastic-net regularisation of the ranking model. ``l1_ratio=1.0``
            is LASSO.
        keep_fraction:
            A node survives when its coefficient magnitude is at least this
            fraction of the largest magnitude inside its block.
        max_features:
            Hard cap on retained anatomical columns. ``keep_fraction`` alone
            leaves the dimensionality data-dependent, so this bounds the p/n
            budget: when more anatomical columns survive, only the highest
            magnitude ``max_features`` are kept. Trailing covariate and
            missing-pattern columns are always retained and not counted here.
            ``None`` disables the cap.
        """

        self.layout = layout
        self.top_blocks = top_blocks
        self.min_blocks = min_blocks
        self.C = C
        self.l1_ratio = l1_ratio
        self.keep_fraction = keep_fraction
        self.max_features = max_features
        self.seed = seed

        self._blocks: list[Any] = []
        self.kept_columns_: np.ndarray | None = None
        self.block_scores_: dict[str, float] | None = None

    # -- sklearn estimator protocol ---------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "top_blocks": self.top_blocks,
            "min_blocks": self.min_blocks,
            "C": self.C,
            "l1_ratio": self.l1_ratio,
            "keep_fraction": self.keep_fraction,
            "max_features": self.max_features,
            "seed": self.seed,
        }

    def set_params(self, **params: Any) -> "SparseBlockSelector":
        for name, value in params.items():
            if not hasattr(self, name):
                raise ValueError(f"unknown parameter {name!r}")
            setattr(self, name, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SparseBlockSelector":
        """Rank blocks and choose surviving nodes from training data."""

        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - exercised via test_env
            raise RuntimeError("scikit-learn is required for feature selection") from exc

        values = np.asarray(X, dtype=np.float64)
        if y is None:
            raise ValueError("selection requires labels")
        labels = np.asarray(y).reshape(-1)
        if values.shape[0] != labels.shape[0]:
            raise ValueError("X and y must have the same row count")
        if values.shape[0] < 4:
            raise ValueError("selection needs at least 4 samples")

        n_classes = len(set(labels.tolist()))
        if n_classes < 2:
            raise ValueError("selection needs at least 2 classes")

        layout = self.layout
        layout.validate(values.shape[1])
        core = values[:, : layout.n_feature_columns]

        prepare = Pipeline(
            [
                (
                    "imputer",
                    # keep_empty_features: an all-NaN column inside this fold
                    # would otherwise be dropped, shifting every later
                    # coefficient away from its anatomical block and tripping
                    # the width check below. An empty column imputes to 0 and
                    # simply cannot earn a nonzero ranking coefficient.
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        prepare.fit(core)
        prepared = prepare.transform(core)

        # liblinear handles L1 for two classes; saga covers the multinomial
        # three-class case, where a pure L1 penalty is unavailable.
        if n_classes == 2:
            model = LogisticRegression(
                C=self.C,
                **_l1_ratio_kwargs(1.0, "liblinear"),
                random_state=self.seed,
                max_iter=1000,
            )
        else:
            model = LogisticRegression(
                C=self.C,
                **_l1_ratio_kwargs(self.l1_ratio, "saga"),
                random_state=self.seed,
                max_iter=3000,
            )
        model.fit(prepared, labels)

        magnitudes = np.abs(model.coef_)
        if magnitudes.ndim == 2:
            magnitudes = magnitudes.max(axis=0)
        if magnitudes.size != layout.n_feature_columns:
            raise ValueError(
                f"ranking model produced {magnitudes.size} coefficients for "
                f"{layout.n_feature_columns} core columns"
            )

        self._blocks = layout.anatomical_blocks()
        self.block_scores_ = {
            block.name: float(np.mean(magnitudes[list(block.columns)]))
            for block in self._blocks
        }

        if self.top_blocks is None:
            retained = self._blocks
        else:
            by_name = {block.name: block for block in self._blocks}
            ranked = sorted(self.block_scores_, key=lambda n: self.block_scores_[n], reverse=True)
            count = max(self.min_blocks, min(int(self.top_blocks), len(ranked)))
            retained = [by_name[name] for name in ranked[:count]]

        prune_mask = np.zeros(layout.n_feature_columns, dtype=bool)
        for block in retained:
            columns = list(block.columns)
            segment = magnitudes[columns]
            if segment.size == 0:
                continue
            threshold = self.keep_fraction * float(np.max(segment))
            prune_mask[columns] = segment >= threshold

        kept = np.flatnonzero(prune_mask)
        if kept.size == 0:
            kept = np.asarray([int(np.argmax(magnitudes))], dtype=int)

        # Apply the feature budget: if more anatomical columns survived the
        # block/threshold stage than allowed, keep only the strongest, so
        # dimensionality is bounded rather than data-dependent. Ties break on
        # column index for determinism.
        if self.max_features is not None and kept.size > self.max_features:
            order = sorted(kept.tolist(), key=lambda column: (-magnitudes[column], column))
            kept = np.asarray(sorted(order[: self.max_features]), dtype=int)

        tail = np.arange(layout.n_feature_columns, values.shape[1], dtype=int)
        if tail.size:
            kept = np.concatenate([kept, tail])
        self.kept_columns_ = np.sort(kept)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Keep the columns chosen during :meth:`fit`."""

        if self.kept_columns_ is None:
            raise RuntimeError("selector must be fit before transform")
        values = np.asarray(X, dtype=np.float64)
        if values.shape[1] != len(self.layout.feature_names()):
            raise ValueError(f"unexpected width {values.shape[1]}")
        return values[:, self.kept_columns_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # -- reporting --------------------------------------------------------

    @property
    def n_selected(self) -> int:
        if self.kept_columns_ is None:
            raise RuntimeError("selector is not fitted")
        return int(self.kept_columns_.size)

    @property
    def retained_blocks(self) -> list[str]:
        """Names of the anatomical blocks that survived selection."""

        if self.block_scores_ is None or self.kept_columns_ is None:
            raise RuntimeError("selector is not fitted")
        kept = set(self.kept_columns_.tolist())
        return sorted(
            block.name
            for block in self._blocks
            if any(column in kept for column in block.columns)
        )

    def retained_nodes(self) -> dict[str, list[int]]:
        """Node offsets retained within each selected block."""

        if self.kept_columns_ is None:
            return {}
        kept = set(self.kept_columns_.tolist())
        result: dict[str, list[int]] = {}
        for block in self._blocks:
            retained = [
                offset
                for offset, column in enumerate(block.columns)
                if column in kept
            ]
            if retained:
                result[block.name] = retained
        return result

    def report(self) -> dict[str, object]:
        return {
            "top_blocks": self.top_blocks,
            "C": self.C,
            "l1_ratio": self.l1_ratio,
            "keep_fraction": self.keep_fraction,
            "n_selected": self.n_selected,
            "n_total": self.layout.total_features,
            "retained_blocks": self.retained_blocks,
            "retained_nodes": self.retained_nodes(),
            "block_scores": self.block_scores_,
        }

    @staticmethod
    def make_pipeline(
        layout: FeatureLayout,
        estimator_factory,
        *,
        top_blocks: int | None = None,
        keep_fraction: float = 0.05,
        seed: int = 42,
    ):
        """Return an sklearn pipeline with fold-local selection in front of a model."""

        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        selector = SparseBlockSelector(
            layout, top_blocks=top_blocks, keep_fraction=keep_fraction, seed=seed
        )
        return Pipeline(
            [
                ("selector", selector),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator_factory()),
            ]
        )
