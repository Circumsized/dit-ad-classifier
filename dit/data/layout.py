"""Column addressing for AFQ feature matrices.

``build_feature_matrix`` returns the matrix; ``describe_layout`` returns the
bookkeeping that maps each column back to a (fiber, node, metric) triple, so
importance scores can be reported as "left uncinate fasciculus, node 82, MD"
instead of "feature 12044". Block-level selection and the tract x node heatmaps
both use this mapping, so it lives next to the feature builders rather than
inside either consumer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

VALID_VIEWS = ("profile", "summary")


@dataclass(frozen=True)
class Block:
    """A group of columns that belong to one anatomical unit."""

    name: str
    start: int
    end: int  # exclusive
    tract_index: int
    metric_index: int
    stat: str | None = None

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def columns(self) -> range:
        return range(self.start, self.end)


@dataclass(frozen=True)
class FeatureLayout:
    """Describes the columns produced by a feature view.

    ``kind`` is ``profile`` (every along-tract node retained), ``summary``
    (mean/std/slope/area per tract and metric) or ``metric`` (one diffusion
    metric with all tract nodes).
    """

    kind: str
    tract_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    n_points: int
    metric_index: int | None = None
    stats: tuple[str, ...] = field(default_factory=tuple)
    has_covariates: bool = False
    has_missing_pattern: bool = False

    @property
    def n_tracts(self) -> int:
        return len(self.tract_names)

    @property
    def n_metrics(self) -> int:
        return len(self.metric_names)

    @property
    def covariate_columns(self) -> list[str]:
        return ["age", "sex"] if self.has_covariates else []

    @property
    def missing_pattern_columns(self) -> list[str]:
        return [f"missing|{tract}" for tract in self.tract_names] if self.has_missing_pattern else []

    @property
    def n_feature_columns(self) -> int:
        """Columns that are neither covariates nor missing-pattern indicators."""

        if self.kind == "summary":
            return self.n_tracts * self.n_metrics * len(self.stats)
        if self.kind == "metric":
            return self.n_tracts * self.n_points
        return self.n_tracts * self.n_points * self.n_metrics

    @property
    def total_features(self) -> int:
        return (
            self.n_feature_columns
            + len(self.missing_pattern_columns)
            + len(self.covariate_columns)
        )

    def blocks(self) -> list[Block]:
        """Return anatomical column groups in matrix order."""

        groups: list[Block] = []
        offset = 0
        if self.kind == "summary":
            for stat in self.stats:
                for tract_index, tract in enumerate(self.tract_names):
                    for metric_index, metric in enumerate(self.metric_names):
                        size = 1
                        groups.append(
                            Block(
                                name=f"{tract}|{metric}|{stat}",
                                start=offset,
                                end=offset + size,
                                tract_index=tract_index,
                                metric_index=metric_index,
                                stat=stat,
                            )
                        )
                        offset += size
        elif self.kind == "metric":
            for tract_index, tract in enumerate(self.tract_names):
                metric = self.metric_names[self.metric_index or 0]
                groups.append(
                    Block(
                        name=f"{tract}|{metric}",
                        start=offset,
                        end=offset + self.n_points,
                        tract_index=tract_index,
                        metric_index=self.metric_index or 0,
                    )
                )
                offset += self.n_points
        else:
            for tract_index, tract in enumerate(self.tract_names):
                for metric_index, metric in enumerate(self.metric_names):
                    groups.append(
                        Block(
                            name=f"{tract}|{metric}",
                            start=offset,
                            end=offset + self.n_points,
                            tract_index=tract_index,
                            metric_index=metric_index,
                        )
                    )
                    offset += self.n_points

        if self.has_missing_pattern:
            groups.append(
                Block(
                    name="missing_pattern",
                    start=offset,
                    end=offset + self.n_tracts,
                    tract_index=-1,
                    metric_index=-1,
                )
            )
            offset += self.n_tracts

        expected = self.n_feature_columns + len(self.missing_pattern_columns)
        if offset != expected:
            raise ValueError(
                f"layout block accounting produced {offset} columns, expected {expected}"
            )
        return groups

    def is_anatomical(self, column: int) -> bool:
        """True when a column is a tract measurement rather than covariate."""

        return column < self.n_feature_columns

    def anatomical_blocks(self) -> list[Block]:
        """Blocks covering tract measurements only.

        Selection ranks these. The missing-pattern and covariate blocks are
        appended by the builders after them and are not pruned.
        """

        return [
            block
            for block in self.blocks()
            if block.end <= self.n_feature_columns
        ]

    def index_blocks(self) -> dict[str, Block]:
        """Return blocks keyed by name, for lookups by anatomical unit."""

        return {block.name: block for block in self.blocks()}

    def feature_names(self) -> list[str]:
        """Return one human-readable name per column."""

        names: list[str] = []
        if self.kind == "summary":
            for stat in self.stats:
                for tract in self.tract_names:
                    for metric in self.metric_names:
                        names.append(f"{tract}|{metric}|{stat}")
        elif self.kind == "metric":
            metric = self.metric_names[self.metric_index or 0]
            for tract in self.tract_names:
                names.extend(f"{tract}|{metric}|node{i:03d}" for i in range(self.n_points))
        else:
            for tract in self.tract_names:
                for metric in self.metric_names:
                    names.extend(
                        f"{tract}|{metric}|node{i:03d}" for i in range(self.n_points)
                    )
        names.extend(self.missing_pattern_columns)
        names.extend(self.covariate_columns)
        if len(names) != self.total_features:
            raise ValueError(f"layout produced {len(names)} names for {self.total_features}")
        return names

    def validate(self, n_features: int) -> None:
        if n_features != self.total_features:
            raise ValueError(
                f"feature matrix has {n_features} columns but layout expects "
                f"{self.total_features}"
            )

    def describe(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "n_tracts": self.n_tracts,
            "n_metrics": self.n_metrics,
            "n_points": self.n_points,
            "metric_index": self.metric_index,
            "stats": list(self.stats),
            "has_covariates": self.has_covariates,
            "n_feature_columns": self.n_feature_columns,
            "total_features": self.total_features,
        }

    def __str__(self) -> str:
        return (
            f"FeatureLayout({self.kind}, {self.n_tracts} tracts x "
            f"{self.n_points} points x {self.n_metrics} metrics, "
            f"{self.total_features} columns)"
        )


def view_for_layout(
    view: str,
    metric_names: Iterable[str],
    tract_names: Iterable[str],
    n_points: int,
    *,
    has_covariates: bool,
    has_missing_pattern: bool = False,
) -> FeatureLayout:
    """Build a layout matching the view string used by the feature builders."""

    metrics = tuple(metric_names)
    tracts = tuple(tract_names)
    normalized = str(view).upper()
    if normalized == "PROFILE":
        return FeatureLayout(
            kind="profile",
            tract_names=tracts,
            metric_names=metrics,
            n_points=n_points,
            has_covariates=has_covariates,
            has_missing_pattern=has_missing_pattern,
        )
    if normalized == "SUMMARY":
        return FeatureLayout(
            kind="summary",
            tract_names=tracts,
            metric_names=metrics,
            n_points=n_points,
            stats=("mean", "std", "slope", "area"),
            has_covariates=has_covariates,
            has_missing_pattern=has_missing_pattern,
        )
    if normalized in {name.upper() for name in metrics}:
        return FeatureLayout(
            kind="metric",
            tract_names=tracts,
            metric_names=metrics,
            n_points=n_points,
            metric_index=[name.upper() for name in metrics].index(normalized),
            has_covariates=has_covariates,
            has_missing_pattern=has_missing_pattern,
        )
    raise ValueError(f"unknown feature view {view!r}")


def expected_feature_count(
    view: str,
    metric_names: Iterable[str],
    tract_names: Iterable[str],
    n_points: int,
    *,
    include_covariates: bool,
    include_missing_pattern: bool = False,
) -> int:
    """Column count a builder is expected to produce for a given view."""

    layout = view_for_layout(
        view,
        metric_names,
        tract_names,
        n_points,
        has_covariates=include_covariates,
        has_missing_pattern=include_missing_pattern,
    )
    return layout.total_features
