"""Direct tests for ``dit.interpret``.

The package previously had only end-to-end coverage through the CLI smoke
test, so the column bookkeeping (which column is which tract/node/metric), the
literature-region check, and the covariate exclusion in coefficient importance
were untested.  Layouts here are hand-built so the tract names can carry the
published region substrings that the synthetic bundle's ``tract_NN`` names do
not.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dit.data.layout import FeatureLayout, view_for_layout
from dit.data.synthetic import make_synthetic_bundle
from dit.interpret.heatmap import literature_hits, node_matrix, render_metric_panel
from dit.interpret.importance import (
    block_permutation_importance,
    node_importance_from_coefficients,
)
from dit.models.classical import (
    build_feature_matrix,
    describe_feature_layout,
    make_search_estimator,
)

# "Left Anterior Thalamic Radiation (ATR)" intentionally matches two published
# regions ("thalamic" and "atr"); "Motor Tract" matches none.
TRACTS = ("Left Uncinate Fasciculus", "Left Anterior Thalamic Radiation (ATR)", "Motor Tract")
METRICS = ("FA", "MD")


def _profile_layout(n_points: int = 6) -> FeatureLayout:
    return view_for_layout(
        "profile", METRICS, TRACTS, n_points=n_points, has_covariates=True
    )


def _row(tract: int, metric_index: int, node: int | None, magnitude: float = 0.9) -> dict[str, object]:
    metric = METRICS[metric_index]
    return {
        "block": f"{TRACTS[tract]}|{metric}",
        "tract_index": tract,
        "tract": TRACTS[tract],
        "metric_index": metric_index,
        "metric": metric,
        "stat": None,
        "node": node,
        "column": 0,
        "magnitude": magnitude,
    }


class TestNodeMatrix:
    def test_values_land_on_the_tract_by_node_grid(self) -> None:
        layout = _profile_layout()
        rows = [_row(0, 0, 3, 0.5), _row(1, 0, 5, 0.9), _row(2, 0, 0, 0.1)]
        matrix = node_matrix(rows, layout, metric_index=0)
        assert matrix.shape == (layout.n_tracts, layout.n_points)
        assert matrix[0, 3] == 0.5
        assert matrix[1, 5] == 0.9
        assert matrix[2, 0] == 0.1
        assert matrix[0, 0] == 0.0  # untouched cells stay zero
        assert matrix.sum() == pytest.approx(1.5)

    def test_other_metrics_are_dropped_not_averaged(self) -> None:
        # The docstring says non-requested metrics are "averaged out"; the
        # implementation drops them.  Pin the behaviour that ships so a change
        # either way is a conscious one.
        layout = _profile_layout()
        rows = [_row(0, 0, 1, 0.4), _row(0, 1, 1, 0.8)]
        matrix = node_matrix(rows, layout, metric_index=0)
        assert matrix[0, 1] == pytest.approx(0.4)
        assert matrix.sum() == pytest.approx(0.4)

    def test_profile_view_without_a_metric_stays_empty(self) -> None:
        # A multi-metric profile layout has no single default metric, so
        # requesting nothing places nothing rather than guessing metric 0.
        layout = _profile_layout()
        rows = [_row(0, 0, 1, 0.4), _row(0, 1, 2, 0.8)]
        assert not np.any(node_matrix(rows, layout))

    def test_metric_view_defaults_to_its_own_metric(self) -> None:
        layout = view_for_layout("MD", METRICS, TRACTS, n_points=6, has_covariates=False)
        rows = [_row(0, 1, 2, 0.4), _row(0, 0, 2, 0.9)]
        matrix = node_matrix(rows, layout)  # no metric_index: uses layout.metric_index == 1
        assert matrix[0, 2] == pytest.approx(0.4)
        assert matrix.sum() == pytest.approx(0.4)

    def test_rows_without_a_node_are_skipped(self) -> None:
        layout = _profile_layout()
        rows = [dict(_row(0, 0, 2, 0.4), node=None)]
        assert not np.any(node_matrix(rows, layout, metric_index=0))

    def test_summary_layout_has_no_node_axis(self) -> None:
        layout = view_for_layout("summary", METRICS, TRACTS, n_points=6, has_covariates=False)
        with pytest.raises(ValueError, match="node axis"):
            node_matrix([], layout)


class TestLiteratureHits:
    def test_peak_inside_the_published_interval_is_a_hit(self) -> None:
        layout = _profile_layout(n_points=100)
        # 0-based node 79 is the 80th node, inside the uncinate 75-100 window.
        result = literature_hits([_row(0, 0, 79)], layout)
        uncinate = next(r for r in result["regions"] if r["region_key"] == "uncinate")
        assert result["status"] == "computed"
        assert uncinate["peak_node"] == 80
        assert uncinate["expected_node_range"] == [75, 100]
        assert uncinate["hit"] is True
        assert result["n_hits"] == 1

    def test_peak_outside_the_interval_is_a_miss(self) -> None:
        layout = _profile_layout(n_points=100)
        result = literature_hits([_row(0, 0, 5)], layout)
        uncinate = next(r for r in result["regions"] if r["region_key"] == "uncinate")
        assert uncinate["peak_node"] == 6
        assert uncinate["hit"] is False
        assert result["n_hits"] == 0

    @pytest.mark.parametrize(("node", "expected"), [(74, True), (99, True), (73, False), (100, False)])
    def test_the_interval_is_inclusive_in_one_based_nodes(self, node: int, expected: bool) -> None:
        layout = _profile_layout(n_points=100)
        result = literature_hits([_row(0, 0, node)], layout)
        uncinate = next(r for r in result["regions"] if r["region_key"] == "uncinate")
        assert uncinate["hit"] is expected

    def test_peak_is_the_largest_magnitude_across_metrics(self) -> None:
        layout = _profile_layout(n_points=100)
        rows = [_row(0, 0, 90, 0.3), _row(0, 1, 80, 0.9)]
        result = literature_hits(rows, layout)
        uncinate = next(r for r in result["regions"] if r["region_key"] == "uncinate")
        assert uncinate["peak_node"] == 81

    def test_a_tract_can_match_more_than_one_region(self) -> None:
        layout = _profile_layout(n_points=100)
        result = literature_hits([_row(1, 0, 0)], layout)  # node 1 is inside ATR/Thalamic 1-13
        atr_rows = [r for r in result["regions"] if r["tract"].endswith("(ATR)")]
        assert {r["region_key"] for r in atr_rows} == {"atr", "thalamic"}
        assert all(r["hit"] for r in atr_rows)
        assert result["n_regions"] == len(result["regions"])
        assert result["n_hits"] == sum(1 for r in result["regions"] if r["hit"])

    def test_a_tract_with_no_rows_reports_no_peak_and_no_hit(self) -> None:
        layout = _profile_layout(n_points=100)
        result = literature_hits([_row(2, 0, 3)], layout)  # only the motor tract has rows
        uncinate = next(r for r in result["regions"] if r["region_key"] == "uncinate")
        assert uncinate["peak_node"] is None
        assert uncinate["hit"] is False
        assert result["status"] == "computed"

    def test_no_matching_tract_names_report_no_regions(self) -> None:
        layout = view_for_layout("profile", METRICS, ("tract_a", "tract_b"), n_points=6, has_covariates=False)
        result = literature_hits([_row(0, 0, 1, 0.5)], layout)
        assert result["status"] == "no_matching_tracts"
        assert result["n_regions"] == 0
        assert result["regions"] == []

    def test_summary_view_has_no_node_axis(self) -> None:
        layout = view_for_layout("summary", METRICS, TRACTS, n_points=6, has_covariates=False)
        result = literature_hits([], layout)
        assert result["status"] == "unavailable"
        assert "node axis" in str(result["reason"])


class TestNodeImportanceFromCoefficients:
    def _fitted_logistic(self, layout: FeatureLayout, *, covariate_driven: bool) -> Pipeline:
        rng = np.random.default_rng(5)
        features = rng.normal(size=(60, layout.total_features))
        if covariate_driven:
            # The label is a pure function of the trailing covariate columns,
            # so the largest coefficients land on demographics.
            labels = (features[:, layout.n_feature_columns] > 0).astype(int)
        else:
            labels = rng.integers(0, 2, 60)
        pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
        )
        pipeline.fit(features, labels)
        return pipeline

    def test_every_anatomical_column_is_ranked_by_magnitude(self) -> None:
        layout = _profile_layout()
        pipeline = self._fitted_logistic(layout, covariate_driven=False)
        rows = node_importance_from_coefficients(pipeline, layout)
        # One row per anatomical column, each exactly once.
        assert len(rows) == layout.n_feature_columns
        assert sorted(row["column"] for row in rows) == list(range(layout.n_feature_columns))
        # Magnitudes are the model's own, and the list is ranked.
        coefficients = np.abs(pipeline.named_steps["model"].coef_)[0]
        by_column = {row["column"]: row["magnitude"] for row in rows}
        for column in range(layout.n_feature_columns):
            assert by_column[column] == pytest.approx(float(coefficients[column]))
        assert all(
            rows[i]["magnitude"] >= rows[i + 1]["magnitude"] for i in range(len(rows) - 1)
        )
        # Profile blocks carry a node offset and a metric name.
        assert rows[0]["node"] is not None
        assert rows[0]["metric"] in METRICS
        assert rows[0]["tract"] in TRACTS

    def test_covariate_columns_are_excluded_even_when_they_dominate(self) -> None:
        layout = _profile_layout()
        pipeline = self._fitted_logistic(layout, covariate_driven=True)
        coefficients = np.abs(pipeline.named_steps["model"].coef_)[0]
        assert int(np.argmax(coefficients)) >= layout.n_feature_columns
        rows = node_importance_from_coefficients(pipeline, layout)
        assert rows
        assert all(row["column"] < layout.n_feature_columns for row in rows)

    def test_calibrated_svm_search_yields_nonzero_node_importance(self) -> None:
        # The CLI path: GridSearchCV over a calibrated linear SVM.  Coefficient
        # resolution has to descend through the calibration wrapper.
        bundle = make_synthetic_bundle(n_samples=60, n_tracts=4, n_points=12, n_metrics=2, seed=17)
        view = bundle.task_view("binary")
        features = build_feature_matrix(view, view="profile", smooth_window=5, include_covariates=True)
        layout = describe_feature_layout(view, view="profile", include_covariates=True)
        search = make_search_estimator(
            "linear_svm", cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=17), seed=17
        )
        search.fit(features, view.y)
        rows = node_importance_from_coefficients(search, layout)
        assert len(rows) == layout.n_feature_columns
        assert any(row["magnitude"] > 0 for row in rows)
        assert all(row["column"] < layout.n_feature_columns for row in rows)

    def test_summary_blocks_carry_no_node_offset(self) -> None:
        bundle = make_synthetic_bundle(n_samples=60, n_tracts=3, n_points=8, n_metrics=2, seed=17)
        view = bundle.task_view("multiclass")
        features = build_feature_matrix(view, view="summary", smooth_window=5, include_covariates=True)
        layout = describe_feature_layout(view, view="summary", include_covariates=True)
        search = make_search_estimator(
            "logistic", cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=17), seed=17
        )
        search.fit(features, view.y)
        rows = node_importance_from_coefficients(search, layout)
        assert len(rows) == layout.n_feature_columns
        assert all(row["node"] is None for row in rows)
        assert any(row["magnitude"] > 0 for row in rows)

    def test_a_model_without_coefficients_is_refused(self) -> None:
        layout = _profile_layout()
        rng = np.random.default_rng(0)
        pipeline = Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=0))])
        pipeline.fit(rng.normal(size=(40, layout.total_features)), rng.integers(0, 2, 40))
        with pytest.raises(ValueError, match="coef_"):
            node_importance_from_coefficients(pipeline, layout)


class TestBlockPermutationImportance:
    def test_every_anatomical_block_is_scored_once(self) -> None:
        bundle = make_synthetic_bundle(n_samples=60, n_tracts=4, n_points=12, n_metrics=2, seed=17)
        view = bundle.task_view("binary")
        features = build_feature_matrix(view, view="profile", smooth_window=5, include_covariates=True)
        layout = describe_feature_layout(view, view="profile", include_covariates=True)
        search = make_search_estimator(
            "logistic", cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=17), seed=17
        )
        search.fit(features, view.y)
        result = block_permutation_importance(
            search, features, view.y, layout, n_repeats=3, seed=17
        )
        blocks = layout.anatomical_blocks()
        assert len(result["blocks"]) == len(blocks)
        assert sorted(row["block"] for row in result["blocks"]) == sorted(b.name for b in blocks)
        assert 0.0 <= result["baseline_score"] <= 1.0
        assert result["n_holdout"] == view.n_samples - int(view.n_samples * 0.7)
        assert result["top_blocks"][0] == result["blocks"][0]["block"]
        assert all(np.isfinite(row["importance"]) for row in result["blocks"])


class TestMetricPanelFilenames:
    """Interpretation artefacts must stay inside the output directory.

    Metric names come from ``.mat`` struct field names — arbitrary strings —
    and the heatmap writer is the one place they reach the filesystem.
    """

    def _layout(self, metric_names: tuple[str, ...]) -> FeatureLayout:
        return FeatureLayout(
            kind="metric",
            tract_names=("L_UF", "R_UF"),
            metric_names=metric_names,
            n_points=4,
            metric_index=0,
        )

    def _rows(self, n_metrics: int = 1) -> list[dict[str, object]]:
        return [
            {"tract_index": tract, "metric_index": metric, "node": node, "magnitude": 1.0 + node}
            for metric in range(n_metrics)
            for tract in range(2)
            for node in range(4)
        ]

    @pytest.mark.parametrize(
        "metric_name",
        ["../../pwn", r"..\..\pwn", ".hidden", "a/b", r"C:\Windows\x", "..", "MD "],
    )
    def test_metric_names_cannot_escape_the_output_directory(self, tmp_path, metric_name) -> None:
        paths = render_metric_panel(self._rows(), self._layout((metric_name,)), out_dir=tmp_path)
        assert len(paths) == 1
        target = paths[0][1]
        assert target.parent == tmp_path
        stem = target.name[: -len(".png")]
        assert stem.startswith("importance_")
        assert "/" not in stem and "\\" not in stem and ":" not in stem
        assert not stem.removeprefix("importance_").startswith(".")
        assert ".." not in stem

    def test_colliding_metric_names_get_distinct_files(self, tmp_path) -> None:
        paths = render_metric_panel(self._rows(n_metrics=2), self._layout(("MD", "MD ")), out_dir=tmp_path)
        assert len(paths) == 2
        assert len({path.name for _, path in paths}) == 2
        assert all(path.parent == tmp_path for _, path in paths)

    def test_chained_collisions_all_get_distinct_files(self, tmp_path) -> None:
        # "_A" cleans to "A" (collides with metric 0), whose disambiguated
        # "A_2" collides with metric 1's natural stem: one suffix is not
        # enough, the third heatmap must not silently overwrite the second.
        paths = render_metric_panel(self._rows(n_metrics=3), self._layout(("A", "A_2", "_A")), out_dir=tmp_path)
        assert len(paths) == 3
        assert len({path.name for _, path in paths}) == 3
        assert all(path.parent == tmp_path for _, path in paths)
    def test_long_metric_names_are_truncated_not_fatal(self, tmp_path) -> None:
        # Windows rejects path components over 255 characters; a .mat field
        # name is arbitrary, so the stem is capped instead of raising OSError
        # halfway through the panel.
        paths = render_metric_panel(
            self._rows(), self._layout(("M" * 300,)), out_dir=tmp_path
        )
        assert len(paths) == 1
        target = paths[0][1]
        assert target.parent == tmp_path
        assert len(target.name) <= 120

    def test_case_colliding_metric_names_get_distinct_files(self, tmp_path) -> None:
        # .mat struct field names are case-sensitive and can coexist, but the
        # filesystem is not: without a case-folded key the second heatmap
        # would silently overwrite the first.
        paths = render_metric_panel(self._rows(n_metrics=2), self._layout(("FA", "fa")), out_dir=tmp_path)
        assert len(paths) == 2
        assert len({path.name for _, path in paths}) == 2
        assert len({path.name.casefold() for _, path in paths}) == 2
        assert all(path.parent == tmp_path for _, path in paths)


class TestInterpretRefusesUnimplementedFlags:
    """The interpretation path fits one model on every row, so the flags that
    only make sense inside evaluation folds must fail loudly instead of being
    silently ignored."""

    @pytest.mark.parametrize(
        ("extra", "expected"),
        (
            (["--select"], "--select"),
            (["--covariate", "residualize"], "residualize"),
            (["--control-view", "demographics"], "control-view"),
            (["--alignment", "coral"], "alignment"),
        ),
    )
    def test_refuses_flags_it_does_not_implement(self, tmp_path, extra, expected) -> None:
        from dit.cli.main import main

        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "interpret",
                    "--synthetic",
                    "--n-samples",
                    "40",
                    "--out",
                    str(tmp_path),
                    "--quiet",
                    *extra,
                ]
            )
        assert str(raised.value).startswith("error:")
        assert expected in str(raised.value)
