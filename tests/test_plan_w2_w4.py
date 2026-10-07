"""Regression tests for the W2/W3/W4 work packages of the optimization plan.

Data-snapshot identity in fold provenance and paired comparisons (W2), the
all-NaN-column contract (W2), negative-control input views (W2), inner-search
budget accounting (W3), and the fit/predict deployment closure (W4).
"""

from __future__ import annotations

import numpy as np
import pytest

from dit.data.schema import DatasetBundle
from dit.data.synthetic import make_synthetic_bundle
from dit.evaluation.experiment import ExperimentConfig, run_experiment
from dit.evaluation.provenance import fold_provenance
from dit.evaluation.site_balance import paired_fold_comparison
from dit.models.classical import build_control_matrix, make_search_estimator
from dit.data.selection import SparseBlockSelector
from dit.data.layout import view_for_layout


def _bundle(seed: int = 3, n: int = 60) -> DatasetBundle:
    return make_synthetic_bundle(n_samples=n, n_tracts=3, n_points=10, n_metrics=2, seed=seed)


def _fold(fold_id: str, digest: str, data_digest: str | None, metric_value: float) -> dict[str, object]:
    fold: dict[str, object] = {
        "fold": fold_id,
        "test_index_digest": digest,
        "task": "binary",
        "split_strategy": "stratified",
        "test_class_counts": {"0": 2, "1": 2},
        "n_test": 4,
        "balanced_accuracy": metric_value,
    }
    if data_digest is not None:
        fold["data_digest"] = data_digest
    return fold


class TestDataIdentity:
    def test_digest_is_order_and_content_sensitive(self) -> None:
        bundle = _bundle(seed=5)
        same = _bundle(seed=5)
        assert bundle.data_digest == same.data_digest

        shifted = DatasetBundle(
            X=np.roll(bundle.X, 1, axis=0),
            site=np.roll(bundle.site, 1),
            age=None if bundle.age is None else np.roll(bundle.age, 1),
            sex=None if bundle.sex is None else np.roll(bundle.sex, 1),
            tract_names=bundle.tract_names,
            metric_names=bundle.metric_names,
        )
        assert shifted.data_digest != bundle.data_digest

        tweaked = bundle.X.copy()
        tweaked[0, 0, 0, 0] += 1e-3
        different_values = DatasetBundle(
            X=tweaked,
            site=bundle.site,
            age=bundle.age,
            sex=bundle.sex,
            tract_names=bundle.tract_names,
            metric_names=bundle.metric_names,
        )
        assert different_values.data_digest != bundle.data_digest

    def test_fold_manifest_and_comparison_carry_the_snapshot(self) -> None:
        bundle = _bundle(seed=7)
        view = bundle.task_view("binary")
        manifest = fold_provenance(
            np.arange(4),
            view.y,
            fold_id=0,
            task="binary",
            split_strategy="stratified",
            data_digest=view.data_digest,
        )
        assert manifest["data_digest"] == view.data_digest

        matched = paired_fold_comparison(
            [_fold("0", "d0", "snap-a", 0.6), _fold("1", "d1", "snap-a", 0.7)],
            [_fold("0", "d0", "snap-a", 0.65), _fold("1", "d1", "snap-a", 0.7)],
        )
        assert matched["status"] != "not_comparable"

        mismatched = paired_fold_comparison(
            [_fold("0", "d0", "snap-a", 0.6), _fold("1", "d1", "snap-a", 0.7)],
            [_fold("0", "d0", "snap-b", 0.65), _fold("1", "d1", "snap-b", 0.7)],
        )
        assert mismatched["status"] == "not_comparable"
        assert "data snapshot" in mismatched["reason"]

    def test_comparison_still_allows_legacy_manifests_without_digests(self) -> None:
        legacy = paired_fold_comparison(
            [_fold("0", "d0", None, 0.6), _fold("1", "d1", None, 0.7)],
            [_fold("0", "d0", None, 0.65), _fold("1", "d1", None, 0.7)],
        )
        assert legacy["status"] != "not_comparable"


class TestAllNanColumns:
    def _matrix_with_dead_column(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(2)
        matrix = rng.normal(size=(60, 6))
        matrix[:, 3] = np.nan
        return matrix, np.repeat([0, 1], 30)

    def test_selector_survives_a_fully_missing_column(self) -> None:
        matrix, labels = self._matrix_with_dead_column()
        layout = view_for_layout(
            "summary",
            ["metric_a", "metric_b"],
            [f"tract_{i}" for i in range(3)],
            n_points=1,
            has_covariates=False,
        )
        # 3 tracts x 2 metrics x 4 stats = 24 summary columns; pad the matrix
        # up to the layout width with well-behaved columns.
        rng = np.random.default_rng(4)
        matrix = np.column_stack([matrix, rng.normal(size=(60, 18))])
        selector = SparseBlockSelector(layout, seed=1)
        selector.fit(matrix, labels)
        assert selector.n_selected > 0

    def test_pipeline_keeps_column_count_through_the_imputer(self) -> None:
        matrix, labels = self._matrix_with_dead_column()
        from sklearn.model_selection import KFold

        search = make_search_estimator("logistic", cv=KFold(n_splits=3, shuffle=True, random_state=0))
        search.fit(matrix, labels)
        imputer = search.best_estimator_.named_steps["imputer"]
        transformed = imputer.transform(matrix)
        # Dropping the dead column would shift every later coefficient away
        # from its anatomical column. keep_empty_features keeps it in place
        # (filled with 0); add_indicator appends its indicator *after* the
        # feature columns, so the leading width stays aligned with the layout.
        assert transformed.shape[1] == matrix.shape[1] + 1
        assert np.all(transformed[:, 3] == 0)


class TestControlViews:
    def _config(self, **overrides: object) -> ExperimentConfig:
        values: dict[str, object] = {
            "model": "logistic",
            "n_splits": 2,
            "inner_splits": 2,
            "seed": 1,
            "control_view": "demographics",
        }
        values.update(overrides)
        return ExperimentConfig(**values)  # type: ignore[arg-type]

    def test_demographics_control_runs_and_reports_itself(self) -> None:
        result = run_experiment(_bundle(seed=9), self._config())
        assert result.aggregate["control_view"] == "demographics"
        assert result.feature_names == ("age", "sex")

    def test_missingness_control_names_its_tracts(self) -> None:
        result = run_experiment(
            _bundle(seed=9), self._config(control_view="missingness")
        )
        assert result.feature_names[0].startswith("missing|")
        assert len(result.feature_names) == 3

    def test_control_matrix_builders(self) -> None:
        bundle = _bundle(seed=11)
        matrix, names = build_control_matrix(bundle, "demographics")
        assert matrix.shape == (bundle.n_samples, 2)
        assert names == ["age", "sex"]
        matrix, names = build_control_matrix(bundle, "missingness")
        assert matrix.shape == (bundle.n_samples, bundle.n_tracts)
        with pytest.raises(ValueError, match="unknown control view"):
            build_control_matrix(bundle, "imaging")

    def test_control_combinations_are_refused(self) -> None:
        from dit.evaluation.experiment import run_configured_experiment

        bundle = _bundle(seed=13)
        with pytest.raises(ValueError, match="classical models only"):
            run_experiment(bundle, self._config(model="tract_transformer"))
        with pytest.raises(ValueError, match="cannot enter an ensemble"):
            run_configured_experiment(bundle, self._config(model="ensemble"))
        with pytest.raises(ValueError, match="selection does not apply"):
            run_experiment(bundle, self._config(enable_selection=True))
        with pytest.raises(ValueError, match="residualization does not apply"):
            run_experiment(bundle, self._config(covariate_strategy="residualize"))
        with pytest.raises(ValueError, match="unknown control_view"):
            run_experiment(bundle, self._config(control_view="imaging"))


class TestBudgetAccounting:
    def test_classical_folds_report_inner_budget(self) -> None:
        result = run_experiment(
            _bundle(seed=15),
            ExperimentConfig(model="linear_svm", n_splits=2, inner_splits=2, seed=1),
        )
        for fold in result.folds:
            budget = fold["inner_budget"]
            assert budget["n_candidates"] == 9  # linear SVM C grid
            assert budget["n_inner_folds"] == 2
            assert budget["n_fits"] == 9 * 2 + 1
            assert budget["candidate_fit_seconds"] >= 0.0

    def test_deep_folds_report_the_requested_epoch_budget(self) -> None:
        pytest.importorskip("torch", reason="deep budget accounting requires torch")
        result = run_experiment(
            make_synthetic_bundle(n_samples=50, n_tracts=2, n_points=8, n_metrics=2, seed=6),
            ExperimentConfig(
                model="tract_transformer",
                n_splits=2,
                inner_splits=2,
                deep_epochs=2,
                deep_batch_size=8,
                deep_patience=1,
                deep_d_model=32,
                deep_layers=1,
                seed=1,
            ),
        )
        for fold in result.folds:
            assert fold["epochs_requested"] == 2
            assert fold["training"]["epochs_run"] <= 2


class TestDeploymentClosure:
    def _fit_model(self, bundle: DatasetBundle):
        from dit.deployment import fit_deployment_model

        config = ExperimentConfig(
            model="linear_svm", n_splits=2, inner_splits=2, seed=1
        )
        return fit_deployment_model(bundle, config)

    def test_fit_save_load_predict_roundtrip(self, tmp_path) -> None:
        from dit.deployment import (
            load_deployment_artifact,
            predict_unlabeled,
            save_deployment_artifact,
        )

        bundle = _bundle(seed=17, n=80)
        model = self._fit_model(bundle)
        artifact = save_deployment_artifact(model, tmp_path / "model.joblib")

        loaded = load_deployment_artifact(artifact)
        assert loaded.best_params == model.best_params
        assert loaded.label_map == model.label_map
        assert loaded.data_digest == model.data_digest
        # Saving always writes the checksum sidecar loading verifies.
        assert artifact.with_name(artifact.name + ".sha256").is_file()

        unlabeled = DatasetBundle(
            X=bundle.X,
            age=bundle.age,
            sex=bundle.sex,
            site=bundle.site,
            tract_names=bundle.tract_names,
            metric_names=bundle.metric_names,
        )
        outcome = predict_unlabeled(unlabeled, loaded)
        probabilities = outcome["probabilities"]
        assert probabilities.shape == (bundle.n_samples, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert np.array_equal(outcome["predictions"], probabilities.argmax(axis=1))
        assert set(outcome["predicted_names"]) <= {"NC", "AD"}

    def test_load_refuses_a_tampered_artifact(self, tmp_path) -> None:
        from dit.deployment import load_deployment_artifact, save_deployment_artifact

        model = self._fit_model(_bundle(seed=29, n=80))
        artifact = save_deployment_artifact(model, tmp_path / "model.joblib")
        payload = bytearray(artifact.read_bytes())
        payload[len(payload) // 2] ^= 0xFF
        artifact.write_bytes(bytes(payload))
        with pytest.raises(ValueError, match="checksum mismatch"):
            load_deployment_artifact(artifact)

    def test_load_refuses_a_missing_sidecar(self, tmp_path) -> None:
        from dit.deployment import load_deployment_artifact, save_deployment_artifact

        model = self._fit_model(_bundle(seed=31, n=80))
        artifact = save_deployment_artifact(model, tmp_path / "model.joblib")
        artifact.with_name(artifact.name + ".sha256").unlink()
        # A pickle without its checksum is an unverifiable executable; loading
        # must refuse rather than fall back to blind unpickling.
        with pytest.raises(FileNotFoundError, match="sidecar"):
            load_deployment_artifact(artifact)

    def test_load_refuses_an_empty_sidecar_with_a_clear_error(self, tmp_path) -> None:
        from dit.deployment import load_deployment_artifact, save_deployment_artifact

        model = self._fit_model(_bundle(seed=32, n=80))
        artifact = save_deployment_artifact(model, tmp_path / "model.joblib")
        # A truncated sidecar is still fail-closed, but must not surface as an
        # IndexError from the parse — it is a checksum error like any other.
        artifact.with_name(artifact.name + ".sha256").write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="sidecar is empty"):
            load_deployment_artifact(artifact)

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("\ufeff", id="utf8-bom-prefix"),
            pytest.param("中文校验和  model.joblib\n", id="non-ascii-token"),
            pytest.param("deadbeef  model.joblib\n", id="too-short-digest"),
        ],
    )
    def test_load_refuses_a_malformed_sidecar_with_a_clear_error(
        self, tmp_path, content: str
    ) -> None:
        from dit.deployment import load_deployment_artifact, save_deployment_artifact

        model = self._fit_model(_bundle(seed=33, n=80))
        artifact = save_deployment_artifact(model, tmp_path / "model.joblib")
        # hmac.compare_digest raises TypeError on non-ASCII strings, so a BOM
        # or garbled token must be rejected as malformed up front instead of
        # surfacing as "comparing strings with non-ASCII characters is not
        # supported".
        artifact.with_name(artifact.name + ".sha256").write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="malformed"):
            load_deployment_artifact(artifact)

    def test_load_refuses_modules_outside_the_allowlist(self, tmp_path) -> None:
        import datetime
        import hashlib
        import pickle as pickle_module

        from dit.deployment import ARTIFACT_VERSION, _digest_path, load_deployment_artifact

        # A correct sidecar plus a class reference from a non-allowlisted
        # module: the checksum passes, the unpickler must still refuse the
        # global before the class can be instantiated.
        payload = pickle_module.dumps(
            {"artifact_version": ARTIFACT_VERSION, "stamp": datetime.datetime}
        )
        target = tmp_path / "foreign.pkl"
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        _digest_path(target).write_text(f"{digest}  {target.name}\n", encoding="utf-8")
        with pytest.raises(pickle_module.UnpicklingError, match="outside the allowlist"):
            load_deployment_artifact(target)

    def test_load_refuses_builtins_globals(self, tmp_path) -> None:
        import hashlib
        import pickle as pickle_module

        from dit.deployment import ARTIFACT_VERSION, _digest_path, load_deployment_artifact

        # ``builtins`` is the most dangerous root imaginable (eval, exec,
        # open, ...); the allowlist test above only proves a non-listed root
        # is refused, so pin the canonical one explicitly — a weakened
        # allowlist that re-admits builtins must fail this test.
        payload = pickle_module.dumps(
            {"artifact_version": ARTIFACT_VERSION, "stamp": eval}
        )
        target = tmp_path / "builtins.pkl"
        target.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        _digest_path(target).write_text(f"{digest}  {target.name}\n", encoding="utf-8")
        with pytest.raises(pickle_module.UnpicklingError, match="outside the allowlist"):
            load_deployment_artifact(target)

    def test_deep_and_ensemble_and_control_are_refused(self) -> None:
        from dit.deployment import fit_deployment_model

        bundle = _bundle(seed=19)
        with pytest.raises(ValueError, match="classical models only"):
            fit_deployment_model(
                bundle, ExperimentConfig(model="tract_transformer", seed=1)
            )
        with pytest.raises(ValueError, match="ensemble"):
            fit_deployment_model(bundle, ExperimentConfig(model="ensemble", seed=1))
        with pytest.raises(ValueError, match="not deployable"):
            fit_deployment_model(
                bundle,
                ExperimentConfig(model="logistic", seed=1, control_view="demographics"),
            )

    def test_column_contract_is_enforced_at_predict_time(self) -> None:
        from dit.deployment import predict_unlabeled

        model = self._fit_model(_bundle(seed=23, n=80))
        wider = make_synthetic_bundle(
            n_samples=30, n_tracts=5, n_points=12, n_metrics=3, seed=2
        )
        with pytest.raises(ValueError, match="input columns"):
            predict_unlabeled(wider, model)

    def test_cli_fit_and_predict_roundtrip(self, tmp_path) -> None:
        from dit.cli.main import main

        artifact = tmp_path / "model.joblib"
        predictions_csv = tmp_path / "predictions.csv"
        rc = main(
            [
                "fit", "--synthetic", "--n-samples", "80",
                "--n-splits", "2", "--inner-splits", "2",
                "--artifact", str(artifact),
            ]
        )
        assert rc == 0
        assert artifact.is_file()

        rc = main(
            [
                "predict", "--synthetic", "--n-samples", "80",
                "--artifact", str(artifact), "--out", str(predictions_csv),
            ]
        )
        assert rc == 0
        lines = predictions_csv.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("subject_id,predicted_label,predicted_class,p_")
        assert len(lines) == 81  # header + 80 subjects
