"""Parser and subcommand wiring tests.

Two subcommands were silently broken here before: ``interpret`` imported a
name that was not exported, and ``python -m dit.cli`` had no ``__main__``.
Neither failure showed up in the model tests, so the wiring is asserted
directly rather than inferred from behaviour.
"""

from __future__ import annotations

import json

import pytest

from dit.cli.main import _config_from_args, build_parser


def _parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_every_subcommand_binds_a_handler() -> None:
    required: dict[str, list[str]] = {
        "evaluate": ["--synthetic"],
        "ablation": ["--synthetic"],
        "interpret": ["--synthetic"],
        "fetch": ["--url", "https://example.org/a.mat", "--out", "a.mat"],
        "info": [],
    }
    for command, extra in required.items():
        args = _parse([command, *extra])
        assert callable(args.func), f"{command} has no handler"


def test_module_entry_point_is_importable() -> None:
    import dit.cli.__main__ as entry

    assert callable(entry.main)


def test_deep_flags_reach_the_config() -> None:
    args = _parse(
        [
            "evaluate",
            "--synthetic",
            "--model",
            "tract_transformer",
            "--alignment",
            "coral",
            "--deep-epochs",
            "33",
            "--deep-batch-size",
            "9",
            "--deep-patience",
            "7",
            "--deep-d-model",
            "48",
            "--deep-layers",
            "2",
            "--deep-calibration",
            "sigmoid",
            "--device",
            "cpu",
        ]
    )
    config = _config_from_args(args)
    assert config.model == "tract_transformer"
    assert config.alignment == "coral"
    assert config.deep_epochs == 33
    assert config.deep_batch_size == 9
    assert config.deep_patience == 7
    assert config.deep_d_model == 48
    assert config.deep_layers == 2
    assert config.deep_calibration == "sigmoid"


def test_alignment_choices_are_constrained() -> None:
    for alignment in ("none", "coral", "mmd", "dann"):
        assert _config_from_args(_parse(["evaluate", "--synthetic", "--alignment", alignment])).alignment == alignment
    with pytest.raises(SystemExit):
        _parse(["evaluate", "--synthetic", "--alignment", "gmm"])


def test_calibration_defaults_to_unset() -> None:
    # Unset rather than "none": the ensemble uses the difference to decide
    # whether to calibrate a deep member on its behalf.
    config = _config_from_args(_parse(["evaluate", "--synthetic"]))
    assert config.deep_calibration is None
    for choice in ("none", "temperature", "sigmoid"):
        assert (
            _config_from_args(_parse(["evaluate", "--synthetic", "--deep-calibration", choice])).deep_calibration
            == choice
        )
    with pytest.raises(SystemExit):
        _parse(["evaluate", "--synthetic", "--deep-calibration", "isotonic"])


def test_threshold_choices_are_constrained() -> None:
    assert _config_from_args(_parse(["evaluate", "--synthetic", "--threshold", "balanced"])).threshold_criterion == "balanced"
    with pytest.raises(SystemExit):
        _parse(["evaluate", "--synthetic", "--threshold", "loosest"])


def test_strategy_and_task_are_constrained() -> None:
    config = _config_from_args(_parse(["evaluate", "--synthetic", "--strategy", "loso", "--task", "multiclass"]))
    assert config.split_strategy == "loso"
    assert config.task == "multiclass"
    with pytest.raises(SystemExit):
        _parse(["evaluate", "--synthetic", "--strategy", "random"])


def test_ensemble_flags_reach_the_config() -> None:
    args = _parse(
        [
            "evaluate",
            "--synthetic",
            "--model",
            "ensemble",
            "--ensemble-models",
            " linear_svm , logistic ,",
            "--ensemble-weighting",
            "equal",
        ]
    )
    config = _config_from_args(args)
    assert config.ensemble_models == ("linear_svm", "logistic")
    assert config.ensemble_weighting == "equal"


def test_ensemble_weighting_choices_are_constrained() -> None:
    for weighting in ("equal", "inner_score"):
        assert _config_from_args(
            _parse(["evaluate", "--synthetic", "--ensemble-weighting", weighting])
        ).ensemble_weighting == weighting
    with pytest.raises(SystemExit):
        _parse(["evaluate", "--synthetic", "--ensemble-weighting", "greedy"])


def test_empty_ensemble_lineup_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one base model"):
        _config_from_args(_parse(["evaluate", "--synthetic", "--ensemble-models", ""]))


def test_ensemble_rejects_a_repeated_model() -> None:
    from dit.evaluation.experiment import ExperimentConfig

    with pytest.raises(ValueError, match="must not repeat"):
        ExperimentConfig(ensemble_models=("logistic", "logistic"))


def test_interpret_exports_the_panel_renderer() -> None:
    from dit.interpret import render_metric_panel

    assert callable(render_metric_panel)


def test_version_flag_reports_the_package_version(capsys) -> None:
    import dit

    with pytest.raises(SystemExit) as raised:
        _parse(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.strip() == f"dit-afq {dit.__version__}"


def test_fetch_requires_both_url_and_out() -> None:
    with pytest.raises(SystemExit):
        _parse(["fetch", "--url", "https://example.org/a.mat"])


# --- optional-torch startup (regression: CLI used to import torch eagerly) ----

import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

_REPO = str(Path(__file__).resolve().parents[1])

# A meta-path finder that makes only ``import torch`` fail, the way a real
# no-torch install does.  ``sys.modules['torch'] = None`` is NOT equivalent: it
# hands scipy a None module and breaks its array-API probe.
_BLOCK_TORCH = """
import sys, importlib.abc
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ModuleNotFoundError('No module named ' + repr(name), name=name)
        return None
sys.meta_path.insert(0, _Block())
"""


def _run_without_torch(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_TORCH + body],
        capture_output=True,
        text=True,
        cwd=_REPO,
        timeout=300,
    )


def test_cli_imports_without_torch() -> None:
    result = _run_without_torch(
        "import sys\n"
        "from dit.cli.main import build_parser\n"
        "build_parser()\n"
        "print('torch_loaded', 'torch' in sys.modules)\n"
    )
    assert result.returncode == 0, result.stderr
    assert "torch_loaded False" in result.stdout


def test_classical_evaluate_runs_without_torch(tmp_path) -> None:
    result = _run_without_torch(
        "from dit.cli.main import main\n"
        f"raise SystemExit(main(['evaluate','--synthetic','--n-samples','60',"
        f"'--n-sites','3','--model','linear_svm','--task','binary',"
        f"'--n-splits','2','--inner-splits','2','--out',r'{tmp_path}','--quiet']))\n"
    )
    assert result.returncode == 0, result.stderr


def test_deep_model_without_torch_gives_install_hint(tmp_path) -> None:
    result = _run_without_torch(
        "from dit.cli.main import main\n"
        f"main(['evaluate','--synthetic','--n-samples','60','--n-sites','3',"
        f"'--model','tract_transformer','--task','binary','--n-splits','2',"
        f"'--inner-splits','2','--out',r'{tmp_path}','--quiet'])\n"
    )
    assert result.returncode != 0
    assert "PyTorch" in result.stderr and ".[torch]" in result.stderr


def test_module_dit_entrypoint_runs_smoke(tmp_path) -> None:
    # ``python -m dit --smoke`` is the acceptance contract for a self-contained
    # end-to-end run; assert the module entry exists and evaluate --smoke works.
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "dit", "evaluate", "--smoke", "--out", str(tmp_path), "--quiet"],
        capture_output=True,
        text=True,
        cwd=_REPO,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "evaluation.json").exists()
    assert (tmp_path / "rad_scores.csv").exists()


def test_matrix_runs_dual_task_dual_split(tmp_path) -> None:
    from dit.cli.main import main

    assert main(["matrix", "--smoke", "--out", str(tmp_path), "--quiet"]) == 0
    summary = json.loads((tmp_path / "matrix_summary.json").read_text(encoding="utf-8"))
    combos = {(run["task"], run["split_strategy"]) for run in summary["runs"]}
    assert combos == {
        ("binary", "stratified"),
        ("binary", "loso"),
        ("multiclass", "stratified"),
        ("multiclass", "loso"),
    }
    # Binary group must drop MCI; multiclass keeps it.
    binary = json.loads((tmp_path / "binary_stratified" / "evaluation.json").read_text(encoding="utf-8"))
    assert binary["results"][0]["aggregate"]["class_names"] == ["NC", "AD"]
    multi = json.loads((tmp_path / "multiclass_loso" / "evaluation.json").read_text(encoding="utf-8"))
    assert multi["results"][0]["aggregate"]["class_names"] == ["NC", "MCI", "AD"]


def test_rad_scores_are_out_of_fold_per_subject(tmp_path) -> None:
    import csv

    from dit.cli.main import main

    assert main(["evaluate", "--smoke", "--out", str(tmp_path), "--quiet"]) == 0
    with (tmp_path / "rad_scores.csv").open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert "disease_score" in reader.fieldnames
    assert "subject_id" in reader.fieldnames
    assert rows, "rad_scores must have one row per subject"
    for row in rows:
        score = float(row["disease_score"])
        assert 0.0 <= score <= 1.0


def test_ablation_writes_csv_and_strict_json(tmp_path) -> None:
    import csv

    from dit.cli.main import main

    assert main(["ablation", "--smoke", "--out", str(tmp_path), "--quiet"]) == 0
    with (tmp_path / "ablation_table.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3  # three covariate strategies at fixed model/view
    # The table JSON must be valid strict JSON (no bare NaN).
    table = json.loads((tmp_path / "ablation_table.json").read_text(encoding="utf-8"))
    assert len(table) == 3
