"""Config loading: the shipped YAML files must load, and bad ones must fail loudly.

``configs/*.yaml`` is the reproducibility entry point -- a run is regenerated
by rerunning the file that produced it -- so an example file with a stale key
name is a broken promise, and the loader's rejection branches are the only
thing standing between a typo and a silently ignored setting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dit.evaluation.experiment import ExperimentConfig
from dit.config import config_to_experiment, experiment_payload, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_shipped_configs_load_and_build_a_config() -> None:
    shipped = sorted(CONFIGS.glob("*.yaml"))
    assert shipped, "configs/ lost its example files"
    for path in shipped:
        payload = load_config(path)
        config = config_to_experiment(payload)
        # The payload round-trips through the same keys the dataclass accepts.
        assert set(payload["experiment"]) <= set(config.describe())


def test_missing_file_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config(tmp_path / "absent.yaml")


def test_unknown_top_level_key_is_refused(tmp_path) -> None:
    target = tmp_path / "c.yaml"
    target.write_text("experiment: {}\nunknown: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level config keys"):
        load_config(target)


def test_unknown_experiment_key_is_refused(tmp_path) -> None:
    target = tmp_path / "c.yaml"
    target.write_text("experiment:\n  no_such_field: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown experiment keys"):
        load_config(target)


def test_non_mapping_root_is_refused(tmp_path) -> None:
    target = tmp_path / "c.yaml"
    target.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="config root must be a mapping"):
        load_config(target)


def test_wrongly_typed_value_is_a_value_error(tmp_path) -> None:
    # A YAML string in a numeric field must fail as the CLI's ValueError
    # contract, not escape as a TypeError traceback.
    target = tmp_path / "c.yaml"
    target.write_text("experiment:\n  n_splits: five\n", encoding="utf-8")
    with pytest.raises(ValueError, match="n_splits must be an integer"):
        config_to_experiment(load_config(target))


def test_experiment_payload_round_trips() -> None:
    shipped = sorted(CONFIGS.glob("*.yaml"))
    assert shipped, "configs/ lost its example files"
    config = config_to_experiment(load_config(shipped[0]))
    # Two directions: the payload rebuilds an equal config (across the
    # JSON/YAML boundary, where ensemble_models is a list), and describing
    # the rebuild equals the original description.
    rebuilt = config_to_experiment({"experiment": experiment_payload(config)})
    assert rebuilt == config
    assert experiment_payload(rebuilt) == experiment_payload(config)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": 5}, "model must be a string"),
        ({"task": 5}, "task must be a string"),
        ({"feature_view": 5}, "feature_view must be a string"),
        ({"split_strategy": 5}, "split_strategy must be a string"),
        ({"device": 5}, "device must be a string"),
        ({"deep_deterministic": "no"}, "deep_deterministic must be a boolean"),
        ({"include_missing_pattern": "no"}, "include_missing_pattern must be a boolean"),
        ({"top_blocks": 0}, "top_blocks must be >= 1"),
        ({"max_features": -3}, "max_features must be >= 1"),
    ][:9],
)
def test_wrongly_typed_or_ranged_values_are_value_errors(kwargs, message) -> None:
    # YAML delivers these directly; each must fail as the CLI's ValueError
    # contract at construction, not as a distant AttributeError/TypeError.
    with pytest.raises(ValueError, match=message):
        ExperimentConfig(**kwargs)


def test_bare_string_ensemble_models_is_refused() -> None:
    # A bare string is iterable, so without the guard the duplicate check
    # would report "must not repeat a model" for a scalar typo.
    with pytest.raises(ValueError, match="sequence of model names"):
        ExperimentConfig(ensemble_models="logistic")
