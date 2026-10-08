"""``dit`` command-line interface.

Runs from a YAML file or from explicit flags. The synthetic data path
exercises the whole pipeline without the AI4AD ``.mat`` files, which are
distributed separately by the data owners.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from dit import __version__
from dit.config import config_to_experiment, experiment_payload, load_config
from dit.data.schema import DatasetBundle
from dit.data.synthetic import make_synthetic_bundle
from dit.deployment import (
    fit_deployment_model,
    load_deployment_artifact,
    predict_unlabeled,
    save_deployment_artifact,
)
from dit.evaluation.experiment import (
    DEFAULT_ENSEMBLE_MODELS,
    ENSEMBLE_WEIGHTINGS,
    ExperimentConfig,
    ExperimentResult,
    run_configured_experiment,
)
from dit.evaluation.metrics import competition_metrics
from dit.evaluation.reporting import (
    assemble_report,
    build_covariate_ablation_comparisons,
    environment_fingerprint,
    write_json,
    write_markdown,
)
from dit.evaluation.runner import EvaluationResult
from dit.evaluation.site_balance import per_site_metrics, site_composition
from dit.models.classical import (
    build_feature_matrix,
    describe_feature_layout,
    make_search_estimator,
)


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mat", help="path to MCAD_AFQ_competition.mat or MCAD_AFQ_test.mat")
    source.add_argument("--synthetic", action="store_true", help="use the deterministic synthetic bundle")
    source.add_argument(
        "--smoke",
        action="store_true",
        help="deterministic small synthetic run for end-to-end pipeline checks",
    )
    parser.add_argument("--split", choices=("auto", "train", "test"), default="auto")
    parser.add_argument("--n-samples", type=int, default=140)
    parser.add_argument("--n-sites", type=int, default=7)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=42)


def _add_experiment_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    parser.add_argument("--strategy", choices=("stratified", "loso", "site_stratified"), default="stratified")
    parser.add_argument("--model", default="linear_svm")
    parser.add_argument("--view", default="summary", help="summary, profile, or a metric name such as MD")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--covariate", choices=("none", "feature", "residualize"), default="feature")
    parser.add_argument(
        "--control-view",
        choices=("demographics", "missingness"),
        default=None,
        help="negative-control input: demographics alone or missingness alone",
    )
    parser.add_argument("--missing-pattern", action="store_true")
    parser.add_argument("--select", action="store_true", help="enable nested block/node selection")
    parser.add_argument("--top-blocks", type=int, default=None)
    parser.add_argument("--keep-fraction", type=float, default=0.05)
    parser.add_argument(
        "--max-features",
        type=int,
        default=300,
        help="hard cap on retained anatomical columns when --select is set (<=0 disables)",
    )
    parser.add_argument("--threshold", choices=("f1", "balanced", "fixed"), default="f1")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--alignment",
        choices=("none", "coral", "mmd", "dann"),
        default="none",
        help="domain-alignment loss for the profile Transformer",
    )
    parser.add_argument("--deep-epochs", type=int, default=120)
    parser.add_argument("--deep-batch-size", type=int, default=16)
    parser.add_argument("--deep-patience", type=int, default=20)
    parser.add_argument("--deep-d-model", type=int, default=64)
    parser.add_argument("--deep-layers", type=int, default=3)
    parser.add_argument(
        "--deep-calibration",
        choices=("none", "temperature", "sigmoid"),
        default=None,
        help=(
            "calibration for the Transformer's probabilities; left unset it is "
            "applied automatically to a deep member of --ensemble-models"
        ),
    )

    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "force deterministic kernels for deep training (plan P0.6); CPU "
            "kernels used here all support it, some CUDA ops would raise"
        ),
    )
    parser.add_argument(
        "--ensemble-models",
        default=",".join(DEFAULT_ENSEMBLE_MODELS),
        help="comma-separated base lineup for --model ensemble",
    )
    parser.add_argument(
        "--ensemble-weighting",
        choices=ENSEMBLE_WEIGHTINGS,
        default="inner_score",
        help="how base-model probabilities are combined",
    )


def _load_dataset(args: argparse.Namespace) -> DatasetBundle:
    if getattr(args, "smoke", False):
        # Small deterministic bundle; the size is fixed so the smoke report
        # stays reproducible.
        return make_synthetic_bundle(n_samples=60, n_sites=3, n_classes=3, seed=args.data_seed)
    if args.synthetic:
        return make_synthetic_bundle(
            n_samples=args.n_samples,
            n_sites=args.n_sites,
            n_classes=args.n_classes,
            seed=args.data_seed,
        )
    if not args.mat:
        raise SystemExit("provide --mat, --synthetic or --smoke")
    from dit.data.mat_loader import load_ai4ad_mat

    return load_ai4ad_mat(args.mat, split=args.split)


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    # ``--smoke`` runs on a tiny bundle, so shrink the outer/inner fold counts
    # to what 60 synthetic subjects support, unless the user set them lower.
    n_splits = args.n_splits
    inner_splits = args.inner_splits
    if getattr(args, "smoke", False):
        n_splits = min(n_splits, 3)
        inner_splits = min(inner_splits, 2)
    return ExperimentConfig(
        model=args.model,
        task=args.task,
        split_strategy=args.strategy,
        feature_view=args.view,
        smooth_window=args.smooth_window,
        covariate_strategy=args.covariate,
        include_missing_pattern=args.missing_pattern,
        top_blocks=args.top_blocks,
        keep_fraction=args.keep_fraction,
        max_features=(args.max_features if args.max_features and args.max_features > 0 else None),
        threshold_criterion=args.threshold,
        n_splits=n_splits,
        inner_splits=inner_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
        enable_selection=args.select,
        alignment=args.alignment,
        deep_calibration=args.deep_calibration,
        deep_epochs=args.deep_epochs,
        deep_batch_size=args.deep_batch_size,
        deep_patience=args.deep_patience,
        deep_d_model=args.deep_d_model,
        deep_layers=args.deep_layers,
        device=args.device,
        deep_deterministic=getattr(args, "deterministic", False),
        ensemble_models=tuple(
            name.strip() for name in args.ensemble_models.split(",") if name.strip()
        ),
        ensemble_weighting=args.ensemble_weighting,
        control_view=getattr(args, "control_view", None),
    )


def _run(dataset: DatasetBundle, config: ExperimentConfig) -> ExperimentResult:
    return run_configured_experiment(dataset, config)


def _bundle(
    results: list[ExperimentResult],
    dataset: DatasetBundle,
    config: dict,
    *,
    include_covariate_comparisons: bool = False,
) -> dict:
    if not results:
        raise ValueError("report bundle requires at least one result")
    tasks = {result.config.task for result in results}
    if len(tasks) != 1:
        raise ValueError(
            "report bundle requires results from one task; binary and multiclass "
            "views do not share the same subjects"
        )
    task = next(iter(tasks))
    view = dataset.task_view(task)
    paired_comparisons = (
        build_covariate_ablation_comparisons(results)
        if include_covariate_comparisons
        else None
    )
    evaluations = [result.as_evaluation_result() for result in results]
    policies = {result.key: result.threshold for result in results}
    composition = None
    per_site = None
    if view.site is not None and view.y is not None and int(np.min(view.site)) >= 0:
        try:
            # Site composition and performance describe the same task view.
            # A binary view drops MCI, so using the full data here would make
            # the composition and the predictions refer to different cohorts.
            composition = site_composition(view.site, view.y, label_map=view.label_map)
            per_site = {
                result.key: per_site_metrics(
                    view.site,
                    view.y,
                    result.predictions,
                    label_map=view.label_map,
                )
                for result in results
            }
        except ValueError as exc:
            composition = {"note": f"site metadata is incomplete: {exc}"}
    return assemble_report(
        evaluations,
        dataset_summary=dataset.summary(),
        configuration=config,
        threshold_policies=policies,
        site_composition=composition,
        per_site_metrics=per_site,
        paired_comparisons=paired_comparisons,
        extra={
            result.key: {
                "selection": result.selection_report,
                "oof_argmax": result.oof_argmax,
            }
            for result in results
        },
    )


def _write_all(out_dir: str | Path, payload: dict, tag: str) -> list[Path]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    written = [write_json(target / f"{tag}.json", payload)]
    written.append(write_markdown(target / f"{tag}.md", payload))
    return written


def cmd_evaluate(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args)
    if args.config:
        payload_cfg = load_config(args.config)
        config = config_to_experiment(payload_cfg)
    else:
        config = _config_from_args(args)
    if not args.quiet:
        print(f"dataset: {dataset.n_samples} subjects x {dataset.n_tracts} tracts x "
              f"{dataset.n_points} nodes x {dataset.n_metrics} metrics", file=sys.stderr)
        print(f"config: {json.dumps(experiment_payload(config), sort_keys=True)}", file=sys.stderr)
    result = _run(dataset, config)
    payload = _bundle([result], dataset, experiment_payload(config))
    written = _write_all(args.out, payload, "evaluation")
    written.append(_write_rad_scores(args.out, dataset, config, result))
    competition = competition_metrics(result.aggregate)
    print(json.dumps(
        {"key": result.key, "metrics": competition, "reports": [str(p) for p in written]},
        indent=2,
    ))
    return 0


def _write_rad_scores(
    out_dir: str | Path,
    dataset: DatasetBundle,
    config: ExperimentConfig,
    result: ExperimentResult,
) -> Path:
    """Write per-subject out-of-fold disease scores for external ranking.

    Every row is an out-of-fold prediction: the probability came from a model
    that did not train on that subject. Separate from the ``interpret``
    command, which refits on all data and is not a performance estimate.
    """

    view = dataset.task_view(config.task)
    labels = view.y
    subject_ids = (
        view.subject_id
        if view.subject_id is not None
        else np.arange(view.n_samples)
    )
    probabilities = result.probabilities
    predictions = result.predictions
    class_labels = [int(c) for c in sorted(set(labels.tolist()))]
    disease = max(class_labels)
    name_of = view.label_map
    target = Path(out_dir) / "rad_scores.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        ["subject_id", "true_label", "predicted_label", "disease_score"]
        + [f"p_{name_of[c]}" for c in class_labels]
    )
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index in range(view.n_samples):
            row = [
                str(subject_ids[index]),
                name_of[int(labels[index])],
                name_of[int(predictions[index])],
                f"{float(probabilities[index, disease]):.6f}",
            ]
            row += [f"{float(probabilities[index, c]):.6f}" for c in class_labels]
            writer.writerow(row)
    return target


ABLATION_COVARIATES = ("none", "feature", "residualize")
ABLATION_VIEWS = ("summary", "profile")
ABLATION_MODELS = ("linear_svm", "logistic")


def cmd_ablation(args: argparse.Namespace) -> int:
    """Sweep covariate policy x feature view x model and tabulate the deltas."""

    dataset = _load_dataset(args)
    rows: list[dict[str, object]] = []
    payloads: list[dict] = []
    written: list[Path] = []
    # Match the evaluate smoke behaviour so ``ablation --smoke`` also runs on the
    # tiny bundle without exceeding its per-class fold budget.
    n_splits = min(args.n_splits, 3) if getattr(args, "smoke", False) else args.n_splits
    inner_splits = min(args.inner_splits, 2) if getattr(args, "smoke", False) else args.inner_splits
    base = dict(
        task=args.task,
        split_strategy=args.strategy,
        smooth_window=args.smooth_window,
        threshold_criterion=args.threshold,
        n_splits=n_splits,
        inner_splits=inner_splits,
        seed=args.seed,
        n_jobs=args.n_jobs,
        ensemble_models=tuple(
            name.strip() for name in args.ensemble_models.split(",") if name.strip()
        ),
        ensemble_weighting=args.ensemble_weighting,
    )
    models = ABLATION_MODELS if args.full else (args.model,)
    views = ABLATION_VIEWS if args.full else (args.view,)
    for covariate in ABLATION_COVARIATES:
        for view in views:
            for model in models:
                config = ExperimentConfig(
                    model=model,
                    feature_view=view,
                    covariate_strategy=covariate,
                    **base,
                )
                if not args.quiet:
                    print(f"[ablation] {model} / {view} / {covariate}", file=sys.stderr)
                result = _run(dataset, config)
                metrics = result.aggregate
                rows.append(
                    {
                        "model": model,
                        "view": view,
                        "covariate_strategy": covariate,
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "roc_auc": metrics.get("roc_auc"),
                        "ece": metrics.get("ece"),
                        "aggregate_argmax_accuracy": metrics["aggregate_argmax_accuracy"],
                        "threshold": (
                            result.threshold.thresholds[0]
                            if result.threshold and result.threshold.thresholds
                            else 0.5
                        ),
                        "threshold_criterion": (
                            result.threshold.criterion
                            if result.threshold
                            else "default"
                        ),
                    }
                )
                payloads.append(result)

    payload = _bundle(
        payloads,
        dataset,
        {"ablation": experiment_payload_of_base(base), **base},
        include_covariate_comparisons=True,
    )
    written = _write_all(args.out, payload, "ablation")
    # Strict JSON so an out-of-range float raises instead of emitting a bare
    # NaN that downstream parsers reject.
    written.append(write_json(Path(args.out) / "ablation_table.json", rows))
    written.append(_write_ablation_csv(args.out, rows))
    _write_markdown_table(args.out, rows, written)
    print(json.dumps({"rows": len(rows), "reports": [str(p) for p in written]}, indent=2))
    return 0


def _write_ablation_csv(out_dir: str | Path, rows: list[dict[str, object]]) -> Path:
    """Write the ablation sweep as a CSV for spreadsheet-based inspection."""

    columns = [
        "model",
        "view",
        "covariate_strategy",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "roc_auc",
        "ece",
        "aggregate_argmax_accuracy",
        "threshold",
        "threshold_criterion",
    ]
    target = Path(out_dir) / "ablation_table.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    return target


def experiment_payload_of_base(base: dict) -> dict[str, object]:
    return base


def _write_markdown_table(out_dir: str | Path, rows: list[dict[str, object]], written: list[Path]) -> None:
    header = "| model | view | covariate | ACC | bal.ACC | F1 | AUC | argmax ACC | thr |"
    lines = ["# Ablation table", "", header, "|" + "---|" * 9]
    for row in rows:
        lines.append(
            "| {model} | {view} | {cov} | {acc:.3f} | {bal:.3f} | {f1:.3f} | {auc} | {am:.3f} | {thr:.2f} |".format(
                model=row["model"],
                view=row["view"],
                cov=row["covariate_strategy"],
                acc=float(row["accuracy"]),
                bal=float(row["balanced_accuracy"]),
                f1=float(row["macro_f1"]),
                auc=(f"{float(row['roc_auc']):.3f}" if row.get("roc_auc") is not None else "-"),
                am=float(row["aggregate_argmax_accuracy"]),
                thr=float(row["threshold"]),
            )
        )
    target = Path(out_dir) / "ablation_table.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(target)


def cmd_interpret(args: argparse.Namespace) -> int:
    """Coefficient importance and tract x node heatmaps.

    Refits on every labeled subject, so this produces explanations rather than
    a performance estimate.
    """

    dataset = _load_dataset(args)
    config = _config_from_args(args)
    view = dataset.task_view(config.task)
    if view.y is None:
        raise SystemExit("interpretation requires labels")
    features = build_feature_matrix(
        view,
        view=config.feature_view,
        smooth_window=config.smooth_window,
        include_covariates=config.covariates_as_features,
        include_missing_pattern=config.include_missing_pattern,
    )
    layout = describe_feature_layout(
        view,
        view=config.feature_view,
        include_covariates=config.covariates_as_features,
        include_missing_pattern=config.include_missing_pattern,
    )
    layout.validate(features.shape[1])

    from sklearn.model_selection import StratifiedKFold

    counts = np.unique(view.y, return_counts=True)[1]
    folds = min(config.inner_splits, int(np.min(counts)))
    cv = StratifiedKFold(n_splits=max(folds, 2), shuffle=True, random_state=config.seed)
    search = make_search_estimator(config.model, cv=cv, seed=config.seed, n_jobs=config.n_jobs)
    search.fit(features, view.y)

    from dit.interpret import (
        block_permutation_importance,
        literature_hits,
        node_importance_from_coefficients,
        render_metric_panel,
    )

    coefficients = node_importance_from_coefficients(search, layout)
    permutation = block_permutation_importance(search, features, view.y, layout, seed=config.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": experiment_payload(config),
        "layout": layout.describe(),
        "n_features": int(features.shape[1]),
        "top_predictors": coefficients[: args.top],
        "block_permutation_importance": permutation,
        "literature_hits": literature_hits(coefficients, layout),
        "environment": environment_fingerprint(config.seed),
    }
    written = [write_json(out_dir / "interpret.json", payload)]
    if layout.kind in {"profile", "metric"}:
        heatmaps = render_metric_panel(coefficients, layout, out_dir=out_dir / "heatmaps")
        payload["heatmaps"] = [str(path) for path in heatmaps]
        written = [write_json(out_dir / "interpret.json", payload)] + heatmaps
    print(json.dumps({"top_predictors": coefficients[:5], "reports": [str(p) for p in written]}, indent=2))
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit the frozen configuration on every labeled row and save an artifact.

    This is the deployment closure the evaluation path deliberately does not
    provide: hyperparameters are re-tuned on all labeled rows for the final
    model, so any score printed here is a selection score and the unbiased
    numbers remain the cross-validation reports.
    """

    dataset = _load_dataset(args)
    if args.config:
        config = config_to_experiment(load_config(args.config))
    else:
        config = _config_from_args(args)
    model = fit_deployment_model(dataset, config)
    artifact = save_deployment_artifact(model, args.artifact)
    best_params = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in model.best_params.items()
    }
    print(json.dumps(
        {
            "artifact": str(artifact),
            "model": config.model,
            "task": config.task,
            "best_params": best_params,
            "classes": model.label_map,
            "data_digest": model.data_digest,
            "selection_metric": config.selection_metric,
        },
        indent=2,
        sort_keys=True,
    ))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Score rows with a saved artifact and write a submission-ready CSV."""

    dataset = _load_dataset(args)
    model = load_deployment_artifact(args.artifact)
    outcome = predict_unlabeled(dataset, model)
    probabilities = outcome["probabilities"]
    predictions = outcome["predictions"]
    class_names = outcome["class_names"]
    subject_ids = (
        dataset.subject_id
        if dataset.subject_id is not None
        else np.arange(dataset.n_samples)
    )
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["subject_id", "predicted_label", "predicted_class"]
            + [f"p_{name}" for name in class_names]
        )
        for index in range(dataset.n_samples):
            writer.writerow(
                [
                    str(subject_ids[index]),
                    int(predictions[index]),
                    outcome["predicted_names"][index],
                    *[float(probabilities[index, column]) for column in range(len(class_names))],
                ]
            )
    print(json.dumps(
        {
            "predictions": str(target),
            "n_rows": int(dataset.n_samples),
            "artifact": str(args.artifact),
            "data_digest": outcome["data_digest"],
            "classes": class_names,
        },
        indent=2,
    ))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download a public data file after validating the request target."""

    from dit.data.source import FetchError, UnsafeURL, safe_request

    try:
        body, target = safe_request(args.url, timeout=args.timeout)
    except (UnsafeURL, FetchError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(body)
    print(json.dumps({"host": target.host, "bytes": len(body), "path": str(out)}, indent=2))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    print(json.dumps(environment_fingerprint(args.seed), indent=2))
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    """Run binary/multiclass x stratified/LOSO in one command.

    The competition scores both tasks and both cross-validation protocols.
    Each (task, split) group is reported separately because the binary view
    drops MCI: its subject set and site composition differ from the
    multiclass view, so they are not pooled or paired across protocols.
    """

    dataset = _load_dataset(args)
    tasks = ("binary", "multiclass")
    strategies = ("stratified", "loso")
    n_splits = min(args.n_splits, 3) if getattr(args, "smoke", False) else args.n_splits
    inner_splits = min(args.inner_splits, 2) if getattr(args, "smoke", False) else args.inner_splits

    summary: list[dict[str, object]] = []
    written: list[Path] = []
    for task in tasks:
        for strategy in strategies:
            config = ExperimentConfig(
                model=args.model,
                task=task,
                split_strategy=strategy,
                feature_view=args.view,
                smooth_window=args.smooth_window,
                covariate_strategy=args.covariate,
                threshold_criterion=args.threshold,
                n_splits=n_splits,
                inner_splits=inner_splits,
                seed=args.seed,
                n_jobs=args.n_jobs,
            )
            if not args.quiet:
                print(f"[matrix] {task} / {strategy}", file=sys.stderr)
            result = _run(dataset, config)
            group_dir = Path(args.out) / f"{task}_{strategy}"
            payload = _bundle([result], dataset, experiment_payload(config))
            written += _write_all(group_dir, payload, "evaluation")
            metrics = competition_metrics(result.aggregate)
            summary.append(
                {
                    "task": task,
                    "split_strategy": strategy,
                    "key": result.key,
                    "metrics": metrics,
                    "fold_macro": result.aggregate.get("fold_macro"),
                }
            )
    written.append(write_json(Path(args.out) / "matrix_summary.json", {"runs": summary}))
    print(json.dumps({"runs": len(summary), "reports": [str(p) for p in written]}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dit",
        description="Reproducible AI4AD AFQ classification baselines.",
    )
    parser.add_argument("--version", action="version", version=f"dit-afq {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate", help="run one configuration")
    _add_dataset_args(evaluate)
    _add_experiment_args(evaluate)
    evaluate.add_argument("--config", help="YAML config overriding the flags above")
    evaluate.add_argument("--out", default="reports")
    evaluate.add_argument("--quiet", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    matrix = sub.add_parser(
        "matrix", help="run binary/multiclass x stratified/LOSO in one command"
    )
    _add_dataset_args(matrix)
    matrix.add_argument("--model", default="linear_svm")
    matrix.add_argument("--view", default="summary")
    matrix.add_argument("--covariate", choices=("none", "feature", "residualize"), default="feature")
    matrix.add_argument("--smooth-window", type=int, default=5)
    matrix.add_argument("--threshold", choices=("f1", "balanced", "fixed"), default="f1")
    matrix.add_argument("--n-splits", type=int, default=5)
    matrix.add_argument("--inner-splits", type=int, default=3)
    matrix.add_argument("--seed", type=int, default=42)
    matrix.add_argument("--n-jobs", type=int, default=1)
    matrix.add_argument("--out", default="reports/matrix")
    matrix.add_argument("--quiet", action="store_true")
    matrix.set_defaults(func=cmd_matrix)

    ablation = sub.add_parser("ablation", help="sweep covariate policy, view and model")
    _add_dataset_args(ablation)
    ablation.add_argument("--task", choices=("binary", "multiclass"), default="binary")
    ablation.add_argument("--strategy", choices=("stratified", "loso", "site_stratified"), default="stratified")
    ablation.add_argument("--model", default="linear_svm")
    ablation.add_argument("--view", default="summary")
    ablation.add_argument("--smooth-window", type=int, default=5)
    ablation.add_argument("--threshold", choices=("f1", "balanced", "fixed"), default="f1")
    ablation.add_argument("--n-splits", type=int, default=5)
    ablation.add_argument("--inner-splits", type=int, default=3)
    ablation.add_argument("--seed", type=int, default=42)
    ablation.add_argument("--n-jobs", type=int, default=1)
    ablation.add_argument(
        "--ensemble-models",
        default=",".join(DEFAULT_ENSEMBLE_MODELS),
        help="base lineup for --model ensemble",
    )
    ablation.add_argument(
        "--ensemble-weighting",
        choices=ENSEMBLE_WEIGHTINGS,
        default="inner_score",
    )
    ablation.add_argument("--full", action="store_true", help="sweep every view and model")
    ablation.add_argument("--out", default="reports")
    ablation.add_argument("--quiet", action="store_true")
    ablation.set_defaults(func=cmd_ablation)

    interpret = sub.add_parser("interpret", help="coefficient importance and heatmaps")
    _add_dataset_args(interpret)
    _add_experiment_args(interpret)
    interpret.add_argument("--top", type=int, default=25)
    interpret.add_argument("--out", default="reports/interpretation")
    interpret.add_argument("--quiet", action="store_true")
    interpret.set_defaults(func=cmd_interpret)

    fit = sub.add_parser("fit", help="fit a deployable model on all labeled rows")
    _add_dataset_args(fit)
    _add_experiment_args(fit)
    fit.add_argument("--config", help="YAML config overriding the flags above")
    fit.add_argument("--artifact", required=True, help="output path for the model artifact")
    fit.set_defaults(func=cmd_fit)

    predict = sub.add_parser(
        "predict", help="score unlabeled rows with a saved deployment artifact"
    )
    _add_dataset_args(predict)
    predict.add_argument("--artifact", required=True, help="artifact written by the fit command")
    predict.add_argument("--out", required=True, help="output CSV path for predictions")
    predict.set_defaults(func=cmd_predict)

    fetch = sub.add_parser("fetch", help="download a public file with URL validation")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--timeout", type=float, default=30.0)
    fetch.set_defaults(func=cmd_fetch)

    info = sub.add_parser("info", help="print the environment fingerprint")
    info.add_argument("--seed", type=int, default=None)
    info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        # User-input contract violations (bad split counts, uncovered
        # site/class combinations, ...) print one actionable line instead
        # of a traceback.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
