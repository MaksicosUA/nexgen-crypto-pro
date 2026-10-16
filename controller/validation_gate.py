import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.targets import (
    LOG_RETURN_TARGET,
    compare_hyperparameters,
    log_returns_to_prices,
    read_target_metadata,
)
from controller.model_staging import (
    DEFAULT_JOURNAL_PATH as DEFAULT_STAGING_JOURNAL,
    STAGING_DIR_NAME,
    StagingError,
    active_bundle,
    promote_staged,
    reject_staged,
)


@dataclass(frozen=True)
class GateThresholds:
    max_mape_regression_pct: float = 0.0
    max_rmse_regression_pct: float = 0.0
    max_r2_drop: float = 0.0
    min_directional_accuracy_delta: float = 0.0


def calculate_metrics(y_true, y_pred, execution_prices):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    execution_prices = np.asarray(execution_prices, dtype=np.float64).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(execution_prices)):
        raise ValueError(
            "Metric arrays must have equal length: "
            f"y_true={len(y_true)}, y_pred={len(y_pred)}, execution={len(execution_prices)}"
        )
    if len(y_true) == 0:
        raise ValueError("Cannot calculate gate metrics on an empty holdout")

    non_zero_mask = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100)
        if np.any(non_zero_mask)
        else math.inf
    )
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))

    true_direction = np.sign(y_true - execution_prices)
    predicted_direction = np.sign(y_pred - execution_prices)
    directional_accuracy = float(np.mean(true_direction == predicted_direction) * 100)

    return {
        "mape_pct": mape,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "explained_variance": evs,
        "directional_accuracy_pct": directional_accuracy,
    }


def persistence_metrics(y_true, execution_prices):
    """Metrics of the naive "price will not change" forecast.

    Persistence predicts the same price at t+steps_ahead as at entry time,
    so its forecast is exactly execution_prices.

    Why this matters: 15-minute prices are autocorrelated almost to one, so
    persistence alone gives a MAPE of a fraction of a percent and R² around 0.9998.
    A model that has learned the "return the input" function shows the same nice
    numbers and lands in the 95-100% accuracy category while predicting nothing.
    Without this baseline the model metrics are unreadable.

    directional_accuracy is deliberately excluded: persistence has no view on
    direction (its forecast equals the entry price), so the value degenerates into
    the share of bars where the price did not change at all, and comparing against
    it is meaningless.
    """
    metrics = calculate_metrics(y_true, execution_prices, execution_prices)
    return {key: value for key, value in metrics.items() if key != "directional_accuracy_pct"}


def skill_vs_persistence(model_metrics, baseline_persistence):
    """How much better the model is than the naive forecast.

    Positive skill means it beats persistence. Negative means the model is worse
    than "doing nothing", however low its absolute MAPE may look.
    """
    skill = {}
    for name in ("mape_pct", "rmse", "mae"):
        reference = baseline_persistence[name]
        if reference in (0, math.inf) or not math.isfinite(reference):
            skill[f"{name}_skill_pct"] = None
            continue
        skill[f"{name}_skill_pct"] = float(
            (reference - model_metrics[name]) / reference * 100
        )
    skill["r2_delta"] = float(model_metrics["r2"] - baseline_persistence["r2"])
    skill["beats_persistence"] = bool(
        model_metrics["mape_pct"] < baseline_persistence["mape_pct"]
        and model_metrics["rmse"] < baseline_persistence["rmse"]
    )
    return skill


def evaluate_bundle(model_path, scaler_path, data_path, test_size):
    import joblib
    from tensorflow.keras.models import load_model

    from backtest import (
        inverse_scale_target,
        load_numeric_backtest_frame,
        validate_model_shape,
        validate_price_series,
        validate_scaler,
    )
    from predict import infer_window_and_steps
    from utils.data_utils import split_data
    from utils.prep_utils import create_sequences

    model_path = Path(model_path)
    scaler_path = Path(scaler_path)
    data_path = Path(data_path)
    for path, label in (
        (model_path, "model"),
        (scaler_path, "scaler"),
        (data_path, "data"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")

    df = load_numeric_backtest_frame(data_path)
    feature_columns = df.columns.tolist()
    model = load_model(model_path)
    window_size, steps_ahead = infer_window_and_steps(model)
    scaler = joblib.load(scaler_path)
    warnings = validate_scaler(scaler, feature_columns)
    validate_model_shape(model, window_size, len(feature_columns), steps_ahead)

    scaled_data = scaler.transform(df.values)
    target_meta_pre = read_target_metadata(model_path)
    X, y_level = create_sequences(scaled_data, window_size, steps_ahead, target_col_idx=0)
    if target_meta_pre["target"] == LOG_RETURN_TARGET:
        from utils.targets import build_log_return_targets

        y, _ = build_log_return_targets(
            df["close"].to_numpy(dtype=float),
            window_size,
            steps_ahead,
            target_meta_pre["return_scale"],
        )
    else:
        y = y_level
    if X.size == 0 or y.size == 0:
        raise ValueError(
            f"Not enough data for sequences: rows={len(df)}, window={window_size}, steps={steps_ahead}"
        )

    _, _, X_test, _, _, y_test = split_data(X, y, val_size=0.0, test_size=test_size)
    if len(X_test) < 2:
        raise ValueError(
            f"Holdout is too small: {len(X_test)} samples. Increase data or --test-size."
        )

    raw_predictions = model.predict(X_test, verbose=0)
    if not np.isfinite(raw_predictions).all():
        raise ValueError(f"Model returned NaN/inf predictions: {model_path}")

    execution_prices = inverse_scale_target(X_test[:, -1, 0], scaler, target_col_idx=0).reshape(-1)

    # A return model outputs a log change relative to the entry price rather than a
    # price, so the inverse transform is chosen from the metadata. Metrics are then
    # computed in prices for both types, which keeps them comparable.
    target_meta = read_target_metadata(model_path)
    if target_meta["target"] == LOG_RETURN_TARGET:
        predictions = log_returns_to_prices(
            execution_prices, raw_predictions, target_meta["return_scale"]
        )
        targets = log_returns_to_prices(
            execution_prices, y_test, target_meta["return_scale"]
        )
    else:
        predictions = inverse_scale_target(raw_predictions, scaler, target_col_idx=0)
        targets = inverse_scale_target(y_test, scaler, target_col_idx=0)
    predictions_first = predictions[:, 0] if predictions.ndim > 1 else predictions.reshape(-1)
    targets_first = targets[:, 0] if targets.ndim > 1 else targets.reshape(-1)
    validate_price_series(
        {
            "predictions": predictions_first,
            "targets": targets_first,
            "execution_prices": execution_prices,
        }
    )

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "data_path": str(data_path),
        "window": int(window_size),
        "steps": int(steps_ahead),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "sequence_count": int(len(X)),
        "holdout_count": int(len(X_test)),
        "holdout_target_first": targets_first,
        "execution_prices": execution_prices,
        "target_kind": target_meta["target"],
        "target_metadata": target_meta,
        "metrics": calculate_metrics(targets_first, predictions_first, execution_prices),
        "warnings": warnings,
    }


def relative_limit(value, regression_pct):
    return value * (1 + regression_pct / 100)


def decide_gate(
    baseline_metrics,
    candidate_metrics,
    thresholds,
    candidate_skill=None,
    require_beat_persistence=False,
):
    checks = []

    if require_beat_persistence:
        if candidate_skill is None:
            raise ValueError(
                "require_beat_persistence=True requires candidate_skill from skill_vs_persistence()"
            )
        checks.append(
            {
                "name": "beats_persistence",
                "passed": bool(candidate_skill["beats_persistence"]),
                "baseline": None,
                "candidate": candidate_skill.get("mape_pct_skill_pct"),
                "limit": 0.0,
                "rule": "candidate MAPE and RMSE strictly better than the naive forecast",
            }
        )

    mape_limit = relative_limit(
        baseline_metrics["mape_pct"], thresholds.max_mape_regression_pct
    )
    checks.append(
        {
            "name": "mape",
            "passed": candidate_metrics["mape_pct"] <= mape_limit,
            "baseline": baseline_metrics["mape_pct"],
            "candidate": candidate_metrics["mape_pct"],
            "limit": mape_limit,
            "rule": "candidate <= baseline with allowed regression",
        }
    )

    rmse_limit = relative_limit(
        baseline_metrics["rmse"], thresholds.max_rmse_regression_pct
    )
    checks.append(
        {
            "name": "rmse",
            "passed": candidate_metrics["rmse"] <= rmse_limit,
            "baseline": baseline_metrics["rmse"],
            "candidate": candidate_metrics["rmse"],
            "limit": rmse_limit,
            "rule": "candidate <= baseline with allowed regression",
        }
    )

    r2_limit = baseline_metrics["r2"] - thresholds.max_r2_drop
    checks.append(
        {
            "name": "r2",
            "passed": candidate_metrics["r2"] >= r2_limit,
            "baseline": baseline_metrics["r2"],
            "candidate": candidate_metrics["r2"],
            "limit": r2_limit,
            "rule": "candidate >= baseline minus allowed drop",
        }
    )

    direction_limit = (
        baseline_metrics["directional_accuracy_pct"]
        + thresholds.min_directional_accuracy_delta
    )
    checks.append(
        {
            "name": "directional_accuracy",
            "passed": candidate_metrics["directional_accuracy_pct"] >= direction_limit,
            "baseline": baseline_metrics["directional_accuracy_pct"],
            "candidate": candidate_metrics["directional_accuracy_pct"],
            "limit": direction_limit,
            "rule": "candidate >= baseline plus required delta",
        }
    )

    passed = all(check["passed"] for check in checks)
    return {
        "decision": "PASS" if passed else "REJECT",
        "passed": passed,
        "checks": checks,
    }


def compare_models(
    baseline_model,
    baseline_scaler,
    candidate_model,
    candidate_scaler,
    data_path,
    test_size=0.1,
    thresholds=None,
    require_beat_persistence=False,
):
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
    thresholds = thresholds or GateThresholds()

    baseline = evaluate_bundle(baseline_model, baseline_scaler, data_path, test_size)
    candidate = evaluate_bundle(candidate_model, candidate_scaler, data_path, test_size)

    compatibility_fields = ("window", "steps", "feature_count", "feature_columns", "holdout_count")
    mismatches = [
        field
        for field in compatibility_fields
        if baseline[field] != candidate[field]
    ]
    if mismatches:
        raise ValueError(
            "Baseline and candidate are not directly comparable; mismatched fields: "
            + ", ".join(mismatches)
        )

    if not np.allclose(
        baseline["holdout_target_first"],
        candidate["holdout_target_first"],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            "Baseline and candidate produce different inverse-scaled holdout targets. "
            "Check scaler/data compatibility before comparison."
        )

    # Persistence is computed once: both models are evaluated on the same holdout,
    # and matching target values have already been confirmed above.
    persistence = persistence_metrics(
        baseline["holdout_target_first"], baseline["execution_prices"]
    )
    baseline_skill = skill_vs_persistence(baseline["metrics"], persistence)
    candidate_skill = skill_vs_persistence(candidate["metrics"], persistence)

    # Different hyperparameters make the comparison invalid: the metric gap then
    # reflects training settings rather than the data the gate evaluates.
    hyperparameter_differences = compare_hyperparameters(
        baseline.get("target_metadata"), candidate.get("target_metadata")
    )

    gate = decide_gate(
        baseline["metrics"],
        candidate["metrics"],
        thresholds,
        candidate_skill=candidate_skill,
        require_beat_persistence=require_beat_persistence,
    )
    report = {
        "decision": gate["decision"],
        "passed": gate["passed"],
        "test_size": test_size,
        "thresholds": asdict(thresholds),
        "require_beat_persistence": bool(require_beat_persistence),
        "checks": gate["checks"],
        "persistence": persistence,
        "hyperparameter_differences": hyperparameter_differences,
        "baseline": {key: value for key, value in baseline.items() if not isinstance(value, np.ndarray)},
        "candidate": {key: value for key, value in candidate.items() if not isinstance(value, np.ndarray)},
        "mutations_applied": False,
    }
    report["baseline"]["skill_vs_persistence"] = baseline_skill
    report["candidate"]["skill_vs_persistence"] = candidate_skill
    return report


def format_persistence_summary(report):
    """Human-readable summary: did the models beat the naive forecast."""
    persistence = report["persistence"]
    lines = [
        "",
        "=== Comparison with the naive forecast (persistence) ===",
        f"  persistence:  MAPE {persistence['mape_pct']:.4f}%   "
        f"RMSE {persistence['rmse']:.4f}   R² {persistence['r2']:.6f}",
    ]
    for role in ("baseline", "candidate"):
        metrics = report[role]["metrics"]
        skill = report[role]["skill_vs_persistence"]
        mark = "✅ better" if skill["beats_persistence"] else "❌ NOT better"
        mape_skill = skill.get("mape_pct_skill_pct")
        skill_text = f"{mape_skill:+.2f}%" if mape_skill is not None else "n/a"
        lines.append(
            f"  {role:10}  MAPE {metrics['mape_pct']:.4f}%   "
            f"RMSE {metrics['rmse']:.4f}   R² {metrics['r2']:.6f}   "
            f"→ {mark} than persistence (MAPE skill {skill_text})"
        )
    differences = report.get("hyperparameter_differences") or []
    if differences:
        lines.append("")
        lines.append("  ⚠️  baseline and candidate were trained with DIFFERENT hyperparameters:")
        for item in differences:
            lines.append(
                f"      {item['name']}: baseline {item['baseline']} → candidate {item['candidate']}"
            )
        lines.append(
            "      The metric gap reflects the settings as well, not just the data."
        )

    if not report["candidate"]["skill_vs_persistence"]["beats_persistence"]:
        lines.append(
            "  ⚠️  The candidate does not beat the 'price will not change' forecast. The low MAPE here is"
        )
        lines.append(
            "      a consequence of price autocorrelation, not predictive power."
        )
    return "\n".join(lines)


def resolve_staged_target(candidate_model_path, baseline_model_path):
    """bucket and stem for applying the verdict to artifacts.

    The verdict may only be applied to a real staged bundle whose baseline is the
    current active bundle of the same stem. Otherwise a verdict computed on one set
    of files would move another.
    """
    candidate = Path(candidate_model_path).resolve()
    baseline = Path(baseline_model_path).resolve()

    if candidate.parent.name != STAGING_DIR_NAME:
        raise StagingError(
            "--apply requires the candidate to be a staged bundle at "
            f"<bucket>/{STAGING_DIR_NAME}/<stem>.keras, got: {candidate}"
        )

    bucket_dir = candidate.parent.parent
    stem = candidate.stem
    expected_baseline = active_bundle(bucket_dir, stem).model_path.resolve()
    if baseline != expected_baseline:
        raise StagingError(
            "--apply requires the baseline to be the active bundle for this stem. "
            f"Expected {expected_baseline}, got {baseline}"
        )

    return bucket_dir, stem


def apply_decision(
    report,
    journal_path=DEFAULT_STAGING_JOURNAL,
    dry_run=False,
):
    """Moves artifacts according to an already computed verdict.

    PASS   - the staged candidate becomes active, the previous active goes to archive.
    REJECT - the candidate goes to archive marked as rejected, active is unchanged.
    """
    bucket_dir, stem = resolve_staged_target(
        report["candidate"]["model_path"], report["baseline"]["model_path"]
    )
    action = promote_staged if report["passed"] else reject_staged
    entry = action(bucket_dir, stem, journal_path=journal_path, dry_run=dry_run)
    report["staging"] = entry
    report["mutations_applied"] = not dry_run
    return entry


def write_report(output_path, report):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate and baseline models on the same holdout. "
            "Read-only by default; --apply promotes or rejects the staged candidate."
        )
    )
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--baseline-scaler", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-scaler", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--max-mape-regression-pct", type=float, default=0.0)
    parser.add_argument("--max-rmse-regression-pct", type=float, default=0.0)
    parser.add_argument("--max-r2-drop", type=float, default=0.0)
    parser.add_argument("--min-directional-accuracy-delta", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default=str(PROJECT_DIR / "logs" / "validation_gate_report.json"),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Apply the verdict to artifacts: PASS promotes the staged candidate to active, "
            "REJECT moves it to archive. Requires the candidate to be in "
            f"<bucket>/{STAGING_DIR_NAME}/ and the baseline to be the active bundle of the same stem. "
            "Without this flag the gate stays fully read-only."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --apply: show the move plan without moving anything",
    )
    parser.add_argument("--staging-journal", default=str(DEFAULT_STAGING_JOURNAL))
    parser.add_argument(
        "--require-beat-persistence",
        action="store_true",
        help=(
            "Add a mandatory check: the candidate must be strictly better than "
            "the naive 'price will not change' forecast on MAPE and RMSE. By default "
            "the persistence comparison is only printed and written to the report, "
            "but does not affect the verdict."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = GateThresholds(
        max_mape_regression_pct=args.max_mape_regression_pct,
        max_rmse_regression_pct=args.max_rmse_regression_pct,
        max_r2_drop=args.max_r2_drop,
        min_directional_accuracy_delta=args.min_directional_accuracy_delta,
    )
    report = compare_models(
        baseline_model=args.baseline_model,
        baseline_scaler=args.baseline_scaler,
        candidate_model=args.candidate_model,
        candidate_scaler=args.candidate_scaler,
        data_path=args.data,
        test_size=args.test_size,
        thresholds=thresholds,
        require_beat_persistence=args.require_beat_persistence,
    )

    apply_error = None
    if args.apply:
        try:
            apply_decision(
                report,
                journal_path=args.staging_journal,
                dry_run=args.dry_run,
            )
        except StagingError as error:
            # The verdict is already computed and must reach the report even if the
            # move did not happen; otherwise the comparison result would be lost.
            apply_error = str(error)
            report["staging_error"] = apply_error

    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(format_persistence_summary(report))
    print(f"\nValidation gate report saved to {args.output}")

    if apply_error:
        print(f"\nStaging refused: {apply_error}", file=sys.stderr)
        return 3
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
