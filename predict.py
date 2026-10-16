import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Single source of truth for which directories do not serve inference.
from controller.model_staging import NON_ACTIVE_DIR_NAMES
from utils.targets import (
    LOG_RETURN_TARGET,
    log_returns_to_prices,
    read_target_metadata,
)

DEFAULT_MODELS_ROOT = PROJECT_DIR / "models" / "saved_models"


@dataclass(frozen=True)
class ModelBundle:
    id: str
    model_type: str
    quality_bucket: str
    pair: str
    model_path: str
    scaler_path: str


def infer_model_type(model_path: Path) -> str:
    for part in model_path.parts:
        if part.endswith("_saved_model"):
            return part.replace("_saved_model", "")
    return "unknown"


def infer_quality_bucket(model_path: Path, model_type: str) -> str:
    parts = list(model_path.parts)
    marker = f"{model_type}_saved_model"
    if marker in parts:
        marker_idx = parts.index(marker)
        if marker_idx + 1 < len(parts):
            return parts[marker_idx + 1]
    return "unknown"


def infer_pair_name(model_path: Path, model_type: str) -> str:
    stem = model_path.stem
    suffix = f"_{model_type}"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def discover_model_bundles(models_root=DEFAULT_MODELS_ROOT):
    models_root = Path(models_root)
    bundles = []

    for model_path in sorted(models_root.rglob("*.keras")):
        # staging holds candidates that have not passed the validation gate; archive holds history.
        # Neither may be used for inference.
        if NON_ACTIVE_DIR_NAMES.intersection(model_path.parts):
            continue

        scaler_path = model_path.with_name(f"{model_path.stem}_scaler.pkl")
        if not scaler_path.exists():
            continue

        model_type = infer_model_type(model_path)
        quality_bucket = infer_quality_bucket(model_path, model_type)
        pair = infer_pair_name(model_path, model_type)
        bundle_id = f"{model_type}:{quality_bucket}:{pair}"
        bundles.append(
            ModelBundle(
                id=bundle_id,
                model_type=model_type,
                quality_bucket=quality_bucket,
                pair=pair,
                model_path=str(model_path),
                scaler_path=str(scaler_path),
            )
        )

    return bundles


def print_model_table(bundles):
    if not bundles:
        print("No models with a paired scaler found.")
        return

    print("Available models:")
    for idx, bundle in enumerate(bundles, start=1):
        print(
            f"{idx:02d}. {bundle.id} | "
            f"type={bundle.model_type} | quality={bundle.quality_bucket} | pair={bundle.pair}"
        )


def select_bundle_interactively(bundles):
    print_model_table(bundles)
    if not bundles:
        raise ValueError("No model/scaler bundles available to choose from")

    raw_value = input("\nEnter the model number: ").strip()
    try:
        selected_idx = int(raw_value) - 1
    except ValueError as exc:
        raise ValueError(f"A model number is required, got: {raw_value!r}") from exc

    if selected_idx < 0 or selected_idx >= len(bundles):
        raise ValueError(f"Number out of range: {raw_value}")
    return bundles[selected_idx]


def find_bundle(model_id, bundles):
    for bundle in bundles:
        if bundle.id == model_id:
            return bundle
    available = ", ".join(bundle.id for bundle in bundles)
    raise ValueError(f"Model with id={model_id!r} not found. Available: {available}")


def infer_window_and_steps(model):
    input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
    output_shape = model.output_shape[0] if isinstance(model.output_shape, list) else model.output_shape

    if len(input_shape) != 3:
        raise ValueError(f"Cannot determine window/features from input_shape={model.input_shape}")
    if len(output_shape) < 2:
        raise ValueError(f"Cannot determine steps from output_shape={model.output_shape}")

    window_size = input_shape[1]
    steps_ahead = output_shape[-1]
    if window_size is None or steps_ahead is None:
        raise ValueError(
            f"Model has a dynamic input/output shape: input={model.input_shape}, output={model.output_shape}. "
            "Pass --window and --steps explicitly."
        )
    return int(window_size), int(steps_ahead)


def create_inference_windows(scaled_data, window_size):
    import numpy as np

    values = np.asarray(scaled_data)
    if values.ndim != 2:
        raise ValueError(
            f"Inference data must have shape (rows, features), got {values.shape}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if len(values) < window_size:
        raise ValueError(
            f"Not enough rows for inference: rows={len(values)}, window={window_size}"
        )

    return np.stack(
        [values[end - window_size : end] for end in range(window_size, len(values) + 1)]
    )


def run_prediction(
    data_path,
    model_path,
    scaler_path,
    window=None,
    steps=None,
    all_sequences=False,
    limit=10,
):
    import joblib
    import numpy as np
    from tensorflow.keras.models import load_model

    from backtest import (
        get_git_commit,
        inverse_scale_target,
        load_numeric_backtest_frame,
        validate_model_shape,
        validate_price_series,
        validate_scaler,
    )
    model = load_model(model_path)
    inferred_window, inferred_steps = infer_window_and_steps(model)
    window_size = int(window or inferred_window)
    steps_ahead = int(steps or inferred_steps)

    df = load_numeric_backtest_frame(data_path)
    feature_columns = df.columns.tolist()

    scaler = joblib.load(scaler_path)
    scaler_warnings = validate_scaler(scaler, feature_columns)
    scaled_data = scaler.transform(df.values)

    X = create_inference_windows(scaled_data, window_size)
    if X.ndim != 3:
        raise ValueError(f"Expected X layout (samples, time, features), got X.shape={X.shape}")

    validate_model_shape(model, window_size, len(feature_columns), steps_ahead)

    sample_indices = np.arange(window_size - 1, len(df), dtype=int)
    if all_sequences:
        if limit > 0:
            X_predict = X[-limit:]
            selected_indices = sample_indices[-limit:]
        else:
            X_predict = X
            selected_indices = sample_indices
    else:
        X_predict = X[-1:]
        selected_indices = sample_indices[-1:]

    raw_preds = model.predict(X_predict, verbose=0)
    if not np.isfinite(raw_preds).all():
        raise ValueError("Model returned NaN/inf in prediction")

    execution_prices = inverse_scale_target(X_predict[:, -1, 0], scaler, target_col_idx=0).reshape(-1)

    # Price-level and return models output fundamentally different quantities, so
    # the inverse transform is chosen from metadata. A model without a sidecar is a
    # legacy price-level model: before return models existed there was no other option.
    target_meta = read_target_metadata(model_path)
    if target_meta["target"] == LOG_RETURN_TARGET:
        preds_inv = log_returns_to_prices(
            execution_prices, raw_preds, target_meta["return_scale"]
        )
    else:
        preds_inv = inverse_scale_target(raw_preds, scaler, target_col_idx=0)
    if preds_inv.ndim == 1:
        preds_inv = preds_inv.reshape(-1, 1)
    validate_price_series(
        {
            "predictions": preds_inv.reshape(-1),
            "execution_prices": execution_prices,
        }
    )

    results = []
    for sample_idx, sample_preds, execution_price in zip(
        selected_indices, preds_inv, execution_prices
    ):
        step_predictions = [float(value) for value in sample_preds]
        first_prediction = step_predictions[0]
        results.append(
            {
                "sample_index": int(sample_idx),
                "execution_price": float(execution_price),
                "predictions": step_predictions,
                "first_step_expected_return_pct": float((first_prediction / execution_price - 1) * 100),
            }
        )

    return {
        "git_commit": get_git_commit(),
        "data": str(data_path),
        "model": str(model_path),
        "scaler": str(scaler_path),
        "window": window_size,
        "steps": steps_ahead,
        "rows": int(len(df)),
        "feature_count": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "prediction_count": int(len(results)),
        "predictions": results,
        "warnings": scaler_warnings,
    }


def write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference-only launcher for saved NEXGEN models")
    parser.add_argument("--data", help="CSV file with data for inference")
    parser.add_argument("--models-root", default=str(DEFAULT_MODELS_ROOT), help="Root of saved_models")
    parser.add_argument("--list-models", action="store_true", help="Show available model/scaler bundles")
    parser.add_argument("--interactive", action="store_true", help="Pick a model by its number in the list")
    parser.add_argument("--model-id", help="Model ID from --list-models, e.g. gru:95-100%%:merged_ltc_bybit")
    parser.add_argument("--model", help="Path to the .keras model when --model-id is not used")
    parser.add_argument("--scaler", help="Path to the .pkl scaler when --model-id is not used")
    parser.add_argument("--window", type=int, default=None, help="Override window; taken from the model by default")
    parser.add_argument("--steps", type=int, default=None, help="Override steps; taken from the model by default")
    parser.add_argument("--all", action="store_true", help="Make predictions for all sequences, not just the last one")
    parser.add_argument("--limit", type=int, default=10, help="How many of the latest predictions to output with --all")
    parser.add_argument("--output", help="JSON file for the result")
    return parser.parse_args()


def main():
    args = parse_args()
    bundles = discover_model_bundles(args.models_root)

    if args.list_models:
        print_model_table(bundles)
        if not args.data:
            return

    selected_bundle = None
    if args.interactive:
        selected_bundle = select_bundle_interactively(bundles)
    elif args.model_id:
        selected_bundle = find_bundle(args.model_id, bundles)

    if selected_bundle:
        model_path = selected_bundle.model_path
        scaler_path = selected_bundle.scaler_path
    else:
        model_path = args.model
        scaler_path = args.scaler

    if not args.data:
        raise ValueError("Pass a --data CSV file for inference")
    if not model_path or not scaler_path:
        raise ValueError("Select a model via --model-id/--interactive or pass --model and --scaler")

    result = run_prediction(
        data_path=args.data,
        model_path=model_path,
        scaler_path=scaler_path,
        window=args.window,
        steps=args.steps,
        all_sequences=args.all,
        limit=args.limit,
    )
    if selected_bundle:
        result["selected_model"] = asdict(selected_bundle)

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output:
        write_json(args.output, result)
        print(f"\nPrediction JSON saved to {args.output}")


if __name__ == "__main__":
    main()
