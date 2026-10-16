import argparse
import fcntl
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

DEFAULT_JOURNAL_PATH = PROJECT_DIR / "state" / "forecasts.jsonl"
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ASSET_PATTERN = re.compile(r"^[A-Z0-9]+$")
MIN_PUBLISH_LEAD_MINUTES = 5


def utc_iso(timestamp):
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        value = value.tz_localize("UTC")
    else:
        value = value.tz_convert("UTC")
    return value.isoformat().replace("+00:00", "Z")


def parse_timestamp_column(series):
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        finite = numeric.dropna().abs()
        if finite.empty:
            raise ValueError("Timestamp column contains no valid values")
        unit = "ms" if finite.median() >= 100_000_000_000 else "s"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")

    if parsed.isna().any():
        invalid_count = int(parsed.isna().sum())
        raise ValueError(f"Timestamp column contains {invalid_count} invalid value(s)")
    return parsed


def prepare_closed_input(
    data_path,
    asset,
    quote,
    timeframe_min,
    exchange=None,
    timestamp_column="timestamp",
    now=None,
    max_close_delay_minutes=10,
):
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")
    if timeframe_min <= 0:
        raise ValueError(f"timeframe_min must be positive, got {timeframe_min}")
    if max_close_delay_minutes <= 0:
        raise ValueError(
            "max_close_delay_minutes must be positive, "
            f"got {max_close_delay_minutes}"
        )

    frame = pd.read_csv(data_path)
    frame.columns = [str(column).lower() for column in frame.columns]
    timestamp_column = timestamp_column.lower()
    if timestamp_column not in frame.columns:
        raise ValueError(
            f"CSV must contain timestamp column {timestamp_column!r}; "
            f"available={list(frame.columns)}"
        )
    if "symbol" not in frame.columns:
        raise ValueError(
            "CSV must contain symbol so asset/quote can be verified; metadata guessing is forbidden"
        )

    asset = asset.upper()
    quote = quote.upper()
    expected_symbol = f"{asset}/{quote}"
    normalized_symbols = frame["symbol"].astype(str).str.upper().str.strip()
    frame = frame.loc[normalized_symbols == expected_symbol].copy()
    if frame.empty:
        raise ValueError(f"CSV contains no rows for confirmed symbol {expected_symbol}")

    if "exchange" in frame.columns:
        normalized_exchanges = frame["exchange"].astype(str).str.lower().str.strip()
        if exchange:
            exchange = exchange.lower()
            frame = frame.loc[normalized_exchanges == exchange].copy()
            if frame.empty:
                raise ValueError(
                    f"CSV contains no rows for {expected_symbol} on exchange {exchange}"
                )
        else:
            exchanges = sorted(normalized_exchanges.dropna().unique().tolist())
            if len(exchanges) != 1:
                raise ValueError(
                    "CSV contains multiple exchanges; pass --exchange explicitly to keep one time series: "
                    + ", ".join(exchanges)
                )

    frame[timestamp_column] = parse_timestamp_column(frame[timestamp_column])
    frame.sort_values(timestamp_column, inplace=True)
    if frame[timestamp_column].duplicated().any():
        duplicates = int(frame[timestamp_column].duplicated().sum())
        raise ValueError(
            f"Filtered CSV contains {duplicates} duplicate timestamp(s); one ordered series is required"
        )

    now_utc = pd.Timestamp(datetime.now(timezone.utc) if now is None else now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")

    bar_delta = pd.Timedelta(minutes=timeframe_min)
    closed_mask = frame[timestamp_column] + bar_delta <= now_utc
    closed = frame.loc[closed_mask].copy()
    if closed.empty:
        raise ValueError("CSV contains no fully closed bars at the current UTC time")

    last_bar_ts = closed[timestamp_column].iloc[-1]
    ensure_close_delay(
        bar_timestamp=last_bar_ts,
        timeframe_min=timeframe_min,
        max_close_delay_minutes=max_close_delay_minutes,
        now=now_utc,
    )

    return closed.reset_index(drop=True), last_bar_ts


def ensure_close_delay(
    bar_timestamp,
    timeframe_min,
    max_close_delay_minutes,
    now=None,
):
    if timeframe_min <= 0:
        raise ValueError(f"timeframe_min must be positive, got {timeframe_min}")
    if max_close_delay_minutes <= 0:
        raise ValueError(
            "max_close_delay_minutes must be positive, "
            f"got {max_close_delay_minutes}"
        )
    maximum_safe_delay = timeframe_min - MIN_PUBLISH_LEAD_MINUTES
    if maximum_safe_delay <= 0 or max_close_delay_minutes > maximum_safe_delay:
        raise ValueError(
            "max_close_delay_minutes must leave at least "
            f"{MIN_PUBLISH_LEAD_MINUTES} minutes before h1 maturity; "
            f"for timeframe {timeframe_min}m the maximum is {maximum_safe_delay}"
        )
    now_utc = pd.Timestamp(datetime.now(timezone.utc) if now is None else now)
    if now_utc.tzinfo is None:
        now_utc = now_utc.tz_localize("UTC")
    else:
        now_utc = now_utc.tz_convert("UTC")
    bar_timestamp = pd.Timestamp(bar_timestamp)
    if bar_timestamp.tzinfo is None:
        bar_timestamp = bar_timestamp.tz_localize("UTC")
    else:
        bar_timestamp = bar_timestamp.tz_convert("UTC")

    close_timestamp = bar_timestamp + pd.Timedelta(minutes=timeframe_min)
    close_delay = now_utc - close_timestamp
    if close_delay < pd.Timedelta(0):
        raise ValueError(
            f"Input bar is not closed yet: ts={utc_iso(bar_timestamp)}, "
            f"closes={utc_iso(close_timestamp)}"
        )
    if close_delay > pd.Timedelta(minutes=max_close_delay_minutes):
        raise ValueError(
            "Last closed bar is stale for ex-ante publication: "
            f"ts={utc_iso(bar_timestamp)}, closed={utc_iso(close_timestamp)}, "
            f"delay={close_delay}. Maximum allowed close delay is "
            f"{max_close_delay_minutes} minutes."
        )


def validate_contract_record(record):
    required_keys = {
        "v",
        "ts",
        "model",
        "asset",
        "quote",
        "timeframeMin",
        "windowBars",
        "executionPrice",
        "steps",
    }
    missing = sorted(required_keys - set(record))
    if missing:
        raise ValueError(f"Forecast record is missing required fields: {missing}")
    if record["v"] != 1:
        raise ValueError(f"Unsupported forecast contract version: {record['v']}")
    if not MODEL_NAME_PATTERN.fullmatch(record["model"]):
        raise ValueError(
            "model must be an explicit versioned identifier using letters, digits, dot, dash, underscore"
        )
    for field in ("asset", "quote"):
        if not ASSET_PATTERN.fullmatch(record[field]):
            raise ValueError(f"{field} must be uppercase letters/digits, got {record[field]!r}")
    for field in ("timeframeMin", "windowBars"):
        if not isinstance(record[field], int) or record[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not math.isfinite(record["executionPrice"]) or record["executionPrice"] <= 0:
        raise ValueError("executionPrice must be finite and positive")

    parsed_ts = pd.Timestamp(record["ts"])
    if parsed_ts.tzinfo is None:
        raise ValueError("ts must contain an explicit UTC timezone")
    if parsed_ts.tz_convert("UTC").isoformat().replace("+00:00", "Z") != record["ts"]:
        raise ValueError("ts must be canonical ISO-8601 UTC ending in Z")

    steps = record["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("steps must be a non-empty list")
    expected_horizons = list(range(1, len(steps) + 1))
    actual_horizons = [step.get("h") for step in steps]
    if actual_horizons != expected_horizons:
        raise ValueError(
            f"steps horizons must be sequential from 1, got {actual_horizons}"
        )
    for step in steps:
        price = step.get("price")
        if not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
            raise ValueError(f"step price must be finite and positive, got {price!r}")


def build_contract_record(
    prediction_result,
    bar_timestamp,
    journal_model,
    asset,
    quote,
    timeframe_min,
):
    predictions = prediction_result.get("predictions", [])
    if len(predictions) != 1:
        raise ValueError(
            f"Producer requires exactly one latest prediction, got {len(predictions)}"
        )
    prediction = predictions[0]
    step_prices = prediction.get("predictions")
    if not isinstance(step_prices, list) or not step_prices:
        raise ValueError("Prediction result contains no forecast steps")

    record = {
        "v": 1,
        "ts": utc_iso(bar_timestamp),
        "model": journal_model,
        "asset": asset.upper(),
        "quote": quote.upper(),
        "timeframeMin": int(timeframe_min),
        "windowBars": int(prediction_result["window"]),
        "executionPrice": float(prediction["execution_price"]),
        "steps": [
            {"h": horizon, "price": float(price)}
            for horizon, price in enumerate(step_prices, start=1)
        ],
    }
    validate_contract_record(record)
    return record


def record_identity(record):
    return (
        record["model"],
        record["asset"],
        record["quote"],
        record["timeframeMin"],
    )


def append_forecast_record(journal_path, record):
    validate_contract_record(record)
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    new_ts = pd.Timestamp(record["ts"])
    identity = record_identity(record)

    with journal_path.open("a+", encoding="utf-8") as journal:
        fcntl.flock(journal.fileno(), fcntl.LOCK_EX)
        try:
            journal.seek(0)
            content = journal.read()
            if content and not content.endswith("\n"):
                raise ValueError(
                    f"Journal does not end with newline and may be corrupted: {journal_path}"
                )

            latest_same_series = None
            for line_number, line in enumerate(content.splitlines(), start=1):
                if not line.strip():
                    raise ValueError(f"Blank line found in journal at line {line_number}")
                try:
                    existing = json.loads(line)
                    validate_contract_record(existing)
                except Exception as exc:
                    raise ValueError(
                        f"Existing journal line {line_number} is invalid; refusing append: {exc}"
                    ) from exc
                if record_identity(existing) == identity:
                    existing_ts = pd.Timestamp(existing["ts"])
                    if latest_same_series is None or existing_ts > latest_same_series:
                        latest_same_series = existing_ts

            if latest_same_series is not None and new_ts <= latest_same_series:
                raise ValueError(
                    "Refusing duplicate/backfill append for the same model series: "
                    f"new_ts={record['ts']}, latest_ts={utc_iso(latest_same_series)}"
                )

            journal.seek(0, os.SEEK_END)
            journal.write(serialized + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        finally:
            fcntl.flock(journal.fileno(), fcntl.LOCK_UN)


def run_forecast_producer(
    data_path,
    model_path,
    scaler_path,
    journal_path,
    journal_model,
    asset,
    quote,
    timeframe_min,
    exchange=None,
    max_close_delay_minutes=10,
    dry_run=False,
):
    from predict import run_prediction

    closed_frame, last_bar_ts = prepare_closed_input(
        data_path=data_path,
        asset=asset,
        quote=quote,
        timeframe_min=timeframe_min,
        exchange=exchange,
        max_close_delay_minutes=max_close_delay_minutes,
    )

    with tempfile.TemporaryDirectory(prefix="nexgen_forecast_") as temp_dir:
        inference_csv = Path(temp_dir) / "closed_input.csv"
        closed_frame.to_csv(inference_csv, index=False)
        prediction_result = run_prediction(
            data_path=inference_csv,
            model_path=model_path,
            scaler_path=scaler_path,
            all_sequences=False,
        )

    record = build_contract_record(
        prediction_result=prediction_result,
        bar_timestamp=last_bar_ts,
        journal_model=journal_model,
        asset=asset,
        quote=quote,
        timeframe_min=timeframe_min,
    )
    expected_close = float(closed_frame["close"].iloc[-1])
    if not math.isclose(
        record["executionPrice"], expected_close, rel_tol=1e-5, abs_tol=1e-5
    ):
        raise ValueError(
            "Inference execution price is not the last closed bar close; refusing journal append: "
            f"prediction={record['executionPrice']}, csv_close={expected_close}"
        )

    # Inference can take long enough to cross the ex-ante deadline, so freshness
    # is checked again immediately before the append.
    ensure_close_delay(
        bar_timestamp=last_bar_ts,
        timeframe_min=timeframe_min,
        max_close_delay_minutes=max_close_delay_minutes,
    )

    if not dry_run:
        append_forecast_record(journal_path, record)
    return record


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one successful inference and append one forecast contract v1 JSONL record"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-id", help="Saved model bundle id from predict.py --list-models")
    parser.add_argument("--model", help="Explicit .keras path instead of --model-id")
    parser.add_argument("--scaler", help="Explicit .pkl path instead of --model-id")
    parser.add_argument("--models-root", default=str(PROJECT_DIR / "models" / "saved_models"))
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_PATH))
    parser.add_argument("--journal-model", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--quote", required=True)
    parser.add_argument("--timeframe-min", type=int, required=True)
    parser.add_argument("--exchange")
    parser.add_argument("--max-close-delay-minutes", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_model_paths(args):
    if args.model_id:
        if args.model or args.scaler:
            raise ValueError("Use either --model-id or explicit --model/--scaler, not both")
        from predict import discover_model_bundles, find_bundle

        bundle = find_bundle(
            args.model_id,
            discover_model_bundles(args.models_root),
        )
        return bundle.model_path, bundle.scaler_path

    if not args.model or not args.scaler:
        raise ValueError("Pass --model-id or both --model and --scaler")
    return args.model, args.scaler


def main():
    args = parse_args()
    model_path, scaler_path = resolve_model_paths(args)
    record = run_forecast_producer(
        data_path=args.data,
        model_path=model_path,
        scaler_path=scaler_path,
        journal_path=args.journal,
        journal_model=args.journal_model,
        asset=args.asset,
        quote=args.quote,
        timeframe_min=args.timeframe_min,
        exchange=args.exchange,
        max_close_delay_minutes=args.max_close_delay_minutes,
        dry_run=args.dry_run,
    )
    print(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    if args.dry_run:
        print("Dry run: journal was not modified")
    else:
        print(f"Forecast appended to {args.journal}")


if __name__ == "__main__":
    main()
