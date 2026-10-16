import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from controller.forecast_producer import (
    append_forecast_record,
    build_contract_record,
    ensure_close_delay,
    prepare_closed_input,
)


def make_record(ts="2026-07-11T09:00:00Z"):
    return {
        "v": 1,
        "ts": ts,
        "model": "sol_gru_15m_v1",
        "asset": "SOL",
        "quote": "USDT",
        "timeframeMin": 15,
        "windowBars": 30,
        "executionPrice": 83.12,
        "steps": [{"h": 1, "price": 83.4}, {"h": 2, "price": 83.55}],
    }


class ForecastProducerTests(unittest.TestCase):
    def test_close_delay_must_leave_publish_lead_before_h1(self):
        with self.assertRaisesRegex(ValueError, "leave at least 5 minutes"):
            ensure_close_delay(
                bar_timestamp="2026-07-11T09:00:00Z",
                timeframe_min=15,
                max_close_delay_minutes=11,
                now="2026-07-11T09:20:00Z",
            )

    def test_builds_contract_from_prediction_result(self):
        prediction_result = {
            "window": 30,
            "predictions": [
                {
                    "execution_price": 83.12,
                    "predictions": [83.4, 83.55],
                }
            ],
        }

        record = build_contract_record(
            prediction_result,
            pd.Timestamp("2026-07-11T09:00:00Z"),
            "sol_gru_15m_v1",
            "SOL",
            "USDT",
            15,
        )

        self.assertEqual(record, make_record())

    def test_append_is_one_line_and_rejects_duplicate_or_backfill(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            journal = Path(temp_dir) / "forecasts.jsonl"
            append_forecast_record(journal, make_record())
            append_forecast_record(journal, make_record("2026-07-11T09:15:00Z"))

            lines = journal.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0]), make_record())

            with self.assertRaisesRegex(ValueError, "duplicate/backfill"):
                append_forecast_record(journal, make_record("2026-07-11T09:15:00Z"))
            with self.assertRaisesRegex(ValueError, "duplicate/backfill"):
                append_forecast_record(journal, make_record("2026-07-11T08:45:00Z"))

    def test_uses_last_closed_bar_and_filters_confirmed_symbol(self):
        now = datetime(2026, 7, 11, 9, 37, tzinfo=timezone.utc)
        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2026-07-11T09:00:00Z",
                    "2026-07-11T09:15:00Z",
                    "2026-07-11T09:30:00Z",
                    "2026-07-11T09:15:00Z",
                ],
                "symbol": ["SOL/USDT", "SOL/USDT", "SOL/USDT", "SOL/USDC"],
                "exchange": ["binance", "binance", "binance", "binance"],
                "close": [83.0, 83.1, 83.2, 83.15],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "input.csv"
            frame.to_csv(data_path, index=False)
            closed, last_ts = prepare_closed_input(
                data_path,
                asset="SOL",
                quote="USDT",
                timeframe_min=15,
                exchange="binance",
                now=now,
                max_close_delay_minutes=10,
            )

        self.assertEqual(len(closed), 2)
        self.assertEqual(last_ts, pd.Timestamp("2026-07-11T09:15:00Z"))
        self.assertEqual(closed["symbol"].unique().tolist(), ["SOL/USDT"])

    def test_rejects_stale_input(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2026-07-11T07:00:00Z"],
                "symbol": ["SOL/USDT"],
                "exchange": ["binance"],
                "close": [83.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "input.csv"
            frame.to_csv(data_path, index=False)
            with self.assertRaisesRegex(ValueError, "stale"):
                prepare_closed_input(
                    data_path,
                    asset="SOL",
                    quote="USDT",
                    timeframe_min=15,
                    exchange="binance",
                    now=datetime(2026, 7, 11, 9, 0, tzinfo=timezone.utc),
                    max_close_delay_minutes=10,
                )

    def test_rejects_late_bar_before_old_sixty_minute_age_limit(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2026-07-11T09:00:00Z"],
                "symbol": ["SOL/USDT"],
                "exchange": ["binance"],
                "close": [83.0],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "input.csv"
            frame.to_csv(data_path, index=False)
            with self.assertRaisesRegex(ValueError, "stale for ex-ante"):
                prepare_closed_input(
                    data_path,
                    asset="SOL",
                    quote="USDT",
                    timeframe_min=15,
                    exchange="binance",
                    now=datetime(2026, 7, 11, 9, 26, tzinfo=timezone.utc),
                    max_close_delay_minutes=10,
                )


if __name__ == "__main__":
    unittest.main()
