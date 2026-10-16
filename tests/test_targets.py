import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.prep_utils import create_sequences
from utils.targets import (
    DEFAULT_RETURN_SCALE,
    LEVEL_TARGET,
    LOG_RETURN_TARGET,
    TargetError,
    build_log_return_targets,
    compare_hyperparameters,
    entry_indices,
    extract_hyperparameters,
    is_return_model,
    log_returns_to_prices,
    metadata_path,
    prices_to_log_returns,
    read_target_metadata,
    write_target_metadata,
    zero_return_prediction,
)


class AlignmentTests(unittest.TestCase):
    def test_entry_index_matches_create_sequences_layout(self):
        # The target must be aligned with the create_sequences slicing bar for bar:
        # an off-by-one bar is look-ahead leakage that no metric will catch.
        rows, window, steps = 12, 4, 2
        data = np.arange(rows * 2, dtype=float).reshape(rows, 2)

        X, _ = create_sequences(data, window, steps, target_col_idx=0)
        indices = entry_indices(rows, window, steps)

        self.assertEqual(len(X), len(indices))
        for i, entry_idx in enumerate(indices):
            # The last row of the input window is the entry bar.
            np.testing.assert_array_equal(X[i][-1], data[entry_idx])

    def test_targets_use_future_bars_only(self):
        close = np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=float)
        window, steps = 3, 2

        y, entry = build_log_return_targets(close, window, steps)
        indices = entry_indices(len(close), window, steps)

        self.assertEqual(len(y), len(indices))
        for i, entry_idx in enumerate(indices):
            self.assertAlmostEqual(entry[i], close[entry_idx])
            for step in range(steps):
                expected = 100.0 * np.log(close[entry_idx + step + 1] / close[entry_idx])
                self.assertAlmostEqual(y[i, step], expected, places=10)

    def test_sequence_count_matches_create_sequences(self):
        rows, window, steps = 40, 7, 3
        data = np.random.default_rng(0).uniform(1, 2, size=(rows, 4))

        X, _ = create_sequences(data, window, steps, target_col_idx=0)
        y, entry = build_log_return_targets(data[:, 0], window, steps)

        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(entry))


class RoundTripTests(unittest.TestCase):
    def test_prices_survive_a_round_trip(self):
        close = np.array([100.0, 101.0, 99.5, 103.25, 97.0, 105.5, 102.0], dtype=float)
        window, steps = 2, 2

        y, entry = build_log_return_targets(close, window, steps)
        restored = log_returns_to_prices(entry, y)

        indices = entry_indices(len(close), window, steps)
        for i, entry_idx in enumerate(indices):
            for step in range(steps):
                self.assertAlmostEqual(restored[i, step], close[entry_idx + step + 1], places=9)

    def test_prices_to_returns_is_the_inverse(self):
        entry = np.array([100.0, 200.0])
        prices = np.array([[101.0, 102.0], [198.0, 202.0]])

        returns = prices_to_log_returns(entry, prices)
        back = log_returns_to_prices(entry, returns)

        np.testing.assert_allclose(back, prices, rtol=1e-12)

    def test_zero_return_reconstructs_the_entry_price(self):
        # This is exactly persistence: no move at all.
        entry = np.array([100.0, 250.0, 75.5])

        prices = log_returns_to_prices(entry, zero_return_prediction(entry, 2))

        np.testing.assert_allclose(prices, np.column_stack([entry, entry]), rtol=1e-12)

    def test_scale_is_percent_by_default(self):
        close = np.array([100.0, 100.0, 101.0])

        y, _ = build_log_return_targets(close, window_size=2, steps_ahead=1)

        # A +1% move gives a value near 1.0, not 0.01: otherwise gradients vanish.
        self.assertAlmostEqual(float(y[0, 0]), 100 * np.log(1.01), places=10)
        self.assertGreater(abs(float(y[0, 0])), 0.9)


class ValidationTests(unittest.TestCase):
    def test_rejects_non_positive_prices(self):
        with self.assertRaisesRegex(TargetError, "positive prices"):
            build_log_return_targets(np.array([10.0, 0.0, 12.0]), 1, 1)

    def test_rejects_non_positive_horizon(self):
        with self.assertRaisesRegex(TargetError, "must be positive"):
            build_log_return_targets(np.array([1.0, 2.0, 3.0]), 2, 0)

    def test_returns_empty_when_series_is_too_short(self):
        y, entry = build_log_return_targets(np.array([1.0, 2.0]), window_size=5, steps_ahead=2)

        self.assertEqual(len(y), 0)
        self.assertEqual(len(entry), 0)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(TargetError, "count mismatch"):
            log_returns_to_prices(np.array([100.0, 200.0]), np.array([[0.1]]))


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.model_path = Path(self._temp.name) / "SOL_gru.keras"
        self.model_path.write_text("model", encoding="utf-8")

    def test_metadata_lives_next_to_the_model(self):
        path = write_target_metadata(
            self.model_path, LOG_RETURN_TARGET, window_size=30, steps_ahead=2
        )

        self.assertEqual(path, metadata_path(self.model_path))
        self.assertEqual(path.name, "SOL_gru_target.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["target"], LOG_RETURN_TARGET)
        self.assertEqual(payload["window_size"], 30)
        self.assertEqual(payload["return_scale"], DEFAULT_RETURN_SCALE)

    def test_model_without_metadata_is_treated_as_legacy_level(self):
        # Older models must keep working: before return targets, no other option
        # existed.
        meta = read_target_metadata(self.model_path)

        self.assertEqual(meta["target"], LEVEL_TARGET)
        self.assertTrue(meta["legacy"])
        self.assertFalse(is_return_model(self.model_path))

    def test_return_model_is_recognised(self):
        write_target_metadata(self.model_path, LOG_RETURN_TARGET, 30, 2)

        self.assertTrue(is_return_model(self.model_path))

    def test_unknown_target_is_rejected_on_write_and_read(self):
        with self.assertRaisesRegex(TargetError, "Unknown target"):
            write_target_metadata(self.model_path, "magic", 30, 2)

        metadata_path(self.model_path).write_text(
            json.dumps({"target": "magic"}), encoding="utf-8"
        )
        with self.assertRaisesRegex(TargetError, "unknown target"):
            read_target_metadata(self.model_path)

    def test_corrupted_metadata_is_reported_clearly(self):
        metadata_path(self.model_path).write_text("{not json", encoding="utf-8")

        with self.assertRaisesRegex(TargetError, "Corrupted metadata"):
            read_target_metadata(self.model_path)


class HyperparameterTests(unittest.TestCase):
    def test_extract_keeps_only_comparison_relevant_keys(self):
        config = {
            "lr": 0.001,
            "dropout_rate": 0.3,
            "batch_size": 512,
            "units": 64,
            "num_layers": 2,
            "epochs": 100,
            "patience_early_stop": 20,
        }

        extracted = extract_hyperparameters(config)

        # epochs and patience do not affect whether two trained models are comparable.
        self.assertEqual(
            set(extracted), {"lr", "dropout_rate", "batch_size", "units", "num_layers"}
        )
        self.assertEqual(extracted["lr"], 0.001)

    def test_extract_tolerates_missing_keys_and_empty_config(self):
        self.assertEqual(extract_hyperparameters({"lr": 0.001}), {"lr": 0.001})
        self.assertEqual(extract_hyperparameters({}), {})
        self.assertEqual(extract_hyperparameters(None), {})

    def test_identical_hyperparameters_produce_no_differences(self):
        meta = {"hyperparameters": {"lr": 0.001, "units": 64}}

        self.assertEqual(compare_hyperparameters(meta, meta), [])

    def test_differing_lr_is_reported(self):
        # The exact case this check was added for: the lr key was not read for a
        # month and a half, and models were trained with a value other than the config's.
        baseline = {"hyperparameters": {"lr": 0.001, "units": 64}}
        candidate = {"hyperparameters": {"lr": 0.00044, "units": 64}}

        differences = compare_hyperparameters(baseline, candidate)

        self.assertEqual(len(differences), 1)
        self.assertEqual(differences[0]["name"], "lr")
        self.assertEqual(differences[0]["baseline"], 0.001)
        self.assertEqual(differences[0]["candidate"], 0.00044)

    def test_missing_key_on_one_side_counts_as_a_difference(self):
        baseline = {"hyperparameters": {"lr": 0.001}}
        candidate = {"hyperparameters": {"lr": 0.001, "units": 64}}

        differences = compare_hyperparameters(baseline, candidate)

        self.assertEqual([d["name"] for d in differences], ["units"])
        self.assertIsNone(differences[0]["baseline"])

    def test_legacy_model_without_hyperparameters_is_not_a_conflict(self):
        # Older models did not write this block. Reporting an unknown as a difference
        # is a false alarm that would devalue the warning.
        legacy = {"target": LEVEL_TARGET}
        modern = {"hyperparameters": {"lr": 0.001}}

        self.assertEqual(compare_hyperparameters(legacy, modern), [])
        self.assertEqual(compare_hyperparameters(modern, legacy), [])
        self.assertEqual(compare_hyperparameters(None, None), [])

    def test_metadata_round_trip_preserves_hyperparameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "SOL_gru.keras"
            model_path.write_text("model", encoding="utf-8")

            write_target_metadata(
                model_path,
                LOG_RETURN_TARGET,
                window_size=30,
                steps_ahead=2,
                hyperparameters={"lr": 0.001, "units": 64},
            )
            meta = read_target_metadata(model_path)

            self.assertEqual(meta["hyperparameters"], {"lr": 0.001, "units": 64})


if __name__ == "__main__":
    unittest.main()
