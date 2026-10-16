import tempfile
import unittest
from pathlib import Path

import numpy as np

import predict
from predict import create_inference_windows


class CreateInferenceWindowsTests(unittest.TestCase):
    def test_latest_window_ends_at_last_available_row(self):
        values = np.arange(12, dtype=float).reshape(6, 2)

        windows = create_inference_windows(values, window_size=3)

        self.assertEqual(windows.shape, (4, 3, 2))
        np.testing.assert_array_equal(windows[-1], values[-3:])

    def test_accepts_exactly_one_window(self):
        values = np.arange(6, dtype=float).reshape(3, 2)

        windows = create_inference_windows(values, window_size=3)

        self.assertEqual(windows.shape, (1, 3, 2))
        np.testing.assert_array_equal(windows[0], values)

    def test_rejects_insufficient_rows(self):
        values = np.arange(4, dtype=float).reshape(2, 2)

        with self.assertRaisesRegex(ValueError, "Not enough rows"):
            create_inference_windows(values, window_size=3)


class DiscoverModelBundlesTests(unittest.TestCase):
    def test_staged_and_archived_bundles_are_not_offered_for_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bucket = Path(temp_dir) / "gru_saved_model" / "95-100%"
            for relative in (
                "SOL_gru.keras",
                "staging/SOL_gru.keras",
                "archive/SOL_gru_2026-05-31_17-09-23.keras",
            ):
                model_path = bucket / relative
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_text("model", encoding="utf-8")
                model_path.with_name(f"{model_path.stem}_scaler.pkl").write_text(
                    "scaler", encoding="utf-8"
                )

            bundles = predict.discover_model_bundles(Path(temp_dir))

            self.assertEqual([bundle.id for bundle in bundles], ["gru:95-100%:SOL"])
            self.assertEqual(bundles[0].model_path, str(bucket / "SOL_gru.keras"))


if __name__ == "__main__":
    unittest.main()
