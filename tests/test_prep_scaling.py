import unittest

import numpy as np

from utils.prep_utils import (
    create_sequences,
    scale_data,
    scale_data_train_only,
    split_data,
    train_rows_upper_bound,
)


WINDOW, STEPS = 5, 2
VAL, TEST = 0.16, 0.16


def ramp(rows, cols=3):
    """Strictly increasing series: the tail is guaranteed to be above the training range."""
    base = np.arange(rows, dtype=float).reshape(-1, 1)
    return np.hstack([base + col for col in range(cols)])


class TrainOnlyScalingTests(unittest.TestCase):
    def test_scaler_never_sees_holdout_rows(self):
        data = ramp(200)

        _, scaler, train_rows = scale_data_train_only(data, WINDOW, STEPS, VAL, TEST)

        # The maximum the scaler saw is the maximum of the training rows, not of the whole array.
        self.assertAlmostEqual(scaler.data_max_[0], data[:train_rows, 0].max())
        self.assertLess(scaler.data_max_[0], data[:, 0].max())

    def test_holdout_may_leave_the_unit_range_and_that_is_expected(self):
        data = ramp(200)

        scaled, _, _ = scale_data_train_only(data, WINDOW, STEPS, VAL, TEST)
        _, _, X_test, _, _, _ = split_data(
            *create_sequences(scaled, WINDOW, STEPS, target_col_idx=0),
            val_size=VAL,
            test_size=TEST,
        )

        # On a rising series the test part must exceed 1.0, an honest reflection of
        # the market moving above the training range.
        self.assertGreater(X_test.max(), 1.0)

    def test_old_helper_leaks_holdout_into_the_scaler(self):
        # Regression test for the old behavior: scale_data over the whole dataset
        # pins the maximum to exactly 1.0 because it saw the future.
        data = ramp(200)

        leaky, leaky_scaler = scale_data(data)
        honest, honest_scaler, _ = scale_data_train_only(data, WINDOW, STEPS, VAL, TEST)

        self.assertAlmostEqual(leaky.max(), 1.0)
        self.assertGreater(honest.max(), 1.0)
        self.assertGreater(leaky_scaler.data_max_[0], honest_scaler.data_max_[0])

    def test_boundary_covers_every_row_used_by_training_sequences(self):
        rows = 200
        data = ramp(rows)
        scaled, _, train_rows = scale_data_train_only(data, WINDOW, STEPS, VAL, TEST)
        X, y = create_sequences(scaled, WINDOW, STEPS, target_col_idx=0)
        X_train, _, _, _, _, _ = split_data(X, y, val_size=VAL, test_size=TEST)

        # The last training sequence ends exactly at the boundary:
        # none of its rows lie outside what the scaler saw...
        last_train_sequence = len(X_train) - 1
        last_row_used = last_train_sequence + WINDOW + STEPS - 1
        self.assertLessEqual(last_row_used, train_rows - 1 + 1)
        # ...and the boundary does not reach into the first validation sequence
        # by more than the window and horizon length.
        self.assertEqual(train_rows, len(X_train) + WINDOW + STEPS - 1)

    def test_rejects_series_too_short_for_a_training_part(self):
        with self.assertRaisesRegex(ValueError, "Not enough data"):
            scale_data_train_only(ramp(6), WINDOW, STEPS, VAL, TEST)

    def test_upper_bound_is_zero_when_no_sequences_fit(self):
        self.assertEqual(train_rows_upper_bound(4, WINDOW, STEPS, VAL, TEST), 0)


if __name__ == "__main__":
    unittest.main()
