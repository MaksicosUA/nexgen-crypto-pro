import unittest

import numpy as np

from utils.volatility import (
    DEFAULT_HAR_SCALES,
    MIN_REALIZED_VOL,
    VolatilityError,
    build_har_features,
    build_volatility_target,
    entry_indices,
    fit_har,
    independent_windows,
    log_returns,
    minimum_entry_index,
    predict_har,
    r2_score,
    realized_volatility,
    squared_return_cumsum,
    volatility_to_price_band,
)


def geometric_walk(n, seed=0, sigma=0.003):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(0, sigma, n)))


class ReturnsTests(unittest.TestCase):
    def test_returns_are_aligned_with_bars(self):
        close = np.array([100.0, 110.0, 121.0])

        r = log_returns(close)

        # Element zero is a placeholder for the first bar: dropping it would shift
        # indexing relative to the original series.
        self.assertEqual(len(r), len(close))
        self.assertEqual(r[0], 0.0)
        self.assertAlmostEqual(r[1], np.log(1.1), places=12)
        self.assertAlmostEqual(r[2], np.log(1.1), places=12)

    def test_rejects_non_positive_prices(self):
        with self.assertRaisesRegex(VolatilityError, "positive prices"):
            log_returns(np.array([100.0, 0.0]))


class RealizedVolatilityTests(unittest.TestCase):
    def test_matches_direct_computation(self):
        close = geometric_walk(50)
        r = log_returns(close)
        cs = squared_return_cumsum(r)

        self.assertAlmostEqual(
            float(realized_volatility(cs, 10, 20)),
            float(np.sqrt(np.sum(r[10:20] ** 2))),
            places=12,
        )

    def test_flat_series_is_floored_not_zero(self):
        # On a flat stretch the sum of squares is zero, and log would turn into -inf.
        cs = squared_return_cumsum(np.zeros(20))

        value = float(realized_volatility(cs, 5, 15))

        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, np.sqrt(MIN_REALIZED_VOL), places=15)
        self.assertTrue(np.isfinite(np.log(value)))

    def test_rejects_inverted_interval(self):
        cs = squared_return_cumsum(np.zeros(10))

        with self.assertRaisesRegex(VolatilityError, "before the start"):
            realized_volatility(cs, 8, 3)


class AlignmentTests(unittest.TestCase):
    def test_first_entry_leaves_room_for_the_longest_scale(self):
        horizon = 10

        first = minimum_entry_index(horizon)

        # The longest scale is 4*horizon bars back from the entry bar, inclusive.
        self.assertEqual(first, 4 * horizon - 1)
        self.assertEqual(first + 1 - max(DEFAULT_HAR_SCALES) * horizon, 0)

    def test_target_uses_only_future_bars_and_naive_only_past(self):
        close = geometric_walk(300, seed=1)
        horizon = 5
        r = log_returns(close)

        y, naive, idx = build_volatility_target(close, horizon)

        for position, t in enumerate(idx[:20]):
            expected_fwd = np.sqrt(np.sum(r[t + 1 : t + 1 + horizon] ** 2))
            expected_back = np.sqrt(np.sum(r[t + 1 - horizon : t + 1] ** 2))
            self.assertAlmostEqual(y[position], np.log(expected_fwd), places=10)
            self.assertAlmostEqual(naive[position], np.log(expected_back), places=10)

    def test_features_and_target_share_the_same_entry_bars(self):
        close = geometric_walk(400, seed=2)
        horizon = 8

        y, naive, target_idx = build_volatility_target(close, horizon)
        features, feature_idx = build_har_features(close, horizon)

        np.testing.assert_array_equal(target_idx, feature_idx)
        self.assertEqual(len(features), len(y))
        self.assertEqual(len(features), len(naive))

    def test_first_har_column_equals_the_naive_forecast(self):
        # Scale 1 is exactly "RV over the past horizon bars", i.e. the naive forecast.
        close = geometric_walk(300, seed=3)
        horizon = 6

        _, naive, _ = build_volatility_target(close, horizon)
        features, _ = build_har_features(close, horizon)

        np.testing.assert_allclose(features[:, 0], naive, rtol=1e-12)

    def test_scales_are_ordered_from_short_to_long(self):
        close = geometric_walk(500, seed=4)
        horizon = 10
        r = log_returns(close)

        features, idx = build_har_features(close, horizon)

        t = idx[0]
        for column, scale in enumerate(DEFAULT_HAR_SCALES):
            expected = np.sqrt(np.sum(r[t + 1 - scale * horizon : t + 1] ** 2))
            self.assertAlmostEqual(features[0, column], np.log(expected), places=10)

    def test_short_series_yields_no_samples(self):
        close = geometric_walk(20, seed=5)

        y, naive, idx = build_volatility_target(close, horizon=48)
        features, _ = build_har_features(close, horizon=48)

        self.assertEqual(len(idx), 0)
        self.assertEqual(len(y), 0)
        self.assertEqual(len(naive), 0)
        self.assertEqual(len(features), 0)

    def test_rejects_non_positive_horizon(self):
        with self.assertRaisesRegex(VolatilityError, "must be positive"):
            entry_indices(1000, horizon=0)


class HarModelTests(unittest.TestCase):
    def test_recovers_a_known_linear_relationship(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(500, 3))
        y = 0.5 * X[:, 0] - 0.25 * X[:, 1] + 0.1 * X[:, 2] + 2.0

        coefficients = fit_har(X, y)

        np.testing.assert_allclose(coefficients, [0.5, -0.25, 0.1, 2.0], atol=1e-10)
        np.testing.assert_allclose(predict_har(X, coefficients), y, atol=1e-10)

    def test_rejects_mismatched_or_empty_input(self):
        with self.assertRaisesRegex(VolatilityError, "Row count mismatch"):
            fit_har(np.zeros((5, 3)), np.zeros(4))
        with self.assertRaisesRegex(VolatilityError, "empty sample"):
            fit_har(np.zeros((0, 3)), np.zeros(0))

    def test_har_beats_the_naive_forecast_out_of_sample(self):
        # A series with volatility clustering: σ itself follows a slow process.
        rng = np.random.default_rng(11)
        n = 6000
        log_sigma = np.zeros(n)
        for i in range(1, n):
            log_sigma[i] = 0.995 * log_sigma[i - 1] + rng.normal(0, 0.05)
        r = rng.normal(0, 1, n) * np.exp(log_sigma) * 0.003
        close = 100 * np.exp(np.cumsum(r))
        horizon = 24

        y, naive, _ = build_volatility_target(close, horizon)
        X, _ = build_har_features(close, horizon)
        split = int(len(y) * 0.7)

        coefficients = fit_har(X[:split], y[:split])
        r2_har = r2_score(y[split:], predict_har(X[split:], coefficients))
        r2_naive = r2_score(y[split:], naive[split:])

        self.assertGreater(r2_har, r2_naive)
        self.assertGreater(r2_har, 0.0)

    def test_r2_is_zero_for_the_mean_and_negative_for_worse(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])

        self.assertAlmostEqual(r2_score(y, np.full(4, y.mean())), 0.0, places=12)
        self.assertLess(r2_score(y, np.full(4, 10.0)), 0.0)


class PriceBandTests(unittest.TestCase):
    def test_band_is_symmetric_in_log_space_and_widens_with_volatility(self):
        entry = np.array([100.0, 100.0])
        low, high = volatility_to_price_band(entry, np.log([0.01, 0.05]))

        # Symmetry in logs: the product of the bounds equals the squared entry price.
        np.testing.assert_allclose(low * high, entry**2, rtol=1e-12)
        self.assertLess(low[0], entry[0])
        self.assertGreater(high[0], entry[0])
        # Higher volatility means a wider range.
        self.assertGreater(high[1] - low[1], high[0] - low[0])

    def test_rejects_length_mismatch(self):
        with self.assertRaisesRegex(VolatilityError, "Price count mismatch"):
            volatility_to_price_band(np.array([100.0]), np.array([0.1, 0.2]))


class OverlapTests(unittest.TestCase):
    def test_independent_windows_divides_by_horizon(self):
        # 14,000 overlapping observations at horizon 96 amount to fewer than 150
        # truly independent windows.
        self.assertEqual(independent_windows(14000, 96), 145)
        self.assertEqual(independent_windows(100, 10), 10)


if __name__ == "__main__":
    unittest.main()
