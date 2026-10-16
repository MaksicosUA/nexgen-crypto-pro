import tempfile
import unittest
from pathlib import Path

from controller.model_staging import (
    StagingError,
    active_bundle,
    staged_bundle,
)
from controller.validation_gate import (
    GateThresholds,
    apply_decision,
    decide_gate,
    format_persistence_summary,
    persistence_metrics,
    resolve_staged_target,
    skill_vs_persistence,
)


BASELINE = {
    "mape_pct": 2.0,
    "rmse": 3.0,
    "r2": 0.70,
    "directional_accuracy_pct": 60.0,
}


class ValidationGateDecisionTests(unittest.TestCase):
    def test_passes_when_candidate_is_not_worse(self):
        candidate = {
            "mape_pct": 1.8,
            "rmse": 2.9,
            "r2": 0.72,
            "directional_accuracy_pct": 61.0,
        }

        result = decide_gate(BASELINE, candidate, GateThresholds())

        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "PASS")
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_rejects_when_one_required_metric_regresses(self):
        candidate = {
            "mape_pct": 1.8,
            "rmse": 3.1,
            "r2": 0.72,
            "directional_accuracy_pct": 61.0,
        }

        result = decide_gate(BASELINE, candidate, GateThresholds())

        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "REJECT")
        failed_checks = [check["name"] for check in result["checks"] if not check["passed"]]
        self.assertEqual(failed_checks, ["rmse"])

    def test_allows_explicit_regression_tolerance(self):
        candidate = {
            "mape_pct": 2.04,
            "rmse": 3.03,
            "r2": 0.69,
            "directional_accuracy_pct": 59.0,
        }
        thresholds = GateThresholds(
            max_mape_regression_pct=2.0,
            max_rmse_regression_pct=1.0,
            max_r2_drop=0.01,
            min_directional_accuracy_delta=-1.0,
        )

        result = decide_gate(BASELINE, candidate, thresholds)

        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "PASS")


class PersistenceBaselineTests(unittest.TestCase):
    def test_persistence_predicts_the_entry_price(self):
        execution = [100.0, 101.0, 102.0, 103.0]
        targets = [100.5, 101.5, 101.0, 104.0]

        metrics = persistence_metrics(targets, execution)

        # The persistence error is exactly the price move over the horizon.
        self.assertAlmostEqual(metrics["mae"], 0.75, places=6)
        self.assertIn("mape_pct", metrics)
        # Direction is degenerate for persistence, so it must not be in the metrics.
        self.assertNotIn("directional_accuracy_pct", metrics)

    def test_skill_is_positive_when_model_beats_persistence(self):
        persistence = {"mape_pct": 0.400, "rmse": 0.800, "mae": 0.500, "r2": 0.9998}
        model = {"mape_pct": 0.300, "rmse": 0.600, "mae": 0.400, "r2": 0.9999}

        skill = skill_vs_persistence(model, persistence)

        self.assertTrue(skill["beats_persistence"])
        self.assertAlmostEqual(skill["mape_pct_skill_pct"], 25.0, places=6)
        self.assertAlmostEqual(skill["rmse_skill_pct"], 25.0, places=6)
        self.assertGreater(skill["r2_delta"], 0)

    def test_impressive_mape_can_still_lose_to_persistence(self):
        # The exact trap this metric was added for: a MAPE of 0.35% looks great
        # and lands in the 95-100% category, but the naive forecast is better.
        persistence = {"mape_pct": 0.346, "rmse": 0.723, "mae": 0.428, "r2": 0.999797}
        model = {"mape_pct": 0.350, "rmse": 0.730, "mae": 0.431, "r2": 0.999790}

        skill = skill_vs_persistence(model, persistence)

        self.assertFalse(skill["beats_persistence"])
        self.assertLess(skill["mape_pct_skill_pct"], 0)

    def test_gate_can_require_beating_persistence(self):
        candidate = {
            "mape_pct": 1.8,
            "rmse": 2.9,
            "r2": 0.72,
            "directional_accuracy_pct": 61.0,
        }
        losing_skill = {"beats_persistence": False, "mape_pct_skill_pct": -1.2}

        without = decide_gate(BASELINE, candidate, GateThresholds())
        with_check = decide_gate(
            BASELINE,
            candidate,
            GateThresholds(),
            candidate_skill=losing_skill,
            require_beat_persistence=True,
        )

        # The candidate wins against the baseline model but does not beat persistence.
        self.assertTrue(without["passed"])
        self.assertFalse(with_check["passed"])
        self.assertIn(
            "beats_persistence",
            [check["name"] for check in with_check["checks"] if not check["passed"]],
        )

    def test_requiring_persistence_check_without_skill_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "candidate_skill"):
            decide_gate(BASELINE, BASELINE, GateThresholds(), require_beat_persistence=True)

    def test_summary_warns_when_candidate_loses_to_persistence(self):
        report = {
            "persistence": {"mape_pct": 0.346, "rmse": 0.723, "r2": 0.999797},
            "baseline": {
                "metrics": {"mape_pct": 0.34, "rmse": 0.72, "r2": 0.9998},
                "skill_vs_persistence": {"beats_persistence": True, "mape_pct_skill_pct": 1.7},
            },
            "candidate": {
                "metrics": {"mape_pct": 0.35, "rmse": 0.73, "r2": 0.99979},
                "skill_vs_persistence": {"beats_persistence": False, "mape_pct_skill_pct": -1.2},
            },
        }

        summary = format_persistence_summary(report)

        self.assertIn("NOT better", summary)
        self.assertIn("autocorrelation", summary)


STEM = "SOLUSDT_15m_gru"


class ApplyDecisionTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.bucket = Path(self._temp.name) / "gru_saved_model" / "95-100%"
        self.bucket.mkdir(parents=True)
        self.journal = Path(self._temp.name) / "staging.jsonl"
        self.active = active_bundle(self.bucket, STEM)
        self.staged = staged_bundle(self.bucket, STEM)
        for bundle, body in ((self.active, "baseline"), (self.staged, "candidate")):
            bundle.model_path.parent.mkdir(parents=True, exist_ok=True)
            bundle.model_path.write_text(body, encoding="utf-8")
            bundle.scaler_path.write_text(f"{body} scaler", encoding="utf-8")

    def build_report(self, passed):
        return {
            "decision": "PASS" if passed else "REJECT",
            "passed": passed,
            "baseline": {"model_path": str(self.active.model_path)},
            "candidate": {"model_path": str(self.staged.model_path)},
            "mutations_applied": False,
        }

    def test_pass_promotes_candidate_to_active(self):
        report = self.build_report(passed=True)

        apply_decision(report, journal_path=self.journal)

        self.assertEqual(self.active.model_path.read_text(encoding="utf-8"), "candidate")
        self.assertFalse(self.staged.model_path.exists())
        self.assertTrue(report["mutations_applied"])
        self.assertEqual(report["staging"]["action"], "promote")

    def test_reject_archives_candidate_and_keeps_active(self):
        report = self.build_report(passed=False)

        apply_decision(report, journal_path=self.journal)

        self.assertEqual(self.active.model_path.read_text(encoding="utf-8"), "baseline")
        self.assertFalse(self.staged.model_path.exists())
        self.assertEqual(report["staging"]["action"], "reject")

    def test_dry_run_reports_plan_without_mutating(self):
        report = self.build_report(passed=True)

        apply_decision(report, journal_path=self.journal, dry_run=True)

        self.assertEqual(self.active.model_path.read_text(encoding="utf-8"), "baseline")
        self.assertEqual(self.staged.model_path.read_text(encoding="utf-8"), "candidate")
        self.assertFalse(report["mutations_applied"])

    def test_refuses_candidate_outside_staging(self):
        # Candidate is not in staging: the verdict must not be applied to arbitrary paths.
        loose = self.bucket / "loose_candidate.keras"
        loose.write_text("candidate", encoding="utf-8")
        report = self.build_report(passed=True)
        report["candidate"]["model_path"] = str(loose)

        with self.assertRaisesRegex(StagingError, "staged bundle"):
            apply_decision(report, journal_path=self.journal)

    def test_refuses_baseline_that_is_not_the_active_bundle(self):
        # A verdict computed against an archived copy must not move the active bundle.
        other = self.bucket / "archive" / f"{STEM}_2026-05-31_17-09-23.keras"
        other.parent.mkdir(parents=True, exist_ok=True)
        other.write_text("archived baseline", encoding="utf-8")
        report = self.build_report(passed=True)
        report["baseline"]["model_path"] = str(other)

        with self.assertRaisesRegex(StagingError, "active bundle"):
            apply_decision(report, journal_path=self.journal)

        self.assertEqual(self.active.model_path.read_text(encoding="utf-8"), "baseline")

    def test_resolve_returns_bucket_and_stem(self):
        bucket, stem = resolve_staged_target(
            self.staged.model_path, self.active.model_path
        )

        self.assertEqual(bucket, self.bucket.resolve())
        self.assertEqual(stem, STEM)


if __name__ == "__main__":
    unittest.main()
