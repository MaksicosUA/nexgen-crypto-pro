import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from controller.model_staging import (
    StagingError,
    active_bundle,
    archived_bundle,
    describe,
    diff_active_bundles,
    find_active_bundles,
    fingerprint_active_bundles,
    latest_archived,
    list_archived,
    promote_staged,
    reject_staged,
    rollback_to_archived,
    staged_bundle,
    stage_active,
    validate_plan,
)


STEM = "SOLUSDT_15m_gru"


def write_bundle(bundle, model_body="model", scaler_body="scaler"):
    bundle.model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.model_path.write_text(model_body, encoding="utf-8")
    bundle.scaler_path.write_text(scaler_body, encoding="utf-8")


def read_journal(journal_path):
    lines = Path(journal_path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


class StagingHarness(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.bucket = Path(self._temp.name) / "gru_saved_model" / "95-100%"
        self.bucket.mkdir(parents=True)
        self.journal = Path(self._temp.name) / "logs" / "staging.jsonl"

    def active(self):
        return active_bundle(self.bucket, STEM)

    def staged(self):
        return staged_bundle(self.bucket, STEM)


class StageActiveTests(StagingHarness):
    def test_moves_both_files_out_of_active(self):
        write_bundle(self.active(), model_body="candidate")

        entry = stage_active(self.bucket, STEM, journal_path=self.journal)

        self.assertFalse(self.active().model_path.exists())
        self.assertFalse(self.active().scaler_path.exists())
        self.assertTrue(self.staged().complete)
        self.assertEqual(self.staged().model_path.read_text(encoding="utf-8"), "candidate")
        self.assertEqual(entry["action"], "stage")
        self.assertEqual(read_journal(self.journal)[0]["action"], "stage")

    def test_refuses_when_scaler_is_missing(self):
        active = self.active()
        active.model_path.write_text("model", encoding="utf-8")

        with self.assertRaisesRegex(StagingError, "source file is missing"):
            stage_active(self.bucket, STEM, journal_path=self.journal)

        # Half a pair must not have been moved.
        self.assertTrue(active.model_path.exists())
        self.assertFalse(self.staged().model_path.exists())
        self.assertFalse(self.journal.exists())

    def test_refuses_when_staging_slot_is_occupied(self):
        write_bundle(self.active())
        write_bundle(self.staged(), model_body="older candidate")

        with self.assertRaisesRegex(StagingError, "destination already exists"):
            stage_active(self.bucket, STEM, journal_path=self.journal)

        self.assertEqual(
            self.staged().model_path.read_text(encoding="utf-8"), "older candidate"
        )


class PromoteTests(StagingHarness):
    def test_archives_previous_active_and_promotes_candidate(self):
        write_bundle(self.active(), model_body="baseline")
        write_bundle(self.staged(), model_body="candidate")

        entry = promote_staged(
            self.bucket, STEM, timestamp="2026-07-29_01-02-03", journal_path=self.journal
        )

        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "candidate")
        self.assertFalse(self.staged().model_path.exists())
        archived = archived_bundle(self.bucket, STEM, "2026-07-29_01-02-03")
        self.assertTrue(archived.complete)
        self.assertEqual(archived.model_path.read_text(encoding="utf-8"), "baseline")
        self.assertEqual(len(entry["moves"]), 2)

    def test_dry_run_validates_two_step_plan_without_moving(self):
        # Regression: the second step used to falsely fail with "destination already exists"
        # because the first step freed nothing in dry-run.
        write_bundle(self.active(), model_body="baseline")
        write_bundle(self.staged(), model_body="candidate")

        entry = promote_staged(
            self.bucket,
            STEM,
            timestamp="2026-07-29_01-02-03",
            journal_path=self.journal,
            dry_run=True,
        )

        self.assertTrue(entry["dryRun"])
        self.assertEqual(len(entry["moves"]), 2)
        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "baseline")
        self.assertEqual(self.staged().model_path.read_text(encoding="utf-8"), "candidate")
        self.assertFalse(self.journal.exists())

    def test_promotes_into_empty_active_slot(self):
        write_bundle(self.staged(), model_body="candidate")

        promote_staged(self.bucket, STEM, journal_path=self.journal)

        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "candidate")

    def test_refuses_without_staged_bundle(self):
        write_bundle(self.active())

        with self.assertRaisesRegex(StagingError, "No complete staged bundle"):
            promote_staged(self.bucket, STEM, journal_path=self.journal)

    def test_refuses_when_active_slot_is_half_populated(self):
        write_bundle(self.staged())
        self.active().model_path.write_text("orphan model", encoding="utf-8")

        with self.assertRaisesRegex(StagingError, "incomplete bundle"):
            promote_staged(self.bucket, STEM, journal_path=self.journal)

        self.assertTrue(self.staged().complete)

    def test_failed_second_move_restores_first(self):
        write_bundle(self.active(), model_body="baseline")
        write_bundle(self.staged(), model_body="candidate")
        real_move = None

        from controller import model_staging

        real_move = model_staging._move_file
        calls = {"n": 0}

        def flaky_move(source, destination):
            calls["n"] += 1
            if calls["n"] == 3:
                raise OSError("simulated disk failure")
            return real_move(source, destination)

        with patch.object(model_staging, "_move_file", flaky_move):
            with self.assertRaises(OSError):
                promote_staged(self.bucket, STEM, journal_path=self.journal)

        # Rollback must return baseline to active and the candidate to staging.
        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "baseline")
        self.assertTrue(self.active().complete)
        self.assertTrue(self.staged().complete)


class RejectTests(StagingHarness):
    def test_moves_candidate_to_archive_and_leaves_active_alone(self):
        write_bundle(self.active(), model_body="baseline")
        write_bundle(self.staged(), model_body="candidate")

        reject_staged(
            self.bucket, STEM, timestamp="2026-07-29_01-02-03", journal_path=self.journal
        )

        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "baseline")
        self.assertFalse(self.staged().model_path.exists())
        rejected = archived_bundle(self.bucket, STEM, "rejected_2026-07-29_01-02-03")
        self.assertTrue(rejected.complete)
        self.assertEqual(rejected.model_path.read_text(encoding="utf-8"), "candidate")

    def test_rejected_version_is_not_a_default_rollback_target(self):
        write_bundle(self.active(), model_body="baseline")
        write_bundle(self.staged(), model_body="candidate")
        reject_staged(
            self.bucket, STEM, timestamp="2026-07-29_01-02-03", journal_path=self.journal
        )

        self.assertEqual(list_archived(self.bucket, STEM), [])


class RollbackTests(StagingHarness):
    def test_restores_newest_archived_and_demotes_current(self):
        write_bundle(
            archived_bundle(self.bucket, STEM, "2026-05-31_17-09-23"), model_body="older good"
        )
        write_bundle(
            archived_bundle(self.bucket, STEM, "2026-06-15_10-00-00"), model_body="newest good"
        )
        write_bundle(self.active(), model_body="bad retrain")

        entry = rollback_to_archived(self.bucket, STEM, journal_path=self.journal)

        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "newest good")
        self.assertEqual(entry["restoredTimestamp"], "2026-06-15_10-00-00")
        demoted = [
            path
            for path in (self.bucket / "archive").glob("*.keras")
            if "demoted_" in path.name
        ]
        self.assertEqual(len(demoted), 1)
        self.assertEqual(demoted[0].read_text(encoding="utf-8"), "bad retrain")

    def test_restores_explicit_timestamp(self):
        write_bundle(
            archived_bundle(self.bucket, STEM, "2026-05-31_17-09-23"), model_body="older good"
        )
        write_bundle(
            archived_bundle(self.bucket, STEM, "2026-06-15_10-00-00"), model_body="newest good"
        )

        rollback_to_archived(
            self.bucket, STEM, timestamp="2026-05-31_17-09-23", journal_path=self.journal
        )

        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "older good")

    def test_refuses_when_archive_is_empty(self):
        write_bundle(self.active(), model_body="bad retrain")

        with self.assertRaisesRegex(StagingError, "No complete archived bundle"):
            rollback_to_archived(self.bucket, STEM, journal_path=self.journal)

        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "bad retrain")

    def test_ignores_archived_model_without_scaler(self):
        orphan = archived_bundle(self.bucket, STEM, "2026-06-15_10-00-00")
        orphan.model_path.parent.mkdir(parents=True, exist_ok=True)
        orphan.model_path.write_text("model only", encoding="utf-8")
        write_bundle(
            archived_bundle(self.bucket, STEM, "2026-05-31_17-09-23"), model_body="complete pair"
        )

        newest = latest_archived(self.bucket, STEM)

        self.assertEqual(newest["timestamp"], "2026-05-31_17-09-23")


class PlanValidationTests(StagingHarness):
    def test_two_steps_cannot_claim_the_same_destination(self):
        write_bundle(self.active(), model_body="a")
        write_bundle(self.staged(), model_body="b")
        target = archived_bundle(self.bucket, STEM, "2026-07-29_01-02-03")

        with self.assertRaisesRegex(StagingError, "already claimed"):
            validate_plan([(self.active(), target), (self.staged(), target)])

    def test_step_cannot_consume_source_moved_by_earlier_step(self):
        write_bundle(self.active(), model_body="a")
        first_target = archived_bundle(self.bucket, STEM, "2026-07-29_01-02-03")
        second_target = archived_bundle(self.bucket, STEM, "2026-07-29_02-02-03")

        with self.assertRaisesRegex(StagingError, "already moved away"):
            validate_plan([(self.active(), first_target), (self.active(), second_target)])


class MetadataSidecarTests(StagingHarness):
    def test_sidecar_travels_with_the_pair(self):
        write_bundle(self.active(), model_body="candidate")
        self.active().metadata_path.write_text('{"target": "log_return_pct"}', encoding="utf-8")

        entry = stage_active(self.bucket, STEM, journal_path=self.journal)

        self.assertFalse(self.active().metadata_path.exists())
        self.assertTrue(self.staged().metadata_path.is_file())
        self.assertIn("log_return_pct", self.staged().metadata_path.read_text(encoding="utf-8"))
        # Model, scaler and sidecar — three moves within one step.
        self.assertEqual(len(entry["moves"]), 1)

    def test_legacy_pair_without_sidecar_still_moves(self):
        # Price-level models never have a sidecar; requiring one would break their staging.
        write_bundle(self.active())

        stage_active(self.bucket, STEM, journal_path=self.journal)

        self.assertTrue(self.staged().complete)
        self.assertFalse(self.staged().metadata_path.exists())

    def test_sidecar_survives_promote_round_trip(self):
        write_bundle(self.active(), model_body="baseline")
        write_bundle(self.staged(), model_body="candidate")
        self.staged().metadata_path.write_text('{"target": "log_return_pct"}', encoding="utf-8")

        promote_staged(
            self.bucket, STEM, timestamp="2026-07-30_01-02-03", journal_path=self.journal
        )

        self.assertEqual(self.active().model_path.read_text(encoding="utf-8"), "candidate")
        self.assertTrue(self.active().metadata_path.is_file())
        self.assertFalse(self.staged().metadata_path.exists())


class DiscoveryTests(StagingHarness):
    def test_active_discovery_skips_archive_and_staging(self):
        write_bundle(self.active(), model_body="active")
        write_bundle(self.staged(), model_body="staged")
        write_bundle(archived_bundle(self.bucket, STEM, "2026-05-31_17-09-23"))

        found = find_active_bundles(Path(self._temp.name))

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["stem"], STEM)
        self.assertEqual(found[0]["bundle"].model_path, self.active().model_path)

    def test_fingerprint_diff_reports_created_and_replaced(self):
        root = Path(self._temp.name)
        before = fingerprint_active_bundles(root)
        write_bundle(self.active(), model_body="first")
        after_create = fingerprint_active_bundles(root)

        created = diff_active_bundles(before, after_create)
        self.assertEqual([item["change"] for item in created], ["created"])

        self.active().model_path.write_text("first model, longer body", encoding="utf-8")
        after_replace = fingerprint_active_bundles(root)

        replaced = diff_active_bundles(after_create, after_replace)
        self.assertEqual([item["change"] for item in replaced], ["replaced"])

    def test_describe_reports_all_three_slots(self):
        write_bundle(self.active())
        write_bundle(self.staged())
        write_bundle(archived_bundle(self.bucket, STEM, "2026-05-31_17-09-23"))

        state = describe(self.bucket, STEM)

        self.assertTrue(state["active"]["complete"])
        self.assertTrue(state["staged"]["complete"])
        self.assertEqual(
            [item["timestamp"] for item in state["archived"]], ["2026-05-31_17-09-23"]
        )


if __name__ == "__main__":
    unittest.main()
