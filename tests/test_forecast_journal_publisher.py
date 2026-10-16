import json
import fcntl
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import controller.publish_forecast_journal as publisher_module
from controller.publish_forecast_journal import (
    REMOTE_AUDIT_REF,
    publish_forecast_journal,
)


def run_git(repository, *arguments):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def make_record(timestamp="2026-07-11T09:00:00Z"):
    return {
        "v": 1,
        "ts": timestamp,
        "model": "sol_gru_15m_v1",
        "asset": "SOL",
        "quote": "USDT",
        "timeframeMin": 15,
        "windowBars": 30,
        "executionPrice": 83.12,
        "steps": [{"h": 1, "price": 83.4}, {"h": 2, "price": 83.55}],
    }


class ForecastJournalPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.repository = root / "forecast-journal"
        self.remote = root / "remote.git"
        self.repository.mkdir()

        run_git(self.repository, "init", "-b", "journal")
        run_git(self.repository, "config", "user.name", "Forecast Test")
        run_git(self.repository, "config", "user.email", "forecast@test.invalid")
        (self.repository / "forecasts.jsonl").write_text("", encoding="utf-8")
        run_git(self.repository, "add", "forecasts.jsonl")
        run_git(self.repository, "commit", "-m", "initialize journal")
        self.genesis_commit = run_git(self.repository, "rev-parse", "HEAD")
        subprocess.run(
            ["git", "init", "--bare", str(self.remote)],
            capture_output=True,
            text=True,
            check=True,
        )
        run_git(self.repository, "remote", "add", "origin", str(self.remote))
        run_git(self.repository, "push", "-u", "origin", "journal")

    def tearDown(self):
        self.temp_dir.cleanup()

    def append_record(self, timestamp="2026-07-11T09:00:00Z"):
        record = json.dumps(make_record(timestamp), separators=(",", ":"))
        with (self.repository / "forecasts.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(record + "\n")

    def test_rejects_publish_timeout_that_consumes_ex_ante_buffer(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 120"):
            publish_forecast_journal(
                self.repository,
                publish_timeout_seconds=121,
            )

    def test_publishes_only_appended_records_and_noops_cleanly(self):
        self.append_record()

        result = publish_forecast_journal(self.repository)

        self.assertEqual(result["status"], "published")
        self.assertEqual(result["records"], 1)
        self.assertEqual(result["pendingRecords"], 1)
        self.assertEqual(result["lateRecords"], 1)
        self.assertEqual(result["commit"], run_git(self.repository, "rev-parse", "HEAD"))
        self.assertEqual(run_git(self.repository, "rev-list", "--count", "HEAD"), "2")
        self.assertEqual(
            run_git(self.remote, "show", "journal:forecasts.jsonl"),
            (self.repository / "forecasts.jsonl").read_text(encoding="utf-8").strip(),
        )

        second = publish_forecast_journal(self.repository)
        self.assertEqual(second["status"], "up-to-date")
        self.assertEqual(run_git(self.repository, "rev-list", "--count", "HEAD"), "2")

    def test_rejects_rewrite_of_committed_prefix(self):
        self.append_record()
        publish_forecast_journal(self.repository)
        (self.repository / "forecasts.jsonl").write_text(
            json.dumps(make_record("2026-07-11T09:15:00Z"), separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "remote byte-prefix"):
            publish_forecast_journal(self.repository)

    def test_rejects_changes_outside_journal(self):
        self.append_record()
        (self.repository / "unexpected.txt").write_text("do not commit", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "outside the journal"):
            publish_forecast_journal(self.repository)

    def test_rejects_duplicate_or_backfilled_series_timestamp(self):
        self.append_record("2026-07-11T09:15:00Z")
        self.append_record("2026-07-11T09:00:00Z")

        with self.assertRaisesRegex(ValueError, "not strictly append-ordered"):
            publish_forecast_journal(self.repository)

    def test_retries_existing_local_commit_after_push_failure(self):
        self.append_record()
        reject_hook = self.remote / "hooks" / "pre-receive"
        reject_hook.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        reject_hook.chmod(0o755)
        try:
            with self.assertRaisesRegex(RuntimeError, "git push"):
                publish_forecast_journal(self.repository)
        finally:
            reject_hook.unlink()

        self.assertEqual(run_git(self.repository, "rev-list", "--count", "HEAD"), "2")
        retried = publish_forecast_journal(self.repository)

        self.assertEqual(retried["status"], "up-to-date")
        self.assertEqual(
            run_git(self.remote, "show", "journal:forecasts.jsonl"),
            (self.repository / "forecasts.jsonl").read_text(encoding="utf-8").strip(),
        )

    def test_rejects_clean_local_commit_that_rewrites_journal_history(self):
        self.append_record()
        run_git(self.repository, "add", "forecasts.jsonl")
        run_git(self.repository, "commit", "-m", "manual append")
        first_line = json.dumps(
            make_record("2026-07-11T09:15:00Z"),
            separators=(",", ":"),
        )
        (self.repository / "forecasts.jsonl").write_text(first_line + "\n", encoding="utf-8")
        run_git(self.repository, "add", "forecasts.jsonl")
        run_git(self.repository, "commit", "-m", "rewrite local history")

        with self.assertRaisesRegex(ValueError, "remote byte-prefix|strict append-only"):
            publish_forecast_journal(self.repository)

    def test_rejects_remote_history_rewrite(self):
        self.append_record()
        publish_forecast_journal(self.repository)
        run_git(self.remote, "update-ref", "refs/heads/journal", self.genesis_commit)

        with self.assertRaisesRegex(ValueError, "history was deleted, rewritten or diverged"):
            publish_forecast_journal(self.repository)

    def test_rejects_remote_fast_forward_that_rewrites_old_line(self):
        self.append_record()
        publish_forecast_journal(self.repository)
        replacement = json.dumps(
            make_record("2026-07-11T09:15:00Z"),
            separators=(",", ":"),
        )
        (self.repository / "forecasts.jsonl").write_text(
            replacement + "\n",
            encoding="utf-8",
        )
        run_git(self.repository, "add", "forecasts.jsonl")
        run_git(self.repository, "commit", "-m", "rewrite old remote row")
        run_git(self.repository, "push", "origin", "journal")

        with self.assertRaisesRegex(ValueError, "remote byte-prefix|strict append-only"):
            publish_forecast_journal(self.repository)

    def test_first_remote_trust_requires_empty_genesis(self):
        self.append_record()
        publish_forecast_journal(self.repository)
        run_git(self.repository, "update-ref", "-d", REMOTE_AUDIT_REF)

        with self.assertRaisesRegex(ValueError, "empty root commit"):
            publish_forecast_journal(self.repository)

    def test_genesis_anchor_recovers_if_process_fails_after_successful_push(self):
        self.append_record()
        real_run_git = publisher_module.run_git
        push_completed = False

        def fail_final_audit_update(repository, *arguments, **kwargs):
            nonlocal push_completed
            if (
                push_completed
                and arguments
                and arguments[0] == "update-ref"
                and arguments[1] == REMOTE_AUDIT_REF
            ):
                raise RuntimeError("simulated crash after successful push")
            result = real_run_git(repository, *arguments, **kwargs)
            if arguments and arguments[0] == "push":
                push_completed = True
            return result

        with patch.object(
            publisher_module,
            "run_git",
            side_effect=fail_final_audit_update,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                publish_forecast_journal(self.repository)

        recovered = publish_forecast_journal(self.repository)
        self.assertEqual(recovered["status"], "up-to-date")

    def test_rejects_symlink_journal(self):
        journal_path = self.repository / "forecasts.jsonl"
        external_path = Path(self.temp_dir.name) / "external.jsonl"
        external_path.write_text(
            json.dumps(make_record(), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        journal_path.unlink()
        journal_path.symlink_to(external_path)

        with self.assertRaisesRegex(ValueError, "must not be a symlink"):
            publish_forecast_journal(self.repository)

    def test_rejects_crlf_rewrite_of_committed_bytes(self):
        self.append_record()
        publish_forecast_journal(self.repository)
        journal_path = self.repository / "forecasts.jsonl"
        journal_path.write_bytes(journal_path.read_bytes().replace(b"\n", b"\r\n"))

        with self.assertRaisesRegex(ValueError, "remote byte-prefix"):
            publish_forecast_journal(self.repository)

    def test_busy_publisher_lock_is_an_error_not_a_silent_skip(self):
        git_dir = Path(run_git(self.repository, "rev-parse", "--absolute-git-dir"))
        lock_path = git_dir / "nexgen-forecast-publish.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "publisher lock timed out"):
                publish_forecast_journal(
                    self.repository,
                    publish_timeout_seconds=0.1,
                )


if __name__ == "__main__":
    unittest.main()
