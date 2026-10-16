import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from controller.run_forecast_cycle import PROJECT_DIR, run_cycle


class ForecastCycleTests(unittest.TestCase):
    def test_rejects_nexgen_repository_as_publish_repository(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {}
            for name in (
                "pipeline_python",
                "ml_python",
                "pipeline_script",
                "pipeline_config",
                "producer_script",
                "publisher_script",
            ):
                path = root / name
                path.write_text("placeholder", encoding="utf-8")
                files[name] = str(path)
            args = SimpleNamespace(
                **files,
                publish_repo=str(PROJECT_DIR),
                journal=str(PROJECT_DIR / "forecasts.jsonl"),
            )

            with self.assertRaisesRegex(ValueError, "separate from the NEXGEN"):
                run_cycle(args)

    def test_pipeline_failure_does_not_run_producer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {}
            for name in ("pipeline_python", "ml_python", "pipeline_script", "pipeline_config", "producer_script"):
                path = root / name
                path.write_text("placeholder", encoding="utf-8")
                files[name] = str(path)
            args = SimpleNamespace(
                **files,
                data=str(root / "merged.csv"),
                lock=str(root / "cycle.lock"),
                journal=str(root / "forecasts.jsonl"),
                model_id="model-id",
                journal_model="sol_gru_15m_v1",
                asset="SOL",
                quote="USDT",
                timeframe_min=15,
                exchange="binance",
                max_close_delay_minutes=10,
            )

            with patch("controller.run_forecast_cycle.subprocess.run") as run_mock:
                run_mock.return_value.returncode = 9
                result = run_cycle(args)

            self.assertEqual(result, 9)
            self.assertEqual(run_mock.call_count, 1)

    def test_successful_pipeline_runs_producer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {}
            for name in ("pipeline_python", "ml_python", "pipeline_script", "pipeline_config", "producer_script"):
                path = root / name
                path.write_text("placeholder", encoding="utf-8")
                files[name] = str(path)
            data_path = root / "merged.csv"
            args = SimpleNamespace(
                **files,
                data=str(data_path),
                lock=str(root / "cycle.lock"),
                journal=str(root / "forecasts.jsonl"),
                model_id="model-id",
                journal_model="sol_gru_15m_v1",
                asset="SOL",
                quote="USDT",
                timeframe_min=15,
                exchange="binance",
                max_close_delay_minutes=10,
            )

            def fake_run(command, **kwargs):
                if len(fake_run.calls) == 0:
                    data_path.write_text("fresh", encoding="utf-8")
                fake_run.calls.append(command)
                return SimpleNamespace(returncode=0)

            fake_run.calls = []
            with patch(
                "controller.run_forecast_cycle.subprocess.run",
                side_effect=fake_run,
            ):
                result = run_cycle(args)

            self.assertEqual(result, 0)
            self.assertEqual(len(fake_run.calls), 2)
            self.assertIn("--journal-model", fake_run.calls[1])

    def test_successful_cycle_can_publish_separate_journal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            files = {}
            for name in (
                "pipeline_python",
                "ml_python",
                "pipeline_script",
                "pipeline_config",
                "producer_script",
                "publisher_script",
            ):
                path = root / name
                path.write_text("placeholder", encoding="utf-8")
                files[name] = str(path)
            data_path = root / "merged.csv"
            journal_repo = root / "forecast-journal"
            journal_repo.mkdir()
            subprocess.run(
                ["git", "-C", str(journal_repo), "init", "-b", "journal"],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(journal_repo), "config", "user.name", "Forecast Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(journal_repo),
                    "config",
                    "user.email",
                    "forecast@test.invalid",
                ],
                check=True,
            )
            (journal_repo / "forecasts.jsonl").write_text("", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(journal_repo), "add", "forecasts.jsonl"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(journal_repo), "commit", "-m", "initialize"],
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(journal_repo),
                    "remote",
                    "add",
                    "origin",
                    str(root / "remote.git"),
                ],
                check=True,
            )
            args = SimpleNamespace(
                **files,
                data=str(data_path),
                lock=str(root / "cycle.lock"),
                journal=str(journal_repo / "forecasts.jsonl"),
                model_id="model-id",
                journal_model="sol_gru_15m_v1",
                asset="SOL",
                quote="USDT",
                timeframe_min=15,
                exchange="binance",
                max_close_delay_minutes=10,
                publish_repo=str(journal_repo),
                publish_remote="origin",
                publish_branch="journal",
            )

            real_subprocess_run = subprocess.run

            def fake_run(command, **kwargs):
                if command[0] == "git":
                    return real_subprocess_run(command, **kwargs)
                if not fake_run.calls:
                    data_path.write_text("fresh", encoding="utf-8")
                fake_run.calls.append(command)
                return SimpleNamespace(returncode=0)

            fake_run.calls = []
            with patch(
                "controller.run_forecast_cycle.subprocess.run",
                side_effect=fake_run,
            ):
                result = run_cycle(args)

            self.assertEqual(result, 0)
            self.assertEqual(len(fake_run.calls), 3)
            self.assertIn("--repo", fake_run.calls[2])
            self.assertIn("--branch", fake_run.calls[2])


if __name__ == "__main__":
    unittest.main()
