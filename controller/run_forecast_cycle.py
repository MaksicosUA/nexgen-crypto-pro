import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from controller.publish_forecast_journal import validate_repository


def require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def run_cycle(args):
    pipeline_python = require_file(args.pipeline_python, "pipeline Python")
    ml_python = require_file(args.ml_python, "ML Python")
    pipeline_script = require_file(args.pipeline_script, "pipeline script")
    pipeline_config = require_file(args.pipeline_config, "pipeline config")
    producer_script = require_file(args.producer_script, "forecast producer")
    publish_repo = getattr(args, "publish_repo", None)
    publisher_script = None
    if publish_repo:
        publisher_script = require_file(
            args.publisher_script,
            "forecast journal publisher",
        )
        repository_path = Path(publish_repo)
        journal_path = Path(args.journal)
        if not repository_path.is_absolute():
            repository_path = PROJECT_DIR / repository_path
        if not journal_path.is_absolute():
            journal_path = PROJECT_DIR / journal_path
        if repository_path.resolve() == PROJECT_DIR.resolve():
            raise ValueError("--publish-repo must be separate from the NEXGEN code repository")
        if journal_path.resolve().parent != repository_path.resolve():
            raise ValueError(
                "When --publish-repo is set, --journal must be a file in that "
                "separate repository root"
            )
        validate_repository(
            repository_path,
            journal_path.name,
            args.publish_remote,
            args.publish_branch,
        )
    data_path = Path(args.data)
    lock_path = Path(args.lock)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Forecast cycle is already running; skipping overlapping invocation")
            return 0

        previous_mtime_ns = data_path.stat().st_mtime_ns if data_path.exists() else None
        pipeline_env = os.environ.copy()
        pipeline_env["PIPELINE_CONFIG"] = str(pipeline_config)
        pipeline_env["PIPELINE_SEND_DATASET"] = "0"

        pipeline_command = [str(pipeline_python), str(pipeline_script)]
        print("Running forecast input pipeline:", " ".join(pipeline_command))
        pipeline_result = subprocess.run(
            pipeline_command,
            cwd=PROJECT_DIR,
            env=pipeline_env,
            check=False,
        )
        if pipeline_result.returncode != 0:
            print(f"Pipeline failed with exit code {pipeline_result.returncode}; no forecast appended")
            return pipeline_result.returncode or 1

        if not data_path.is_file():
            print(f"Pipeline succeeded but merged CSV is missing: {data_path}; no forecast appended")
            return 1
        if previous_mtime_ns is not None and data_path.stat().st_mtime_ns <= previous_mtime_ns:
            print(f"Merged CSV was not refreshed: {data_path}; no forecast appended")
            return 1

        producer_command = [
            str(ml_python),
            str(producer_script),
            "--data",
            str(data_path),
            "--model-id",
            args.model_id,
            "--journal",
            args.journal,
            "--journal-model",
            args.journal_model,
            "--asset",
            args.asset,
            "--quote",
            args.quote,
            "--timeframe-min",
            str(args.timeframe_min),
            "--exchange",
            args.exchange,
            "--max-close-delay-minutes",
            str(args.max_close_delay_minutes),
        ]
        print("Running forecast producer:", " ".join(producer_command))
        producer_result = subprocess.run(
            producer_command,
            cwd=PROJECT_DIR,
            check=False,
        )
        if producer_result.returncode != 0:
            print(
                f"Producer failed with exit code {producer_result.returncode}; "
                "journal was not appended by this cycle"
            )
            return producer_result.returncode

        if publisher_script:
            publisher_command = [
                str(ml_python),
                str(publisher_script),
                "--repo",
                str(publish_repo),
                "--journal-name",
                Path(args.journal).name,
                "--remote",
                args.publish_remote,
                "--branch",
                args.publish_branch,
            ]
            print("Publishing forecast journal:", " ".join(publisher_command))
            publisher_result = subprocess.run(
                publisher_command,
                cwd=PROJECT_DIR,
                check=False,
            )
            if publisher_result.returncode != 0:
                print(
                    f"Journal publisher failed with exit code {publisher_result.returncode}; "
                    "the local append is preserved for a later retry"
                )
            return publisher_result.returncode

        return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect fresh SOL 15m data and append one forecast journal record"
    )
    parser.add_argument(
        "--pipeline-python",
        default=str(PROJECT_DIR / "pipeline" / ".venv" / "bin" / "python"),
    )
    parser.add_argument(
        "--ml-python",
        default=str(PROJECT_DIR / ".venv" / "bin" / "python"),
    )
    parser.add_argument(
        "--pipeline-script",
        default=str(PROJECT_DIR / "pipeline" / "crypto_pipeline_v4.3.py"),
    )
    parser.add_argument(
        "--pipeline-config",
        default=str(PROJECT_DIR / "configs" / "forecast_sol_15m_pipeline.yaml"),
    )
    parser.add_argument(
        "--producer-script",
        default=str(PROJECT_DIR / "controller" / "forecast_producer.py"),
    )
    parser.add_argument(
        "--publisher-script",
        default=str(PROJECT_DIR / "controller" / "publish_forecast_journal.py"),
    )
    parser.add_argument(
        "--data",
        default=str(
            PROJECT_DIR
            / "state"
            / "forecast_sol_15m"
            / "full_data_base"
            / "merged.csv"
        ),
    )
    parser.add_argument(
        "--lock",
        default=str(PROJECT_DIR / "state" / "forecast_sol_15m" / "cycle.lock"),
    )
    parser.add_argument(
        "--journal",
        default=str(PROJECT_DIR / "state" / "forecasts.jsonl"),
    )
    parser.add_argument(
        "--publish-repo",
        help=(
            "Optional separate forecast-journal clone. When set, publish immediately "
            "after a successful inference. --journal must point into its root."
        ),
    )
    parser.add_argument("--publish-remote", default="origin")
    parser.add_argument("--publish-branch", default="journal")
    parser.add_argument(
        "--model-id",
        default="gru:95-100%:SOLUSDT_SOLUSDC_15m_30d",
    )
    parser.add_argument(
        "--journal-model",
        required=True,
        help="Explicit versioned contract id, e.g. sol_gru_15m_v1; change after retraining",
    )
    parser.add_argument("--asset", default="SOL")
    parser.add_argument("--quote", default="USDT")
    parser.add_argument("--timeframe-min", type=int, default=15)
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--max-close-delay-minutes", type=int, default=10)
    return parser.parse_args()


def main():
    return run_cycle(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
