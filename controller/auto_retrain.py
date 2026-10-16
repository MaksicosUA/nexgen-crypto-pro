import os
import sys
import datetime
import logging
import subprocess
import argparse

# === Paths ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.telegram_utils import send_telegram_message
from controller.model_staging import (
    StagingError,
    active_bundle,
    diff_active_bundles,
    fingerprint_active_bundles,
    latest_archived,
    rollback_to_archived,
    staged_bundle,
    stage_active,
)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_base")
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
LOG_PATH = os.path.join(BASE_DIR, "logs", "auto_retrain.log")
MODELS_ROOT = os.path.join(BASE_DIR, "models", "saved_models")
GATE_SCRIPT = os.path.join(BASE_DIR, "controller", "validation_gate.py")

MODEL_SCRIPTS = {
    "gru": os.path.join(BASE_DIR, "models", "train_gru.py"),
    "lstm": os.path.join(BASE_DIR, "models", "train_lstm.py"),
    "hybrid": os.path.join(BASE_DIR, "models", "train_both.py"),
}

DEFAULT_MODEL_ORDER = ["gru", "lstm", "hybrid"]

# === Logging ===
logging.basicConfig(filename=LOG_PATH, level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Auto retrain for GRU/LSTM/Hybrid models")
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_ORDER),
        help="Comma-separated list of models: gru,lstm,hybrid",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check the commands without actually running training",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help=(
            "Disable the validation gate. By default a new model does NOT become "
            "active on its own: it is moved to staging, the previous baseline is restored "
            "from archive, and only a PASS verdict makes the candidate active."
        ),
    )
    parser.add_argument(
        "--gate-test-size",
        type=float,
        default=0.1,
        help="Holdout fraction for comparing baseline and candidate (default 0.1)",
    )
    return parser.parse_args()


def parse_model_list(models_raw: str):
    requested = [m.strip().lower() for m in models_raw.split(",") if m.strip()]
    invalid = [m for m in requested if m not in MODEL_SCRIPTS]
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Available: {list(MODEL_SCRIPTS.keys())}")
    return requested

def get_latest_data_file():
    """Returns the path to the newest csv file for training."""
    if not os.path.isdir(DATA_DIR):
        return None
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if not files:
        return None
    latest_name = max(files, key=lambda name: os.path.getmtime(os.path.join(DATA_DIR, name)))
    return os.path.join(DATA_DIR, latest_name)


def detect_gpu_name():
    """Tries to detect the GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            return None
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not names:
            return None
        return names[0]
    except Exception:
        return None


def run_model_training(model_key, data_file, dry_run=False):
    """Runs training/retraining of the selected model on the new file."""
    script_path = MODEL_SCRIPTS[model_key]
    command = [
        sys.executable, script_path,
        "--config", CONFIG_PATH,
        "--file", data_file
    ]

    if dry_run:
        return 0, "DRY RUN", " ".join(command)

    env = os.environ.copy()
    env.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    result = subprocess.run(command, capture_output=True, text=True, cwd=BASE_DIR, env=env)
    return result.returncode, result.stdout, result.stderr

GATE_EXIT_MEANING = {
    0: ("PASS", "the candidate passed the gate and became the active model"),
    2: ("REJECT", "the candidate is worse than baseline and was archived; the previous model remains active"),
    3: ("APPLY_FAILED", "the verdict was computed, but moving the artifacts failed"),
}


def run_validation_gate(bucket_dir, stem, data_file, test_size):
    """Compares baseline (active) and candidate (staging) and applies the verdict."""
    baseline = active_bundle(bucket_dir, stem)
    candidate = staged_bundle(bucket_dir, stem)
    command = [
        sys.executable,
        GATE_SCRIPT,
        "--baseline-model", str(baseline.model_path),
        "--baseline-scaler", str(baseline.scaler_path),
        "--candidate-model", str(candidate.model_path),
        "--candidate-scaler", str(candidate.scaler_path),
        "--data", data_file,
        "--test-size", str(test_size),
        "--apply",
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=BASE_DIR)
    verdict, explanation = GATE_EXIT_MEANING.get(
        result.returncode, ("ERROR", f"gate exited with code {result.returncode}")
    )
    details = (result.stderr or result.stdout or "").strip()[-800:]
    return verdict, explanation, details


def gate_candidate(bucket_dir, stem, data_file, test_size):
    """Prevents an unverified model from staying active.

    The training script has already made the new candidate active and moved the previous model
    to archive. Order is restored here: the candidate goes to staging, the baseline
    is restored from archive, and only a PASS verdict makes the candidate active.
    """
    try:
        stage_active(bucket_dir, stem)
    except StagingError as error:
        return "STAGE_FAILED", f"failed to move the candidate to staging: {error}", ""

    if latest_archived(bucket_dir, stem) is None:
        # First training for this pair: nothing to compare against. The candidate stays in
        # staging, active is intentionally empty — a human makes the decision.
        return (
            "NO_BASELINE",
            "no archived baseline to compare against; the candidate is waiting in staging, "
            "no active model is assigned",
            "",
        )

    try:
        rollback_to_archived(bucket_dir, stem)
    except StagingError as error:
        return (
            "ROLLBACK_FAILED",
            f"the candidate is in staging, but restoring the baseline failed: {error}",
            "",
        )

    return run_validation_gate(bucket_dir, stem, data_file, test_size)


def main():
    args = parse_args()
    today = datetime.date.today().strftime("%Y-%m-%d")

    try:
        models_to_run = parse_model_list(args.models)
    except ValueError as e:
        msg = f"❌ [AutoRetrain] {e}"
        logging.error(msg)
        send_telegram_message(msg)
        return

    latest_file = get_latest_data_file()
    if not latest_file:
        msg = f"❌ [AutoRetrain] No new training files for {today}"
        logging.error(msg)
        send_telegram_message(msg)
        return

    gpu_name = detect_gpu_name()
    if gpu_name:
        gpu_msg = f"🖥️ [AutoRetrain] GPU detected: {gpu_name}"
        logging.info(gpu_msg)
        send_telegram_message(gpu_msg)

    gate_enabled = not args.no_gate and not args.dry_run
    mode = "DRY-RUN" if args.dry_run else "RUN"
    gate_label = "validation gate ON" if gate_enabled else "validation gate OFF"
    msg = (
        f"🚦 [AutoRetrain] [{mode}] [{gate_label}] Starting retraining on {os.path.basename(latest_file)}\n"
        f"Models: {', '.join(models_to_run)}"
    )
    logging.info(msg)
    send_telegram_message(msg)

    if args.no_gate and not args.dry_run:
        warn_msg = (
            "⚠️ [AutoRetrain] Gate disabled via --no-gate: the new model will become "
            "active without comparison against baseline."
        )
        logging.warning(warn_msg)
        send_telegram_message(warn_msg)

    results = []
    for model_key in models_to_run:
        start_msg = f"⏳ [AutoRetrain] Starting {model_key.upper()}..."
        logging.info(start_msg)
        send_telegram_message(start_msg)

        bundles_before = fingerprint_active_bundles(MODELS_ROOT) if gate_enabled else {}

        return_code, stdout_text, stderr_text = run_model_training(model_key, latest_file, dry_run=args.dry_run)
        details = (stderr_text or stdout_text or "No output from the process.").strip()
        details = details[:1200]

        if return_code == 0 and gate_enabled:
            changed = diff_active_bundles(
                bundles_before, fingerprint_active_bundles(MODELS_ROOT)
            )
            if not changed:
                skip_msg = (
                    f"🔸 [AutoRetrain] {model_key.upper()}: training finished, but no new "
                    "active artifacts were found — gate skipped"
                )
                logging.warning(skip_msg)
                send_telegram_message(skip_msg)
            for item in changed:
                verdict, explanation, gate_details = gate_candidate(
                    item["bucket"], item["stem"], latest_file, args.gate_test_size
                )
                icon = {"PASS": "✅", "REJECT": "🛡️"}.get(verdict, "⚠️")
                gate_msg = (
                    f"{icon} [AutoRetrain] Gate {item['stem']}: {verdict}\n{explanation}"
                )
                if gate_details:
                    gate_msg += f"\n{gate_details}"
                if verdict in ("PASS", "REJECT"):
                    logging.info(gate_msg)
                else:
                    logging.error(gate_msg)
                send_telegram_message(gate_msg)

        if return_code == 0:
            ok_msg = f"✅ [AutoRetrain] {model_key.upper()} completed successfully"
            logging.info(ok_msg)
            send_telegram_message(ok_msg)
            results.append((model_key, True, "OK"))
        else:
            fail_msg = f"⚠️ [AutoRetrain] {model_key.upper()} error, code {return_code}\n{details}"
            logging.error(fail_msg)
            send_telegram_message(fail_msg)
            results.append((model_key, False, f"code {return_code}"))

    success_count = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    summary_lines = [
        f"🏁 [AutoRetrain] Summary: {success_count}/{total} succeeded",
        f"File: {os.path.basename(latest_file)}",
    ]
    summary_lines.extend([f"- {name.upper()}: {status}" for name, _, status in results])
    summary = "\n".join(summary_lines)
    logging.info(summary)
    send_telegram_message(summary)

if __name__ == "__main__":
    main()
