"""Safely move model artifacts between active, staging and archive.

Training scripts save a new model directly as active and move the previous one to archive.
This means an unverified model starts serving inference before any validation.
This module puts that transition back under control:

    <bucket>/<stem>.keras                      active   — what predict.py sees
    <bucket>/staging/<stem>.keras              staged   — candidate, not yet verified
    <bucket>/archive/<stem>_<timestamp>.keras  archived — history, read-only

Rules the module enforces without exception:

* nothing is deleted, only moved;
* model and scaler always move as a pair;
* an existing destination is never overwritten;
* an operation is first planned and validated in full, and only then executed;
  if any step fails, moves already made are rolled back;
* every mutation is written to an append-only journal.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.targets import metadata_path as target_metadata_path

DEFAULT_MODELS_ROOT = PROJECT_DIR / "models" / "saved_models"
DEFAULT_JOURNAL_PATH = PROJECT_DIR / "logs" / "model_staging_journal.jsonl"

ARTIFACT_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"
STAGING_DIR_NAME = "staging"
ARCHIVE_DIR_NAME = "archive"
MODEL_SUFFIX = ".keras"
SCALER_SUFFIX = ".pkl"

REJECTED_PREFIX = "rejected"
DEMOTED_PREFIX = "demoted"

# Directories whose contents must not serve inference.
NON_ACTIVE_DIR_NAMES = frozenset({ARCHIVE_DIR_NAME, STAGING_DIR_NAME})


class StagingError(RuntimeError):
    """An artifact operation was refused or did not complete."""


@dataclass(frozen=True)
class Bundle:
    """A model + scaler pair that always moves as a whole."""

    model_path: Path
    scaler_path: Path

    @property
    def paths(self):
        return (self.model_path, self.scaler_path)

    @property
    def metadata_path(self):
        """Sidecar with the training target. May be absent for legacy models."""
        return target_metadata_path(self.model_path)

    @property
    def complete(self):
        return all(path.is_file() for path in self.paths)

    @property
    def partial(self):
        return any(path.exists() for path in self.paths) and not self.complete

    @property
    def missing(self):
        return [str(path) for path in self.paths if not path.is_file()]

    def as_dict(self):
        return {
            "model_path": str(self.model_path),
            "scaler_path": str(self.scaler_path),
        }


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def artifact_timestamp(now=None):
    return (now or datetime.now()).strftime(ARTIFACT_TIMESTAMP_FORMAT)


def parse_artifact_timestamp(raw):
    """datetime, or None if the string is not a plain artifact timestamp.

    Deliberately returns None for tagged versions (`rejected_...`, `demoted_...`):
    they stay on disk but must not be offered as the default rollback target.
    Such a version can only be restored with an explicit `--timestamp`.
    """
    try:
        return datetime.strptime(raw, ARTIFACT_TIMESTAMP_FORMAT)
    except (TypeError, ValueError):
        return None


def scaler_name(stem):
    return f"{stem}_scaler{SCALER_SUFFIX}"


def archived_model_name(stem, timestamp):
    return f"{stem}_{timestamp}{MODEL_SUFFIX}"


def archived_scaler_name(stem, timestamp):
    return f"{stem}_scaler_{timestamp}{SCALER_SUFFIX}"


def active_bundle(bucket_dir, stem):
    bucket_dir = Path(bucket_dir)
    return Bundle(
        model_path=bucket_dir / f"{stem}{MODEL_SUFFIX}",
        scaler_path=bucket_dir / scaler_name(stem),
    )


def staged_bundle(bucket_dir, stem):
    staging_dir = Path(bucket_dir) / STAGING_DIR_NAME
    return Bundle(
        model_path=staging_dir / f"{stem}{MODEL_SUFFIX}",
        scaler_path=staging_dir / scaler_name(stem),
    )


def archived_bundle(bucket_dir, stem, timestamp):
    archive_dir = Path(bucket_dir) / ARCHIVE_DIR_NAME
    return Bundle(
        model_path=archive_dir / archived_model_name(stem, timestamp),
        scaler_path=archive_dir / archived_scaler_name(stem, timestamp),
    )


def list_archived(bucket_dir, stem):
    """Complete archived pairs for stem, oldest to newest.

    A model without its paired scaler is skipped: an incomplete pair cannot be restored to active.
    """
    archive_dir = Path(bucket_dir) / ARCHIVE_DIR_NAME
    if not archive_dir.is_dir():
        return []

    prefix = f"{stem}_"
    found = []
    for model_path in archive_dir.glob(f"{prefix}*{MODEL_SUFFIX}"):
        raw_timestamp = model_path.name[len(prefix) : -len(MODEL_SUFFIX)]
        parsed = parse_artifact_timestamp(raw_timestamp)
        if parsed is None:
            continue
        bundle = archived_bundle(bucket_dir, stem, raw_timestamp)
        if not bundle.complete:
            continue
        found.append((parsed, raw_timestamp, bundle))

    found.sort(key=lambda item: item[0])
    return [
        {"timestamp": raw_timestamp, "parsed": parsed, "bundle": bundle}
        for parsed, raw_timestamp, bundle in found
    ]


def latest_archived(bucket_dir, stem):
    archived = list_archived(bucket_dir, stem)
    return archived[-1] if archived else None


def append_journal_entry(journal_path, entry):
    journal_path = Path(journal_path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with open(journal_path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def transfer_pairs(source, destination):
    """Files moved in this step, as source → destination pairs.

    Model and scaler are mandatory. The training-target sidecar is added only if
    the source has one: legacy models never do, and requiring it would
    break their staging.
    """
    pairs = [
        (source.model_path, destination.model_path),
        (source.scaler_path, destination.scaler_path),
    ]
    if source.metadata_path.is_file():
        pairs.append((source.metadata_path, destination.metadata_path))
    return pairs


def validate_plan(steps):
    """Validates the whole plan before the first move.

    Tracks a virtual file system state, so a step may claim a path freed
    by a previous step. Without this, `promote` in dry-run mode would falsely
    fail with "destination already exists".
    """
    freed = set()
    claimed = set()

    for index, (source, destination) in enumerate(steps):
        pairs = transfer_pairs(source, destination)
        sources = [pair[0] for pair in pairs]
        destinations = [pair[1] for pair in pairs]

        for path in sources:
            if path in freed:
                raise StagingError(
                    f"Step {index}: source was already moved away by an earlier step: {path}"
                )
            if not path.is_file():
                raise StagingError(f"Step {index}: source file is missing: {path}")

        for path in destinations:
            if path in claimed:
                raise StagingError(
                    f"Step {index}: destination is already claimed by an earlier step: {path}"
                )
            if path.exists() and path not in freed:
                raise StagingError(
                    f"Step {index}: destination already exists, refusing to overwrite: {path}"
                )

        freed.update(sources)
        freed.difference_update(destinations)
        claimed.update(destinations)
        claimed.difference_update(sources)


def _move_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def apply_plan(steps):
    """Executes a validated plan, rolling back moves already made on failure."""
    done = []
    try:
        for source, destination in steps:
            for source_path, destination_path in transfer_pairs(source, destination):
                _move_file(source_path, destination_path)
                done.append((source_path, destination_path))
    except Exception:
        for original, moved in reversed(done):
            try:
                _move_file(moved, original)
            except OSError:
                # The rollback itself failed — leave things as they are; the journal and the
                # error message provide enough data to sort it out manually.
                pass
        raise


def _run(action, steps, journal_path, dry_run, extra=None):
    validate_plan(steps)
    if not dry_run:
        apply_plan(steps)

    entry = {
        "action": action,
        "at": utc_now_iso(),
        "dryRun": bool(dry_run),
        "moves": [
            {"from": source.as_dict(), "to": destination.as_dict()}
            for source, destination in steps
        ],
    }
    if extra:
        entry.update(extra)
    if not dry_run:
        append_journal_entry(journal_path, entry)
    return entry


def _reject_partial_active(bundle):
    if bundle.partial:
        raise StagingError(
            "Active slot holds an incomplete bundle; resolve it manually first: "
            + ", ".join(bundle.missing)
        )


def stage_active(
    bucket_dir,
    stem,
    journal_path=DEFAULT_JOURNAL_PATH,
    dry_run=False,
):
    """Moves an unverified candidate from active to staging.

    Called right after training: the model the training script has already made
    active has not passed the gate yet and must not serve inference.
    """
    source = active_bundle(bucket_dir, stem)
    destination = staged_bundle(bucket_dir, stem)
    return _run(
        "stage",
        [(source, destination)],
        journal_path,
        dry_run,
        {"stem": stem, "bucket": str(bucket_dir)},
    )


def archive_active(
    bucket_dir,
    stem,
    timestamp=None,
    journal_path=DEFAULT_JOURNAL_PATH,
    dry_run=False,
):
    """Moves the current active pair to archive under the given timestamp."""
    timestamp = timestamp or artifact_timestamp()
    source = active_bundle(bucket_dir, stem)
    destination = archived_bundle(bucket_dir, stem, timestamp)
    return _run(
        "archive_active",
        [(source, destination)],
        journal_path,
        dry_run,
        {"stem": stem, "bucket": str(bucket_dir), "timestamp": timestamp},
    )


def promote_staged(
    bucket_dir,
    stem,
    timestamp=None,
    journal_path=DEFAULT_JOURNAL_PATH,
    dry_run=False,
):
    """Makes the staged candidate active, archiving the previous active first.

    Call only after a PASS verdict.
    """
    candidate = staged_bundle(bucket_dir, stem)
    if not candidate.complete:
        raise StagingError(
            "No complete staged bundle to promote: " + ", ".join(candidate.missing)
        )

    current_active = active_bundle(bucket_dir, stem)
    _reject_partial_active(current_active)

    timestamp = timestamp or artifact_timestamp()
    steps = []
    if current_active.complete:
        steps.append((current_active, archived_bundle(bucket_dir, stem, timestamp)))
    steps.append((candidate, current_active))

    return _run(
        "promote",
        steps,
        journal_path,
        dry_run,
        {"stem": stem, "bucket": str(bucket_dir), "timestamp": timestamp},
    )


def reject_staged(
    bucket_dir,
    stem,
    timestamp=None,
    journal_path=DEFAULT_JOURNAL_PATH,
    dry_run=False,
):
    """Moves a rejected candidate from staging to archive. Leaves active untouched."""
    candidate = staged_bundle(bucket_dir, stem)
    if not candidate.complete:
        raise StagingError(
            "No complete staged bundle to reject: " + ", ".join(candidate.missing)
        )
    timestamp = timestamp or artifact_timestamp()
    destination = archived_bundle(bucket_dir, stem, f"{REJECTED_PREFIX}_{timestamp}")
    return _run(
        "reject",
        [(candidate, destination)],
        journal_path,
        dry_run,
        {"stem": stem, "bucket": str(bucket_dir), "timestamp": timestamp},
    )


def rollback_to_archived(
    bucket_dir,
    stem,
    timestamp=None,
    journal_path=DEFAULT_JOURNAL_PATH,
    dry_run=False,
):
    """Restores an archived version to active.

    Without a timestamp, the newest complete archived pair is used. The current active is moved
    to archive with a `demoted_` tag rather than deleted, so the operation is reversible.
    """
    if timestamp is None:
        newest = latest_archived(bucket_dir, stem)
        if newest is None:
            raise StagingError(f"No complete archived bundle found for {stem}")
        timestamp = newest["timestamp"]

    source = archived_bundle(bucket_dir, stem, timestamp)
    if not source.complete:
        raise StagingError(
            f"Archived bundle {stem} @ {timestamp} is incomplete: " + ", ".join(source.missing)
        )

    current_active = active_bundle(bucket_dir, stem)
    _reject_partial_active(current_active)

    steps = []
    if current_active.complete:
        demoted = archived_bundle(
            bucket_dir, stem, f"{DEMOTED_PREFIX}_{artifact_timestamp()}"
        )
        steps.append((current_active, demoted))
    steps.append((source, current_active))

    return _run(
        "rollback",
        steps,
        journal_path,
        dry_run,
        {"stem": stem, "bucket": str(bucket_dir), "restoredTimestamp": timestamp},
    )


def describe(bucket_dir, stem):
    """Slot state for stem. Changes nothing."""
    active = active_bundle(bucket_dir, stem)
    staged = staged_bundle(bucket_dir, stem)
    return {
        "bucket": str(bucket_dir),
        "stem": stem,
        "active": {**active.as_dict(), "complete": active.complete},
        "staged": {**staged.as_dict(), "complete": staged.complete},
        "archived": [
            {"timestamp": item["timestamp"], **item["bundle"].as_dict()}
            for item in list_archived(bucket_dir, stem)
        ],
    }


def find_active_bundles(models_root=DEFAULT_MODELS_ROOT):
    """All complete active pairs under models_root, skipping archive and staging."""
    models_root = Path(models_root)
    if not models_root.is_dir():
        return []

    bundles = []
    for model_path in sorted(models_root.rglob(f"*{MODEL_SUFFIX}")):
        if NON_ACTIVE_DIR_NAMES.intersection(model_path.parts):
            continue
        bundle = active_bundle(model_path.parent, model_path.stem)
        if not bundle.complete:
            continue
        bundles.append(
            {"bucket": model_path.parent, "stem": model_path.stem, "bundle": bundle}
        )
    return bundles


def fingerprint_active_bundles(models_root=DEFAULT_MODELS_ROOT):
    """Snapshot of active pairs for comparison before and after training.

    Tells which bundle training created or replaced, without parsing
    the training script's stdout or guessing the accuracy category in advance.
    """
    snapshot = {}
    for item in find_active_bundles(models_root):
        stat = item["bundle"].model_path.stat()
        snapshot[str(item["bundle"].model_path)] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "bucket": str(item["bucket"]),
            "stem": item["stem"],
        }
    return snapshot


def diff_active_bundles(before, after):
    """Pairs that appeared or changed between two snapshots."""
    changed = []
    for path, current in sorted(after.items()):
        previous = before.get(path)
        if previous is None:
            changed.append({**current, "model_path": path, "change": "created"})
        elif (previous["size"], previous["mtime_ns"]) != (
            current["size"],
            current["mtime_ns"],
        ):
            changed.append({**current, "model_path": path, "change": "replaced"})
    return changed


MUTATING_ACTIONS = {
    "stage": stage_active,
    "promote": promote_staged,
    "reject": reject_staged,
    "rollback": rollback_to_archived,
    "archive-active": archive_active,
}
ACTIONS = ("status",) + tuple(sorted(MUTATING_ACTIONS))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Move model artifacts between active, staging and archive"
    )
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument(
        "--bucket",
        required=True,
        help="Category directory, e.g. models/saved_models/gru_saved_model/95-100%%",
    )
    parser.add_argument(
        "--stem",
        required=True,
        help="Artifact name without extension, e.g. SOLUSDT_SOLUSDC_15m_30d_gru",
    )
    parser.add_argument(
        "--timestamp",
        help="For rollback — which archived version to restore; defaults to the newest",
    )
    parser.add_argument("--journal", default=str(DEFAULT_JOURNAL_PATH))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the plan without moving anything",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bucket_dir = Path(args.bucket)
    if not bucket_dir.is_dir():
        print(f"Bucket directory not found: {bucket_dir}", file=sys.stderr)
        return 1

    try:
        if args.action == "status":
            result = describe(bucket_dir, args.stem)
        elif args.action in ("rollback", "archive-active"):
            result = MUTATING_ACTIONS[args.action](
                bucket_dir,
                args.stem,
                timestamp=args.timestamp,
                journal_path=args.journal,
                dry_run=args.dry_run,
            )
        else:
            result = MUTATING_ACTIONS[args.action](
                bucket_dir,
                args.stem,
                journal_path=args.journal,
                dry_run=args.dry_run,
            )
    except StagingError as error:
        print(f"Staging refused: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
