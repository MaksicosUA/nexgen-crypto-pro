import argparse
import fcntl
import json
import os
import re
import stat
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from controller.forecast_producer import record_identity, validate_contract_record


DEFAULT_REPOSITORY = PROJECT_DIR / "state" / "forecast-journal"
DEFAULT_JOURNAL_NAME = "forecasts.jsonl"
DEFAULT_REMOTE = "origin"
DEFAULT_BRANCH = "journal"
MAX_PUBLISH_TIMEOUT_SECONDS = 120
REMOTE_AUDIT_REF = "refs/nexgen-forecast/remote-tip"


def run_git(repository, *arguments, check=True, timeout=None, text=True):
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            capture_output=True,
            text=text,
            check=False,
            timeout=timeout,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"git {' '.join(arguments)} timed out after {timeout} seconds"
        ) from exc
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        detail = re.sub(r"(https?://)[^/@\s]+@", r"\1***@", detail)
        raise RuntimeError(
            f"git {' '.join(arguments)} failed with exit code {result.returncode}: {detail}"
        )
    return result


def validate_working_journal_file(repository, journal_name):
    journal_path = Path(repository) / journal_name
    if journal_path.is_symlink():
        raise ValueError("Forecast journal must not be a symlink")
    try:
        journal_mode = journal_path.stat(follow_symlinks=False).st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"Forecast journal not found: {journal_path}")
    if not stat.S_ISREG(journal_mode) or journal_mode & 0o111:
        raise ValueError("Forecast journal must be a non-executable regular file")
    return journal_path


def validate_repository(repository, journal_name, remote, branch):
    repository = Path(repository).resolve()
    if not repository.is_dir():
        raise FileNotFoundError(f"Forecast journal repository not found: {repository}")
    if Path(journal_name).name != journal_name or journal_name in {"", ".", ".."}:
        raise ValueError("journal_name must be a file in the repository root")
    if repository == PROJECT_DIR.resolve():
        raise ValueError("Forecast journal must live in a separate Git repository")

    top_level = Path(
        run_git(repository, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve()
    if top_level != repository:
        raise ValueError(
            f"--repo must point to the separate repository root; detected {top_level}"
        )
    common_dir_value = run_git(repository, "rev-parse", "--git-common-dir").stdout.strip()
    common_dir = Path(common_dir_value)
    if not common_dir.is_absolute():
        common_dir = repository / common_dir
    if common_dir.resolve() == (PROJECT_DIR / ".git").resolve():
        raise ValueError(
            "Forecast journal must not share the NEXGEN Git common directory"
        )

    current_branch = run_git(repository, "branch", "--show-current").stdout.strip()
    if current_branch != branch:
        raise ValueError(
            f"Forecast journal repository must be on branch {branch!r}, "
            f"currently on {current_branch!r}"
        )

    run_git(repository, "remote", "get-url", remote)
    index_entry = run_git(
        repository,
        "ls-files",
        "--stage",
        "--",
        journal_name,
    ).stdout.splitlines()
    if len(index_entry) != 1 or not index_entry[0].startswith("100644 "):
        raise ValueError(
            f"{journal_name} must be one tracked, non-executable regular file (mode 100644)"
        )
    validate_working_journal_file(repository, journal_name)
    git_dir = Path(
        run_git(repository, "rev-parse", "--absolute-git-dir").stdout.strip()
    ).resolve()
    return repository, git_dir


def changed_paths(repository):
    paths = set()
    commands = (
        ("diff", "--name-only"),
        ("diff", "--cached", "--name-only"),
        ("ls-files", "--others", "--exclude-standard"),
    )
    for command in commands:
        output = run_git(repository, *command).stdout
        paths.update(line for line in output.splitlines() if line)
    return paths


def remaining_timeout(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise RuntimeError("Forecast journal publish deadline expired")
    return remaining


def fetch_remote_tip(repository, remote, branch, deadline):
    advertised = run_git(
        repository,
        "ls-remote",
        "--exit-code",
        "--heads",
        remote,
        f"refs/heads/{branch}",
        timeout=remaining_timeout(deadline),
    ).stdout.splitlines()
    if len(advertised) != 1:
        raise RuntimeError(
            f"Expected exactly one remote branch {branch!r}, got {len(advertised)}"
        )
    remote_tip = advertised[0].split()[0]
    run_git(
        repository,
        "fetch",
        "--no-tags",
        "--quiet",
        remote,
        remote_tip,
        timeout=remaining_timeout(deadline),
    )

    previous_audit_tip = run_git(
        repository,
        "rev-parse",
        "--verify",
        "--quiet",
        REMOTE_AUDIT_REF,
        check=False,
    )
    if previous_audit_tip.returncode == 0:
        previous_tip = previous_audit_tip.stdout.strip()
        fast_forward = run_git(
            repository,
            "merge-base",
            "--is-ancestor",
            previous_tip,
            remote_tip,
            check=False,
        )
        if fast_forward.returncode != 0:
            raise ValueError(
                "Remote journal history was deleted, rewritten or diverged; refusing publish"
            )
    else:
        previous_tip = None
    return remote_tip, previous_tip


def validate_journal_content(content):
    if content and not content.endswith("\n"):
        raise ValueError("Forecast journal does not end with a newline")

    records = []
    latest_by_series = {}
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Blank line found in forecast journal at line {line_number}")
        try:
            record = json.loads(line)
            validate_contract_record(record)
        except Exception as exc:
            raise ValueError(
                f"Invalid forecast journal line {line_number}: {exc}"
            ) from exc

        identity = record_identity(record)
        timestamp = datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
        previous = latest_by_series.get(identity)
        if previous is not None and timestamp <= previous:
            raise ValueError(
                "Forecast journal is not strictly append-ordered for series "
                f"{identity}: line {line_number} has {record['ts']}"
            )
        latest_by_series[identity] = timestamp
        records.append(record)
    return records


def read_git_blob(repository, revision, journal_name):
    return run_git(
        repository,
        "show",
        f"{revision}:{journal_name}",
        text=False,
    ).stdout


def validate_tree_entry(repository, revision, journal_name):
    entries = run_git(
        repository,
        "ls-tree",
        revision,
        "--",
        journal_name,
    ).stdout.splitlines()
    if len(entries) != 1 or not entries[0].startswith("100644 blob "):
        raise ValueError(
            f"{revision}:{journal_name} must be a regular non-executable Git blob"
        )


def decode_and_validate_journal(content_bytes):
    try:
        content = content_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Forecast journal must be valid UTF-8") from exc
    return content, validate_journal_content(content)


def first_step_maturity(record):
    timestamp = datetime.fromisoformat(record["ts"].replace("Z", "+00:00"))
    first_horizon = record["steps"][0]["h"]
    return timestamp + timedelta(
        minutes=record["timeframeMin"] * (1 + first_horizon)
    )


def ensure_append_only_prefix(repository, journal_name, current_bytes):
    committed = read_git_blob(repository, "HEAD", journal_name)
    if not current_bytes.startswith(committed):
        raise ValueError(
            "Forecast journal changed or removed previously committed bytes; "
            "only appending new complete lines is allowed"
        )
    return committed


def validate_unpublished_history(
    repository,
    remote_tip,
    journal_name,
    current_bytes,
    local_tip="HEAD",
):
    ancestor = run_git(
        repository,
        "merge-base",
        "--is-ancestor",
        remote_tip,
        local_tip,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError(
            "Remote journal tip is not an ancestor of local HEAD; "
            "refusing automatic merge, rebase or overwrite"
        )

    remote_content = read_git_blob(repository, remote_tip, journal_name)
    validate_tree_entry(repository, remote_tip, journal_name)
    decode_and_validate_journal(remote_content)
    if not current_bytes.startswith(remote_content):
        raise ValueError(
            "Local journal does not preserve the complete remote byte-prefix"
        )

    all_commits = run_git(
        repository,
        "rev-list",
        "--reverse",
        f"{remote_tip}..{local_tip}",
    ).stdout.splitlines()
    first_parent_commits = run_git(
        repository,
        "rev-list",
        "--reverse",
        "--first-parent",
        f"{remote_tip}..{local_tip}",
    ).stdout.splitlines()
    if all_commits != first_parent_commits:
        raise ValueError("Unpublished journal history must be linear and contain no merges")

    previous_commit = remote_tip
    previous_content = remote_content
    for commit in all_commits:
        parents = run_git(
            repository,
            "show",
            "-s",
            "--format=%P",
            commit,
        ).stdout.split()
        if parents != [previous_commit]:
            raise ValueError(
                f"Unpublished journal commit {commit} is not a linear continuation"
            )

        commit_paths = set(
            run_git(
                repository,
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "-r",
                commit,
            ).stdout.splitlines()
        )
        commit_paths.discard("")
        if commit_paths != {journal_name}:
            raise ValueError(
                f"Unpublished commit {commit} changes files outside the journal: "
                + ", ".join(sorted(commit_paths))
            )

        commit_content = read_git_blob(repository, commit, journal_name)
        validate_tree_entry(repository, commit, journal_name)
        decode_and_validate_journal(commit_content)
        if commit_content == previous_content or not commit_content.startswith(
            previous_content
        ):
            raise ValueError(
                f"Unpublished commit {commit} is not a strict append-only journal update"
            )
        previous_commit = commit
        previous_content = commit_content

    if not current_bytes.startswith(previous_content):
        raise ValueError(
            "Working journal changed or removed bytes already present in local HEAD"
        )


def validate_remote_history(
    repository,
    previous_audit_tip,
    remote_tip,
    journal_name,
):
    remote_content = read_git_blob(repository, remote_tip, journal_name)
    if previous_audit_tip is None:
        parents = run_git(
            repository,
            "show",
            "-s",
            "--format=%P",
            remote_tip,
        ).stdout.split()
        tree_paths = set(
            run_git(
                repository,
                "ls-tree",
                "-r",
                "--name-only",
                remote_tip,
            ).stdout.splitlines()
        )
        tree_paths.discard("")
        validate_tree_entry(repository, remote_tip, journal_name)
        decode_and_validate_journal(remote_content)
        if parents or tree_paths != {journal_name} or remote_content:
            raise ValueError(
                "First trusted remote journal must be one empty root commit "
                f"containing only {journal_name}"
            )
        return

    validate_unpublished_history(
        repository,
        previous_audit_tip,
        journal_name,
        remote_content,
        local_tip=remote_tip,
    )


def acquire_publish_lock(lock_file, deadline):
    while True:
        try:
            fcntl.flock(
                lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
            return
        except BlockingIOError as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Forecast journal publisher lock timed out before the shared "
                    "publish deadline"
                ) from exc
            time.sleep(0.1)


def publish_forecast_journal(
    repository,
    journal_name=DEFAULT_JOURNAL_NAME,
    remote=DEFAULT_REMOTE,
    branch=DEFAULT_BRANCH,
    publish_timeout_seconds=MAX_PUBLISH_TIMEOUT_SECONDS,
):
    if not 0 < publish_timeout_seconds <= MAX_PUBLISH_TIMEOUT_SECONDS:
        raise ValueError(
            "publish_timeout_seconds must be between 0 and "
            f"{MAX_PUBLISH_TIMEOUT_SECONDS}"
        )
    repository, git_dir = validate_repository(
        repository=repository,
        journal_name=journal_name,
        remote=remote,
        branch=branch,
    )
    publish_deadline = time.monotonic() + publish_timeout_seconds
    publish_lock_path = git_dir / "nexgen-forecast-publish.lock"
    with publish_lock_path.open("a+", encoding="utf-8") as publish_lock:
        acquire_publish_lock(publish_lock, publish_deadline)

        unexpected = changed_paths(repository) - {journal_name}
        if unexpected:
            raise ValueError(
                "Forecast journal repository contains changes outside the journal: "
                + ", ".join(sorted(unexpected))
            )
        staged = set(
            run_git(repository, "diff", "--cached", "--name-only").stdout.splitlines()
        )
        staged.discard("")
        if staged:
            raise ValueError(
                "Forecast journal repository must not contain pre-staged changes: "
                + ", ".join(sorted(staged))
            )

        remote_tip, previous_audit_tip = fetch_remote_tip(
            repository,
            remote,
            branch,
            publish_deadline,
        )
        validate_remote_history(
            repository,
            previous_audit_tip,
            remote_tip,
            journal_name,
        )
        # Anchor the validated remote before attempting a local commit/push. This
        # closes the crash window between a successful first push and audit-ref update.
        if previous_audit_tip is None:
            run_git(
                repository,
                "update-ref",
                REMOTE_AUDIT_REF,
                remote_tip,
                "0" * 40,
            )
            previous_audit_tip = remote_tip
        elif previous_audit_tip != remote_tip:
            run_git(
                repository,
                "update-ref",
                REMOTE_AUDIT_REF,
                remote_tip,
                previous_audit_tip,
            )
            previous_audit_tip = remote_tip

        journal_path = validate_working_journal_file(repository, journal_name)

        with journal_path.open("rb") as journal:
            fcntl.flock(journal.fileno(), fcntl.LOCK_EX)
            try:
                current_bytes = journal.read()
                _, records = decode_and_validate_journal(current_bytes)
                validate_unpublished_history(
                    repository,
                    remote_tip,
                    journal_name,
                    current_bytes,
                )
                remote_bytes = read_git_blob(repository, remote_tip, journal_name)
                _, pending_records = decode_and_validate_journal(
                    current_bytes[len(remote_bytes) :]
                )
                committed = ensure_append_only_prefix(
                    repository,
                    journal_name,
                    current_bytes,
                )

                journal_changed = journal_name in changed_paths(repository)
                if current_bytes == committed and journal_changed:
                    raise ValueError(
                        "Forecast journal has a non-content Git change; "
                        "file mode/type changes are forbidden"
                    )

                committed_now = False
                if current_bytes != committed:
                    latest_timestamp = max(record["ts"] for record in records)
                    run_git(
                        repository,
                        "-c",
                        "core.hooksPath=/dev/null",
                        "commit",
                        "--only",
                        "--no-gpg-sign",
                        "-m",
                        f"journal: append forecasts through {latest_timestamp}",
                        "--",
                        journal_name,
                        timeout=remaining_timeout(publish_deadline),
                    )
                    committed_now = True

                publish_tip = run_git(repository, "rev-parse", "HEAD").stdout.strip()
                if read_git_blob(repository, publish_tip, journal_name) != current_bytes:
                    raise RuntimeError(
                        "Committed journal blob differs from the locked working file"
                    )
                validate_tree_entry(repository, publish_tip, journal_name)
                validate_unpublished_history(
                    repository,
                    remote_tip,
                    journal_name,
                    current_bytes,
                    local_tip=publish_tip,
                )
            finally:
                fcntl.flock(journal.fileno(), fcntl.LOCK_UN)

        unexpected_after_commit = changed_paths(repository) - {journal_name}
        if unexpected_after_commit:
            raise ValueError(
                "Forecast journal repository changed outside the journal during publish: "
                + ", ".join(sorted(unexpected_after_commit))
            )

        # Push is attempted even without a new commit so a previous network failure
        # can recover on the next cron run. Non-fast-forward updates are never forced.
        run_git(
            repository,
            "push",
            remote,
            f"{publish_tip}:refs/heads/{branch}",
            timeout=remaining_timeout(publish_deadline),
        )
        run_git(
            repository,
            "update-ref",
            REMOTE_AUDIT_REF,
            publish_tip,
            previous_audit_tip,
        )
        publisher_observed_at = datetime.now(timezone.utc)
        late_records = sum(
            publisher_observed_at >= first_step_maturity(record)
            for record in pending_records
        )
        return {
            "status": "published" if committed_now else "up-to-date",
            "records": len(records),
            "pendingRecords": len(pending_records),
            "lateRecords": late_records,
            "publisherObservedAt": publisher_observed_at.isoformat().replace(
                "+00:00", "Z"
            ),
            "branch": branch,
            "commit": publish_tip,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate, commit and push an append-only forecasts.jsonl from its "
            "dedicated Git repository"
        )
    )
    parser.add_argument("--repo", default=str(DEFAULT_REPOSITORY))
    parser.add_argument("--journal-name", default=DEFAULT_JOURNAL_NAME)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--publish-timeout-seconds",
        type=int,
        default=MAX_PUBLISH_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = publish_forecast_journal(
        repository=args.repo,
        journal_name=args.journal_name,
        remote=args.remote,
        branch=args.branch,
        publish_timeout_seconds=args.publish_timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
