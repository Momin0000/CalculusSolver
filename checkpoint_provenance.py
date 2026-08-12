"""
Checkpoint provenance stamping.

Answers, for any checkpoint file, the question the original brief and the
Dev 1 audit both got stuck on: "what exact code and config produced this?"

Every saved checkpoint gets stamped with:
  - the exact git commit hash at save time (+ whether the working tree was dirty)
  - a sha256 hash of the config file it was trained under
  - a sha256 hash of the checkpoint file itself (so the numbers can be tied
    back to one specific binary, not just "whatever's in checkpoints/final/")
  - a UTC timestamp

This is called automatically from train.py right after a checkpoint is saved
-- nobody should be hand-editing docs/TRAINING_RESULTS.md's provenance block.

It's also runnable standalone, which is the tool for Task 3 (re-verifying a
checkpoint someone else hands off): run it against their checkpoint file
before trusting any accuracy number they report.

Usage as a script:
    python checkpoint_provenance.py checkpoints/final/best.pt
    python checkpoint_provenance.py checkpoints/final/best.pt --config config.json
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def get_git_commit_hash(repo_dir: Optional[str] = None) -> dict:
    """
    Returns the current commit hash and whether the working tree has
    uncommitted changes. A checkpoint saved with a dirty tree can't be
    reproduced from the commit alone -- that's flagged, not hidden.
    """
    cwd = repo_dir or "."
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": None, "error": "not a git repo or git unavailable"}

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = None

    return {"commit": commit, "dirty": dirty, "error": None}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_config_hash(config_path: str = "config.json") -> dict:
    p = Path(config_path)
    if not p.exists():
        return {"config_path": config_path, "config_sha256": None, "error": "config file not found"}
    return {"config_path": config_path, "config_sha256": _sha256_of_file(p), "error": None}


def get_checkpoint_hash(checkpoint_path: str) -> dict:
    p = Path(checkpoint_path)
    if not p.exists():
        return {"checkpoint_path": checkpoint_path, "checkpoint_sha256": None, "error": "checkpoint file not found"}
    return {"checkpoint_path": checkpoint_path, "checkpoint_sha256": _sha256_of_file(p), "error": None}


def stamp_checkpoint(checkpoint_path: str, config_path: str = "config.json", repo_dir: Optional[str] = None) -> dict:
    """Assemble the full provenance record for one checkpoint."""
    git_info = get_git_commit_hash(repo_dir)
    config_info = get_config_hash(config_path)
    ckpt_info = get_checkpoint_hash(checkpoint_path)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": ckpt_info["checkpoint_sha256"],
        "git_commit": git_info["commit"],
        "git_dirty": git_info["dirty"],
        "config_path": config_path,
        "config_sha256": config_info["config_sha256"],
        "warnings": [
            w for w in [
                "checkpoint file not found -- hash unavailable" if ckpt_info["error"] else None,
                "config file not found -- hash unavailable" if config_info["error"] else None,
                "not inside a git repo -- commit unavailable" if git_info["error"] else None,
                "working tree has uncommitted changes at save time" if git_info["dirty"] else None,
            ] if w
        ],
    }


def render_provenance_markdown(record: dict) -> str:
    """Render one provenance record as a Markdown block for TRAINING_RESULTS.md."""
    lines = [
        "## Checkpoint Provenance",
        "",
        f"- **Checkpoint:** `{record['checkpoint_path']}`",
        f"- **Checkpoint SHA256:** `{record['checkpoint_sha256'] or 'N/A'}`",
        f"- **Git Commit:** `{record['git_commit'] or 'N/A'}`"
        + (" (dirty working tree)" if record["git_dirty"] else ""),
        f"- **Config File:** `{record['config_path']}`",
        f"- **Config SHA256:** `{record['config_sha256'] or 'N/A'}`",
        f"- **Stamped At (UTC):** {record['timestamp_utc']}",
    ]
    if record["warnings"]:
        lines.append("")
        lines.append("**Warnings:**")
        for w in record["warnings"]:
            lines.append(f"- ⚠️ {w}")
    lines.append("")
    return "\n".join(lines)


def append_provenance_to_training_results(
    record: dict,
    training_results_path: str = "docs/TRAINING_RESULTS.md",
) -> None:
    """
    Appends (or replaces, if one already exists) the Checkpoint Provenance
    section at the end of docs/TRAINING_RESULTS.md. This is the automatic
    write path -- nobody should be pasting commit hashes into this file
    by hand.
    """
    path = Path(training_results_path)
    block = render_provenance_markdown(record)

    if path.exists():
        existing = path.read_text(encoding="utf-8")
        marker = "## Checkpoint Provenance"
        if marker in existing:
            head = existing.split(marker)[0].rstrip()
            new_content = head + "\n\n" + block
        else:
            new_content = existing.rstrip() + "\n\n" + block
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_content = "# Training Results\n\n" + block

    path.write_text(new_content, encoding="utf-8")


def stamp_and_record(
    checkpoint_path: str,
    config_path: str = "config.json",
    training_results_path: str = "docs/TRAINING_RESULTS.md",
    repo_dir: Optional[str] = None,
) -> dict:
    """One-call helper: stamp a checkpoint and write the record into TRAINING_RESULTS.md."""
    record = stamp_checkpoint(checkpoint_path, config_path, repo_dir)
    append_provenance_to_training_results(record, training_results_path)
    return record


def _main():
    parser = argparse.ArgumentParser(
        description="Stamp a checkpoint with git commit + config hash provenance. "
        "Use standalone to re-verify a checkpoint someone else hands off, before "
        "trusting any accuracy number reported against it."
    )
    parser.add_argument("checkpoint", help="Path to the checkpoint file, e.g. checkpoints/final/best.pt")
    parser.add_argument("--config", default="config.json", help="Path to the config file (default: config.json)")
    parser.add_argument(
        "--training-results",
        default="docs/TRAINING_RESULTS.md",
        help="Path to TRAINING_RESULTS.md to write the provenance block into. "
        "Pass --no-write to only print, without writing.",
    )
    parser.add_argument("--no-write", action="store_true", help="Print the record only, don't write to TRAINING_RESULTS.md")
    args = parser.parse_args()

    record = stamp_checkpoint(args.checkpoint, args.config)

    print(json.dumps(record, indent=2))

    if not args.no_write:
        append_provenance_to_training_results(record, args.training_results)
        print(f"\nWritten to {args.training_results}", file=sys.stderr)

    if record["warnings"]:
        print("\nWarnings:", file=sys.stderr)
        for w in record["warnings"]:
            print(f"  - {w}", file=sys.stderr)


if __name__ == "__main__":
    _main()
