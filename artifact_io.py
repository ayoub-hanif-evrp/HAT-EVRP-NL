"""Safeguards for checkpoints, raw CSVs, and experiment provenance."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROTECTED_CHECKPOINT_GLOBS = (
    "*_c3.pt",
    "*_c2.pt",
    "*_bc_only_rq4.pt",
    "*_rl_from_bc_rq4.pt",
)


def git_commit_sha(root: Path | None = None) -> str:
    root = root or Path(__file__).resolve().parent
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def environment_metadata(root: Path | None = None) -> dict:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git_commit_sha(root),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        meta["torch"] = "unavailable"
    try:
        import frvcpy

        meta["frvcpy"] = getattr(frvcpy, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        meta["frvcpy"] = "unavailable"
    return meta


def is_protected_checkpoint(path: Path) -> bool:
    name = path.name
    for pattern in PROTECTED_CHECKPOINT_GLOBS:
        if path.match(pattern) or Path(name).match(pattern):
            return True
    # Also protect exact legacy seed dumps that look historical.
    return False


def safe_save_checkpoint(path: Path, saver, *, overwrite: bool = False) -> Path:
    """Save a checkpoint without clobbering protected historical files.

    ``saver`` is a callable ``saver(path)`` that writes the file.
    If the target exists and overwrite is False (or the file is protected),
    writes to ``path.with_name(path.stem + '_new' + path.suffix)`` or a
    timestamped name if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if path.exists():
        protected = is_protected_checkpoint(path)
        if protected and not overwrite:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            target = path.with_name(f"{path.stem}_new_{stamp}{path.suffix}")
        elif not overwrite:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            target = path.with_name(f"{path.stem}_new_{stamp}{path.suffix}")
        elif protected and overwrite:
            # Explicit overwrite of protected artifacts requires env confirmation.
            if os.environ.get("HAT_EVRP_ALLOW_CHECKPOINT_OVERWRITE") != "1":
                raise PermissionError(
                    f"Refusing to overwrite protected checkpoint {path}. "
                    "Set HAT_EVRP_ALLOW_CHECKPOINT_OVERWRITE=1 to force."
                )
    saver(target)
    return target


def safe_write_csv(path: Path, write_fn, *, overwrite: bool = False) -> Path:
    """Write a CSV without silently overwriting existing raw evidence."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    if path.exists() and not overwrite:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    write_fn(target)
    return target


def make_run_directory(base: Path, study_name: str) -> Path:
    """Create ``base/study_name/<commit>_<timestamp>/`` for new experiments."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = git_commit_sha()[:8]
    run_dir = Path(base) / study_name / f"{commit}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = environment_metadata()
    meta["study"] = study_name
    (run_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return run_dir
