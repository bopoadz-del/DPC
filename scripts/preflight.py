"""Render build preflight checks for deployment readiness."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

MERGE_MARKERS = ("<" * 5, "=" * 5, ">" * 5)
TEXT_SUFFIXES = {".txt", ".py", ".md", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}


def iter_candidate_files(root: Path) -> Iterable[Path]:
    """Yield project files that are likely text and safe to scan for merge markers."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = Path(dirpath, filename)
            if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"render.yaml", "README"}:
                yield path


def ensure_no_merge_conflicts(path: Path) -> None:
    """Raise SystemExit if a file contains git merge markers."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Binary file – ignore.
        return
    for marker in MERGE_MARKERS:
        if marker in text:
            raise SystemExit(
                f"Merge marker '{marker}' detected in {path}. Resolve the conflict before deploying."
            )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    requirements = project_root / "requirements.txt"
    if not requirements.exists():
        raise SystemExit("requirements.txt not found; verify repository contents.")
    ensure_no_merge_conflicts(requirements)

    for candidate in iter_candidate_files(project_root):
        ensure_no_merge_conflicts(candidate)

    print("✅ Merge-marker check passed for requirements.txt and project sources.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(f"❌ Preflight failed: {exc}")
        raise
