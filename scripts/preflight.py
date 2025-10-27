"""Render build preflight checks for deployment readiness."""
from __future__ import annotations

from pathlib import Path

MERGE_MARKERS = ("<" * 5, "=" * 5, ">" * 5)


def ensure_no_merge_conflicts(path: Path) -> None:
    """Raise SystemExit if a file contains git merge markers."""
    text = path.read_text(encoding="utf-8")
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
    print("✅ requirements.txt passes merge-marker check.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        print(f"❌ Preflight failed: {exc}")
        raise
