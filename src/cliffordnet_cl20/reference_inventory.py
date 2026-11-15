"""Read-only content inventory for reference repositories."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_status(root: Path) -> list[str]:
    if not (root / ".git").exists():
        return []
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v2"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def build_inventory(root: Path) -> dict[str, object]:
    """Return stable hashes and Git status without modifying *root*."""
    root = root.resolve()
    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts or not path.is_file():
            continue
        files.append({"path": relative.as_posix(), "size": path.stat().st_size, "sha256": _sha256(path)})
    return {"root_name": root.name, "git_status": _git_status(root), "files": files}
