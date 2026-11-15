from pathlib import Path

from cliffordnet_cl20.reference_inventory import build_inventory


def test_inventory_detects_content_addition_and_deletion(tmp_path: Path) -> None:
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("first", encoding="utf-8")
    initial = build_inventory(tmp_path)

    tracked.write_text("second", encoding="utf-8")
    changed = build_inventory(tmp_path)
    assert changed != initial

    added = tmp_path / "added.txt"
    added.write_text("new", encoding="utf-8")
    with_addition = build_inventory(tmp_path)
    assert with_addition != changed

    added.unlink()
    after_deletion = build_inventory(tmp_path)
    assert after_deletion == changed


def test_inventory_ignores_git_directory(tmp_path: Path) -> None:
    (tmp_path / "payload.txt").write_text("stable", encoding="utf-8")
    before = build_inventory(tmp_path)

    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "index").write_bytes(b"volatile")

    assert build_inventory(tmp_path) == before
