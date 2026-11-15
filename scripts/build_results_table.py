#!/usr/bin/env python3
"""Generate or check README result tables from the validated manifest."""

import argparse
from pathlib import Path

import yaml

from cliffordnet_cl20.results import render_results_table, validate_manifest

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def expected_readme(root: Path) -> str:
    manifest = yaml.safe_load((root / "results/manifest.yaml").read_text())
    validate_manifest(manifest, root)
    readme = (root / "README.md").read_text()
    before, remainder = readme.split(START, 1)
    _, after = remainder.split(END, 1)
    return before + START + "\n\n" + render_results_table(manifest) + "\n\n" + END + after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = expected_readme(root)
    path = root / "README.md"
    if args.check:
        if path.read_text() != expected:
            raise SystemExit("README result tables are stale; run scripts/build_results_table.py")
    else:
        path.write_text(expected)


if __name__ == "__main__":
    main()
