#!/usr/bin/env python3
"""Build a read-only content inventory for a reference repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cliffordnet_cl20.reference_inventory import build_inventory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_inventory(args.root), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
