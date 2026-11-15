#!/usr/bin/env python3
"""Check a checkpoint against a registered architecture without executing it."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cliffordnet_cl20.engine import verify_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("model_name")
    args = parser.parse_args()
    report = verify_checkpoint(args.checkpoint, args.model_name)
    print(json.dumps(asdict(report), indent=2))
    raise SystemExit(0 if report.compatible else 1)


if __name__ == "__main__":
    main()
