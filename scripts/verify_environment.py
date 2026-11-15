#!/usr/bin/env python3
"""Print a machine-readable environment report."""

import json

from cliffordnet_cl20.reproducibility import collect_environment

if __name__ == "__main__":
    print(json.dumps(collect_environment(), indent=2, sort_keys=True))
