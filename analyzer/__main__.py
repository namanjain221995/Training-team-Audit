"""Entrypoint: python -m analyzer  (docker / local mode)

Loads config from env + mounted /data paths, then runs the same pipeline the
EC2 worker uses (analyzer.api.run).
"""
import sys

from .config import load_config
from .api import run


def main() -> int:
    return run(load_config())


if __name__ == "__main__":
    sys.exit(main())
