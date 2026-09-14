"""Simulate the fitted J0012-3954 flux model on the 0012-399 sampling."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simulations import process_0012_399


def main() -> dict:
    return process_0012_399()


if __name__ == "__main__":
    main()
