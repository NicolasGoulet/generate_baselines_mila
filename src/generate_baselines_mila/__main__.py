"""Allow `python -m generate_baselines_mila ...`."""

from __future__ import annotations

import sys

from .cli import main


if __name__ == "__main__":
    sys.exit(main())
