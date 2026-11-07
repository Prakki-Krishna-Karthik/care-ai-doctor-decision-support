"""Convenience wrapper for the backend baseline trainer."""

from __future__ import annotations

import sys
from pathlib import Path

backend = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(backend))

from train import main  # noqa: E402


if __name__ == "__main__":
    main()
