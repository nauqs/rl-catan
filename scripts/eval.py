"""Thin wrapper so `uv run python scripts/eval.py ...` works too."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catan_rl.eval.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
