"""Thin wrapper so `uv run python scripts/eval.py ...` works too."""
from catan_rl.eval.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
