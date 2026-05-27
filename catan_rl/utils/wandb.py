"""Thin W&B init helper."""
from __future__ import annotations

import os
from typing import Any


def init_wandb(*, project: str, mode: str = "online", config: dict[str, Any] | None = None):
    """Create a wandb run if mode != 'disabled'."""
    if mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=project,
        entity=os.environ.get("WANDB_ENTITY"),
        mode=mode,
        config=config or {},
    )
