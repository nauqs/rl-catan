"""Keep macOS awake for the lifetime of the current process.

Long unattended runs (training, the Phase 4 ablation) must not be paused by the
laptop idle-sleeping — when that happened mid-run, SB3's cumulative `fps` was
permanently dragged down (the sleep hours stay in the denominator) and, more
importantly, real calendar time was lost while the process was suspended.

`keep_awake()` spawns `caffeinate -w <our-pid>` so the assertion is released
automatically when this process exits — no cleanup needed, no effect if the
process is killed. No-op off macOS or if `caffeinate` is missing.
"""
from __future__ import annotations

import os
import platform
import subprocess


def keep_awake() -> subprocess.Popen | None:
    """Assert no idle/system/disk sleep until this process exits. Returns the
    caffeinate Popen (or None if unavailable)."""
    if platform.system() != "Darwin":
        return None
    try:
        # -i no idle sleep, -m no disk sleep, -s no system sleep (AC),
        # -w wait on our pid so caffeinate dies with us.
        return subprocess.Popen(
            ["caffeinate", "-i", "-m", "-s", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        return None
