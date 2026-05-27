"""Cast Dict observations to float32 so MPS (which doesn't support float64) is happy.

Catanatron's mixed-representation env returns float64 arrays even though its
observation space declares float32; this wrapper closes the gap.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np


def _cast(obs):
    if isinstance(obs, dict):
        return {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
    return np.asarray(obs, dtype=np.float32)


class CastDtypeWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return _cast(observation)
