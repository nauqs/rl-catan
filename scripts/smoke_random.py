"""End-to-end smoke check: random policy plays one 1v1 and one 4-player game."""
from __future__ import annotations

import gymnasium as gym
import numpy as np

import catanatron.gym  # noqa: F401  -- registers "catanatron/Catanatron-v0"
from catanatron import Color, RandomPlayer


def play_one(env: gym.Env, seed: int) -> tuple[float, int]:
    obs, info = env.reset(seed=seed)
    steps, total_reward = 0, 0.0
    rng = np.random.default_rng(seed)
    while True:
        valid = info["valid_actions"]
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            return total_reward, steps


def main() -> None:
    env_1v1 = gym.make("catanatron/Catanatron-v0")
    r, steps = play_one(env_1v1, seed=0)
    print(f"[1v1]  reward={r:+.1f}  steps={steps}")
    env_1v1.close()

    env_4p = gym.make(
        "catanatron/Catanatron-v0",
        config={
            "enemies": [
                RandomPlayer(Color.RED),
                RandomPlayer(Color.WHITE),
                RandomPlayer(Color.ORANGE),
            ]
        },
    )
    r, steps = play_one(env_4p, seed=0)
    print(f"[4p]   reward={r:+.1f}  steps={steps}")
    env_4p.close()


if __name__ == "__main__":
    main()
