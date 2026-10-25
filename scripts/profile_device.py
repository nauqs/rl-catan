"""Compare PPO training throughput (SPS) on CPU vs MPS.

Runs `n_steps` of `model.learn()` on each backend (with W&B disabled), reports
mean fps, and prints the recommended `train.device` for `configs/ppo_baseline.yaml`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402

from catan_rl.features.board_cnn import BoardCNN  # noqa: E402
from catan_rl.features.hex_encoder import HexEncoder  # noqa: E402
from catan_rl.training.env_factory import make_env  # noqa: E402


def _make_venv(n_envs: int, enable_maritime: bool) -> SubprocVecEnv:
    def _factory(rank: int):
        def _thunk():
            return make_env(
                seed=rank, representation="mixed", opponent="weighted",
                enable_maritime_trade=enable_maritime,
            )
        return _thunk
    return SubprocVecEnv([_factory(i) for i in range(n_envs)])


def _profile_one(device: str, n_envs: int, total_timesteps: int, policy: str) -> float:
    enable_maritime = policy == "spatial"
    venv = _make_venv(n_envs, enable_maritime)
    if policy == "spatial":
        from catanatron.models.player import Color
        from catan_rl.policies.spatial import SpatialMaskableActorCriticPolicy
        policy_cls = SpatialMaskableActorCriticPolicy
        policy_kwargs = dict(
            features_extractor_class=HexEncoder,
            player_colors=(Color.BLUE, Color.RED),
            map_type="BASE",
            enable_maritime=True,
        )
    else:
        policy_cls = "MultiInputPolicy"
        policy_kwargs = dict(
            features_extractor_class=BoardCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
    model = MaskablePPO(
        policy_cls,
        venv,
        n_steps=2048,
        batch_size=512,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )
    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    dt = time.perf_counter() - t0
    venv.close()
    return total_timesteps / dt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=20_000)
    p.add_argument("--policy", choices=["flat", "spatial"], default="flat",
                   help="Which policy to benchmark (flat=BoardCNN, spatial=HexEncoder+pointer head).")
    p.add_argument("--devices", nargs="*", default=None,
                   help="Devices to test. Defaults to cpu+mps when MPS is available, else cpu only.")
    args = p.parse_args(argv)

    devices: Iterable[str]
    if args.devices is None:
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
    else:
        devices = args.devices

    results: dict[str, float] = {}
    for d in devices:
        print(f"[profile] policy={args.policy} device={d} n_envs={args.n_envs} timesteps={args.timesteps}")
        fps = _profile_one(d, args.n_envs, args.timesteps, args.policy)
        results[d] = fps
        print(f"[profile] device={d} -> {fps:,.0f} steps/sec")

    print("\n=== summary ===")
    for d, f in results.items():
        print(f"  {d:<4} : {f:,.0f} sps")
    winner = max(results, key=results.get)
    print(f"\nrecommended train.device: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
