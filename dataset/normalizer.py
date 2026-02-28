"""
dataset/normalizer.py

Computes and stores normalization statistics from the training split.
State/action dimensions for this setup:
  - arm joints: 6 (right UR5)
  - gripper: 1 (Robotiq, already normalised to [0,1])
  - total state_dim = 7 (+ 7 if use_tcp_pose, + 6 if use_force_torque)
  - action_dim = 7 always
"""
from __future__ import annotations
import json
import numpy as np
import torch
import h5py
from pathlib import Path


class JointNormalizer:
    """
    Per-dimension mean/std normalisation for qpos and actions.
    Gripper is already in [0,1] from sync_and_build, but we still
    normalise it with the rest for consistency.
    """

    def __init__(self, stats: dict):
        # stats keys: qpos_mean, qpos_std, action_mean, action_std
        self.stats = stats

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_episodes(cls, episode_paths: list[str]) -> JointNormalizer:
        all_qpos, all_actions = [], []
        for p in episode_paths:
            with h5py.File(p, 'r') as f:
                all_qpos.append(f['observations/qpos'][:])
                all_actions.append(f['action'][:])

        qpos    = np.concatenate(all_qpos,    axis=0)
        actions = np.concatenate(all_actions, axis=0)

        stats = {
            'qpos_mean':   qpos.mean(0).tolist(),
            'qpos_std':    (qpos.std(0) + 1e-6).tolist(),
            'action_mean': actions.mean(0).tolist(),
            'action_std':  (actions.std(0) + 1e-6).tolist(),
        }
        norm = cls(stats)

        print(f"Normalizer computed from {len(episode_paths)} episodes:")
        print(f"  qpos   mean: {[f'{v:.3f}' for v in stats['qpos_mean']]}")
        print(f"  qpos    std: {[f'{v:.3f}' for v in stats['qpos_std']]}")
        print(f"  action mean: {[f'{v:.3f}' for v in stats['action_mean']]}")
        print(f"  action  std: {[f'{v:.3f}' for v in stats['action_std']]}")
        return norm

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"Normalizer saved: {path}")

    @classmethod
    def load(cls, path: str) -> JointNormalizer:
        with open(path) as f:
            return cls(json.load(f))

    # ── Normalise / unnormalise ──────────────────────────────────────────────

    def _get(self, key: str, device, dtype):
        return torch.tensor(self.stats[key], dtype=dtype, device=device)

    def normalize_qpos(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._get('qpos_mean', x.device, x.dtype)
        std  = self._get('qpos_std',  x.device, x.dtype)
        return (x - mean) / std

    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._get('action_mean', x.device, x.dtype)
        std  = self._get('action_std',  x.device, x.dtype)
        return (x - mean) / std

    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        mean = self._get('action_mean', x.device, x.dtype)
        std  = self._get('action_std',  x.device, x.dtype)
        return x * std + mean

    @property
    def state_dim(self) -> int:
        return len(self.stats['qpos_mean'])

    @property
    def action_dim(self) -> int:
        return len(self.stats['action_mean'])