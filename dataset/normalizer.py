import numpy as np
import torch
import json
from pathlib import Path
import h5py


class JointNormalizer:
    """
    Normalize joint positions and actions to [-1, 1].
    Computed from training data statistics.
    """
    def __init__(self, stats: dict = None):
        self.stats = stats  # {'qpos_mean', 'qpos_std', 'action_mean', 'action_std'}
    
    @classmethod
    def from_dataset(cls, episode_paths: list) -> 'JointNormalizer':
        """Compute normalization stats from all training episodes."""
        all_qpos = []
        all_actions = []
        
        for path in episode_paths:
            with h5py.File(path, 'r') as f:
                all_qpos.append(f['observations/qpos'][:])
                all_actions.append(f['action'][:])
        
        all_qpos = np.concatenate(all_qpos, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        stats = {
            'qpos_mean': all_qpos.mean(0).tolist(),
            'qpos_std': (all_qpos.std(0) + 1e-6).tolist(),
            'action_mean': all_actions.mean(0).tolist(),
            'action_std': (all_actions.std(0) + 1e-6).tolist(),
        }
        return cls(stats)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'JointNormalizer':
        with open(path) as f:
            stats = json.load(f)
        return cls(stats)
    
    def normalize_qpos(self, qpos: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.stats['qpos_mean'], dtype=qpos.dtype, device=qpos.device)
        std = torch.tensor(self.stats['qpos_std'], dtype=qpos.dtype, device=qpos.device)
        return (qpos - mean) / std
    
    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.stats['action_mean'], dtype=action.dtype, device=action.device)
        std = torch.tensor(self.stats['action_std'], dtype=action.dtype, device=action.device)
        return (action - mean) / std
    
    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.stats['action_mean'], dtype=action.dtype, device=action.device)
        std = torch.tensor(self.stats['action_std'], dtype=action.dtype, device=action.device)
        return action * std + mean