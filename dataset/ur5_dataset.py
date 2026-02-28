"""
dataset/ur5_dataset.py

PyTorch Dataset for ACT training on extracted UR5 HDF5 episodes.

Sample format:
  images  : (1, 3, H, W)  float32  ImageNet-normalised
  qpos    : (state_dim,)  float32  normalised joint state
  actions : (chunk_size, action_dim)  float32  normalised action chunk
  is_pad  : (chunk_size,)  bool  True for padded steps at end of episode
"""
from __future__ import annotations
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image as PILImage
import h5py
from pathlib import Path

from .normalizer import JointNormalizer


class UR5ACTDataset(Dataset):

    def __init__(
        self,
        episode_paths: list[str],
        chunk_size: int,
        normalizer: JointNormalizer,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = True,
    ):
        self.paths      = episode_paths
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.image_size = image_size

        # Build flat index: (episode_idx, start_timestep)
        self.index = []
        self.ep_lengths = []
        for ep_idx, p in enumerate(self.paths):
            with h5py.File(p, 'r') as f:
                T = int(f.attrs['episode_len'])
            self.ep_lengths.append(T)
            for t in range(T):
                self.index.append((ep_idx, t))

        # Image transforms
        base = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ]
        aug = [
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                    saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03),
                                     scale=(0.97, 1.03)),
        ] if augment else []
        # Apply aug before normalising
        self.img_transform = transforms.Compose(aug + base)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, t = self.index[idx]
        T = self.ep_lengths[ep_idx]

        with h5py.File(self.paths[ep_idx], 'r') as f:
            qpos   = f['observations/qpos'][t].astype(np.float32)   # (state_dim,)
            image  = f['observations/images/main'][t]                # (H, W, 3) uint8

            # Action chunk [t, t+chunk_size)
            t_end = min(t + self.chunk_size, T)
            chunk = f['action'][t:t_end].astype(np.float32)          # (actual_len, action_dim)

        actual_len = chunk.shape[0]

        # Pad chunk to fixed size if near end of episode
        if actual_len < self.chunk_size:
            pad = np.tile(chunk[-1:], (self.chunk_size - actual_len, 1))
            chunk = np.concatenate([chunk, pad], axis=0)

        # to tensors
        qpos   = torch.from_numpy(qpos)
        chunk  = torch.from_numpy(chunk)

        # normalise
        qpos  = self.normalizer.normalize_qpos(qpos)
        chunk = self.normalizer.normalize_action(chunk)

        # image
        img_pil    = PILImage.fromarray(image)
        img_tensor = self.img_transform(img_pil)      # (3, H, W)
        images     = img_tensor.unsqueeze(0)           # (1, 3, H, W)

        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        is_pad[actual_len:] = True

        return {
            'images':  images,   # (1, 3, H, W)
            'qpos':    qpos,     # (state_dim,)
            'actions': chunk,    # (chunk_size, action_dim)
            'is_pad':  is_pad,   # (chunk_size,)
        }


# ── Factory ──────────────────────────────────────────────────────────────────

def make_dataloaders(
    config: dict,
    extracted_dir: str,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, JointNormalizer]:

    all_eps = sorted(Path(extracted_dir).glob('episode_*.hdf5'))
    assert len(all_eps) > 0, f"No episodes found in {extracted_dir}"

    random.seed(seed)
    eps = list(all_eps)
    random.shuffle(eps)

    n_val   = max(1, int(len(eps) * val_split))
    val_eps  = eps[:n_val]
    train_eps = eps[n_val:]

    print(f"\nDataset split:  {len(train_eps)} train / {len(val_eps)} val")

    # Compute normalizer from training episodes only
    normalizer = JointNormalizer.from_episodes([str(p) for p in train_eps])
    normalizer.save('data/normalizer_stats.json')

    chunk_size = config['chunk_size']
    image_size = tuple(config['image_size'])
    batch_size = config['training']['batch_size']

    train_ds = UR5ACTDataset([str(p) for p in train_eps], chunk_size,
                              normalizer, image_size, augment=True)
    val_ds   = UR5ACTDataset([str(p) for p in val_eps],   chunk_size,
                              normalizer, image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True, drop_last=True,
                               persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True,
                               persistent_workers=True)

    return train_loader, val_loader, normalizer