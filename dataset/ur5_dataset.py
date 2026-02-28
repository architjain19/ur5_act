import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from torchvision import transforms
from .normalizer import JointNormalizer


class UR5ACTDataset(Dataset):
    """
    Dataset for ACT training on UR5 teleoperation data.
    
    Each sample:
      - obs_image: (num_cameras, 3, H, W) float32 in [0,1]
      - obs_qpos:  (state_dim,) normalized joint positions
      - actions:   (chunk_size, action_dim) normalized target joints
    """
    def __init__(
        self,
        episode_paths: list,
        chunk_size: int = 100,
        normalizer: JointNormalizer = None,
        use_wrist_cam: bool = False,
        image_size: tuple = (224, 224),
        augment: bool = True,
    ):
        self.episode_paths = episode_paths
        self.chunk_size = chunk_size
        self.normalizer = normalizer
        self.use_wrist_cam = use_wrist_cam
        self.image_size = image_size
        
        # Build index: (episode_idx, start_timestep)
        self.index = []
        self.episode_lengths = []
        
        for ep_idx, path in enumerate(episode_paths):
            with h5py.File(path, 'r') as f:
                T = f['observations/qpos'].shape[0]
                self.episode_lengths.append(T)
                # Sample a starting point for every timestep
                # (ACT samples uniformly over valid start positions)
                for t in range(T):
                    self.index.append((ep_idx, t))
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),          # HWC uint8 -> CHW float [0,1]
            transforms.Resize(image_size),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ] + ([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)),
        ] if augment else []))
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        path = self.episode_paths[ep_idx]
        T = self.episode_lengths[ep_idx]
        
        with h5py.File(path, 'r') as f:
            # Current observation
            qpos = f['observations/qpos'][t]  # (state_dim,)
            image = f['observations/images/main'][t]  # (H, W, 3) uint8
            
            wrist_image = None
            if self.use_wrist_cam and 'observations/images/wrist' in f:
                wrist_image = f['observations/images/wrist'][t]
            
            # Action chunk: t to t+chunk_size, pad at end of episode
            t_end = min(t + self.chunk_size, T)
            actions = f['action'][t:t_end]          # (actual_len, action_dim)
            actual_len = actions.shape[0]
        
        # Pad actions if near episode end
        if actual_len < self.chunk_size:
            pad = np.repeat(actions[-1:], self.chunk_size - actual_len, axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        
        # Convert to tensors
        qpos = torch.from_numpy(qpos).float()
        actions = torch.from_numpy(actions).float()
        
        # Normalize
        if self.normalizer:
            qpos = self.normalizer.normalize_qpos(qpos)
            actions = self.normalizer.normalize_action(actions)
        
        # Process images
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray(image)
        img_tensor = self.img_transform(img_pil)  # (3, H, W)
        
        if wrist_image is not None:
            wrist_pil = PILImage.fromarray(wrist_image)
            wrist_tensor = self.img_transform(wrist_pil)
            images = torch.stack([img_tensor, wrist_tensor], dim=0)  # (2, 3, H, W)
        else:
            images = img_tensor.unsqueeze(0)  # (1, 3, H, W)
        
        # is_pad: 1 where padded (for loss masking)
        is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        is_pad[actual_len:] = True
        
        return {
            'images': images,       # (num_cams, 3, H, W)
            'qpos': qpos,           # (state_dim,)
            'actions': actions,     # (chunk_size, action_dim)
            'is_pad': is_pad,       # (chunk_size,) bool
        }


def make_dataloaders(config: dict, extracted_dir: str, val_split: float = 0.1):
    """Create train/val dataloaders."""
    from torch.utils.data import DataLoader
    import random
    
    all_episodes = sorted(Path(extracted_dir).glob('episode_*.hdf5'))
    assert len(all_episodes) > 0, f"No episodes found in {extracted_dir}"
    
    random.shuffle(all_episodes)
    n_val = max(1, int(len(all_episodes) * val_split))
    val_eps = all_episodes[:n_val]
    train_eps = all_episodes[n_val:]
    
    print(f"Train: {len(train_eps)} episodes, Val: {len(val_eps)} episodes")
    
    # Compute normalization stats from training episodes only
    normalizer = JointNormalizer.from_dataset([str(p) for p in train_eps])
    normalizer.save('data/normalizer_stats.json')
    
    chunk_size = config['chunk_size']
    image_size = tuple(config['image_size'])
    use_wrist = config['topics'].get('camera_wrist') is not None
    
    train_dataset = UR5ACTDataset(
        [str(p) for p in train_eps],
        chunk_size=chunk_size,
        normalizer=normalizer,
        use_wrist_cam=use_wrist,
        image_size=image_size,
        augment=True,
    )
    val_dataset = UR5ACTDataset(
        [str(p) for p in val_eps],
        chunk_size=chunk_size,
        normalizer=normalizer,
        use_wrist_cam=use_wrist,
        image_size=image_size,
        augment=False,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, val_loader, normalizer