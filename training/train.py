#!/usr/bin/env python3
"""
training/train.py

Main ACT training loop for UR5 pick-and-place.

Usage:
  python training/train.py
  python training/train.py --config config/task_config.yaml --extracted_dir data/extracted
  python training/train.py --resume checkpoints/run_xyz/epoch_0499.pth
"""
from __future__ import annotations
import argparse
import json
import time
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.ur5_dataset import make_dataloaders
from model.act_model import ACTPolicy


# ── Helpers ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Epoch functions ──────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, train: bool, grad_clip: float = 1.0):
    model.train(train)
    total = {'loss': 0.0, 'kl': 0.0, 'l1': 0.0}
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            images  = batch['images'].to(device, non_blocking=True)
            qpos    = batch['qpos'].to(device,   non_blocking=True)
            actions = batch['actions'].to(device, non_blocking=True)
            is_pad  = batch['is_pad'].to(device,  non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            loss, kl, l1 = model(images, qpos, actions, is_pad)

            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            bs = images.size(0)
            total['loss'] += loss.item() * bs
            total['kl']   += kl.item()   * bs
            total['l1']   += l1.item()   * bs
            n += bs

    return {k: v / n for k, v in total.items()}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',        default='config/task_config.yaml')
    parser.add_argument('--extracted_dir', default='data/extracted')
    parser.add_argument('--ckpt_dir',      default='checkpoints')
    parser.add_argument('--resume',        default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    train_cfg = config['training']
    set_seed(train_cfg.get('seed', 42))

    # Check CUDA capability — PyTorch >=2.0 requires sm_70+.
    # The MX330 (sm_61) and other Pascal-era GPUs will segfault or error.
    _use_cuda = False
    if torch.cuda.is_available():
        cap_major, _ = torch.cuda.get_device_capability(0)
        if cap_major >= 7:
            _use_cuda = True
        else:
            print(f"WARNING: GPU {torch.cuda.get_device_name(0)} has CUDA capability "
                  f"sm_{cap_major}x which is below the sm_70 minimum for this "
                  f"PyTorch build. Falling back to CPU.")
            print("   To use this GPU install PyTorch 1.x or upgrade to an sm_70+ card.")

    device = torch.device('cuda' if _use_cuda else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, normalizer = make_dataloaders(
        config,
        args.extracted_dir,
        val_split=train_cfg.get('val_split', 0.1),
        seed=train_cfg.get('seed', 42),
    )

    # ── Model config ───────────────────────────────────────────────────────
    # state_dim is read from the first episode HDF5 (set by extraction)
    from pathlib import Path as _P
    import h5py
    ep0 = sorted(_P(args.extracted_dir).glob('episode_*.hdf5'))[0]
    with h5py.File(ep0, 'r') as f:
        state_dim  = int(f.attrs['state_dim'])
        action_dim = int(f.attrs['action_dim'])

    model_cfg = {
        'state_dim':           state_dim,
        'action_dim':          action_dim,
        'chunk_size':          config['chunk_size'],
        'latent_dim':          config.get('latent_dim', 32),
        'd_model':             config.get('d_model', 256),
        'nhead':               config.get('nhead', 8),
        'num_encoder_layers':  config.get('num_encoder_layers', 4),
        'num_decoder_layers':  config.get('num_decoder_layers', 7),
        'kl_weight':           config.get('kl_weight', 10.0),
        'pretrained_vision':   config.get('pretrained_vision', True),
    }

    model = ACTPolicy.from_config(model_cfg).to(device)
    print(f"Model parameters: {count_params(model):,}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    n_epochs   = train_cfg['epochs']
    lr         = train_cfg['lr']
    wd         = train_cfg['weight_decay']
    grad_clip  = train_cfg.get('grad_clip', 1.0)
    save_every = train_cfg.get('save_every', 100)

    optimizer  = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler  = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 0.01)

    # ── Checkpoint dir ─────────────────────────────────────────────────────
    run_name  = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir  = Path(args.ckpt_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save configs next to checkpoints for reproducibility
    with open(ckpt_dir / 'task_config.yaml', 'w') as f:
        yaml.dump(config, f)
    with open(ckpt_dir / 'model_config.json', 'w') as f:
        json.dump(model_cfg, f, indent=2)
    normalizer.save(str(ckpt_dir / 'normalizer_stats.json'))

    writer = SummaryWriter(log_dir=str(ckpt_dir / 'tb'))

    print(f"\nCheckpoints: {ckpt_dir}")
    print(f"TensorBoard: tensorboard --logdir {ckpt_dir / 'tb'}\n")

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch   = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch - 1}")

    # ── Training loop ──────────────────────────────────────────────────────
    history = []

    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()

        tr = run_epoch(model, train_loader, optimizer, device, train=True,  grad_clip=grad_clip)
        va = run_epoch(model, val_loader,   optimizer, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        # Log to TensorBoard
        writer.add_scalar('Loss/train',    tr['loss'], epoch)
        writer.add_scalar('Loss/val',      va['loss'], epoch)
        writer.add_scalar('L1/train',      tr['l1'],   epoch)
        writer.add_scalar('L1/val',        va['l1'],   epoch)
        writer.add_scalar('KL/train',      tr['kl'],   epoch)
        writer.add_scalar('LR',            lr_now,     epoch)

        # Console
        print(
            f"Ep {epoch:4d}/{n_epochs}  "
            f"Train L={tr['loss']:.4f} (L1={tr['l1']:.4f} KL={tr['kl']:.4f})  "
            f"Val L={va['loss']:.4f} (L1={va['l1']:.4f})  "
            f"LR={lr_now:.1e}  {elapsed:.0f}s"
        )

        # History
        history.append({**{f'train_{k}': v for k, v in tr.items()},
                         **{f'val_{k}':   v for k, v in va.items()},
                         'epoch': epoch, 'lr': lr_now})

        # Best checkpoint
        if va['loss'] < best_val_loss:
            best_val_loss = va['loss']
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_config': model_cfg, 'best_val_loss': best_val_loss,
            }, ckpt_dir / 'best.pth')
            print(f"  [BEST] New best val loss: {best_val_loss:.4f}")

        # Periodic checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_config': model_cfg, 'best_val_loss': best_val_loss,
            }, ckpt_dir / f'epoch_{epoch:04d}.pth')

        # Save history every epoch
        with open(ckpt_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    writer.close()
    print(f"\nTraining complete.")
    print(f"Best val loss : {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == '__main__':
    main()