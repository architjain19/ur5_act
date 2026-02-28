#!/usr/bin/env python3
"""
Main training script for UR5 ACT policy.
Usage: python training/train.py --config config/task_config.yaml
"""
import argparse
import yaml
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import time
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.ur5_dataset import make_dataloaders
from model.act_model import ACTPolicy


def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    model.train()
    total_loss = total_kl = total_l1 = 0
    n_batches = 0
    
    for batch in loader:
        images = batch['images'].to(device)       # (B, cams, 3, H, W)
        qpos = batch['qpos'].to(device)           # (B, state_dim)
        actions = batch['actions'].to(device)     # (B, T, action_dim)
        is_pad = batch['is_pad'].to(device)       # (B, T)
        
        optimizer.zero_grad()
        loss, kl_loss, l1_loss = model(images, qpos, actions, is_pad)
        loss.backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_kl += kl_loss.item()
        total_l1 += l1_loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'kl_loss': total_kl / n_batches,
        'l1_loss': total_l1 / n_batches,
    }


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_loss = total_kl = total_l1 = 0
    n_batches = 0
    
    for batch in loader:
        images = batch['images'].to(device)
        qpos = batch['qpos'].to(device)
        actions = batch['actions'].to(device)
        is_pad = batch['is_pad'].to(device)
        
        loss, kl_loss, l1_loss = model(images, qpos, actions, is_pad)
        
        total_loss += loss.item()
        total_kl += kl_loss.item()
        total_l1 += l1_loss.item()
        n_batches += 1
    
    return {
        'loss': total_loss / n_batches,
        'kl_loss': total_kl / n_batches,
        'l1_loss': total_l1 / n_batches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/task_config.yaml')
    parser.add_argument('--extracted_dir', default='data/extracted')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # ─── Training hyperparameters ───────────────────────────────────────────
    train_config = config.get('training', {})
    n_epochs      = train_config.get('epochs', 2000)
    lr            = train_config.get('lr', 1e-4)
    weight_decay  = train_config.get('weight_decay', 1e-4)
    grad_clip     = train_config.get('grad_clip', 1.0)
    save_every    = train_config.get('save_every', 100)
    batch_size    = train_config.get('batch_size', 8)
    
    config['batch_size'] = batch_size
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # ─── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, normalizer = make_dataloaders(
        config, args.extracted_dir, val_split=0.1
    )
    
    # ─── Model config ──────────────────────────────────────────────────────
    use_wrist = config['topics'].get('camera_wrist') is not None
    state_dim = len(config['joint_names']) + 1   # arm joints + gripper
    action_dim = state_dim
    
    model_config = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'chunk_size': config['chunk_size'],
        'latent_dim': config.get('latent_dim', 32),
        'num_cameras': 2 if use_wrist else 1,
        'd_model': config.get('d_model', 256),
        'nhead': config.get('nhead', 8),
        'num_encoder_layers': config.get('num_encoder_layers', 4),
        'num_decoder_layers': config.get('num_decoder_layers', 7),
        'kl_weight': config.get('kl_weight', 10.0),
        'pretrained_vision': True,
    }
    
    model = ACTPolicy.from_config(model_config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    
    # ─── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # ─── Save dir ──────────────────────────────────────────────────────────
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configs alongside checkpoints
    with open(ckpt_dir / 'task_config.yaml', 'w') as f:
        yaml.dump(config, f)
    with open(ckpt_dir / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    normalizer.save(str(ckpt_dir / 'normalizer_stats.json'))
    
    print(f"\nCheckpoints → {ckpt_dir}")
    print(f"Training for {n_epochs} epochs\n")
    
    # ─── Training loop ─────────────────────────────────────────────────────
    history = []
    
    for epoch in range(start_epoch, n_epochs):
        t0 = time.time()
        
        train_metrics = train_epoch(model, train_loader, optimizer, device, grad_clip)
        val_metrics = val_epoch(model, val_loader, device)
        scheduler.step()
        
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']
        
        log = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_kl': train_metrics['kl_loss'],
            'train_l1': train_metrics['l1_loss'],
            'val_loss': val_metrics['loss'],
            'val_l1': val_metrics['l1_loss'],
            'lr': lr_now,
            'elapsed': elapsed,
        }
        history.append(log)
        
        print(
            f"Epoch {epoch:4d}/{n_epochs} | "
            f"Train L={train_metrics['loss']:.4f} "
            f"(L1={train_metrics['l1_loss']:.4f}, KL={train_metrics['kl_loss']:.4f}) | "
            f"Val L={val_metrics['loss']:.4f} (L1={val_metrics['l1_loss']:.4f}) | "
            f"LR={lr_now:.2e} | {elapsed:.1f}s"
        )
        
        # Save best
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_config': model_config,
                'best_val_loss': best_val_loss,
            }, ckpt_dir / 'best.pth')
            print(f"  ✓ New best val loss: {best_val_loss:.4f}")
        
        # Save periodic
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'model_config': model_config,
                'best_val_loss': best_val_loss,
            }, ckpt_dir / f'epoch_{epoch:04d}.pth')
        
        # Save history
        with open(ckpt_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == '__main__':
    main()