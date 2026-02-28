"""
ACT: Action Chunking with Transformers
Adapted from https://github.com/tonyzhaozh/act for UR5 + ROS2 bags.

Key changes from original:
  - Multi-camera support (1 or 2 cameras)
  - Configurable state/action dims
  - Cleaner forward() signature
  - HuggingFace-style from_config() constructor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import math


# ─── Vision Backbone ─────────────────────────────────────────────────────────

class ImageEncoder(nn.Module):
    """ResNet18 backbone, returns spatial feature maps."""
    def __init__(self, pretrained: bool = True, freeze_bn: bool = True):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # Remove avgpool and fc — keep spatial features
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_dim = 512  # ResNet18 final channels
        
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False
    
    def forward(self, x):
        # x: (B, 3, H, W)
        return self.backbone(x)  # (B, 512, H/32, W/32)


class SinusoidalPositionEmbedding2D(nn.Module):
    """2D sinusoidal position encoding for spatial feature maps."""
    def __init__(self, d_model: int, temperature: float = 10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        device = x.device
        
        y_embed = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
        x_embed = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)
        
        dim_t = torch.arange(C // 2, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / (C // 2))
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1).permute(2, 0, 1)  # (C, H, W)
        return pos.unsqueeze(0).expand(B, -1, -1, -1)


# ─── CVAE Encoder ─────────────────────────────────────────────────────────────

class CVAEEncoder(nn.Module):
    """
    Encodes (current_state, action_sequence) → latent z.
    Used during training only — at inference, z is sampled from N(0,1).
    """
    def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Project state and actions to d_model
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Positional encoding for action sequence
        self.pos_embed = nn.Embedding(500, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project CLS token output to latent params
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        
        nn.init.normal_(self.cls_token)
    
    def forward(self, qpos, actions):
        # qpos: (B, state_dim)
        # actions: (B, T, action_dim)
        B, T, _ = actions.shape
        
        state_emb = self.state_proj(qpos).unsqueeze(1)  # (B, 1, d_model)
        action_emb = self.action_proj(actions)           # (B, T, d_model)
        
        positions = torch.arange(T, device=actions.device)
        action_emb = action_emb + self.pos_embed(positions).unsqueeze(0)
        
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, d_model)
        seq = torch.cat([cls, state_emb, action_emb], dim=1)  # (B, 2+T, d_model)
        
        out = self.transformer(seq)  # (B, 2+T, d_model)
        cls_out = out[:, 0]          # (B, d_model)
        
        mu = self.to_mu(cls_out)
        logvar = self.to_logvar(cls_out)
        return mu, logvar
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


# ─── ACT Policy (Transformer Decoder) ────────────────────────────────────────

class ACTPolicy(nn.Module):
    """
    Full ACT model:
      - Visual encoder for each camera
      - CVAE encoder (training) / prior (inference)
      - Transformer decoder predicts action chunk
    """
    def __init__(self, config: dict):
        super().__init__()
        
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.chunk_size = config['chunk_size']
        self.latent_dim = config['latent_dim']
        self.num_cameras = config['num_cameras']
        d_model = config.get('d_model', 256)
        nhead = config.get('nhead', 8)
        num_encoder_layers = config.get('num_encoder_layers', 4)
        num_decoder_layers = config.get('num_decoder_layers', 7)
        
        # Vision
        self.image_encoder = ImageEncoder(pretrained=config.get('pretrained_vision', True))
        self.pos_embed_2d = SinusoidalPositionEmbedding2D(self.image_encoder.feature_dim)
        self.cam_embed = nn.Embedding(self.num_cameras, self.image_encoder.feature_dim)
        
        # Project image features to d_model
        self.img_proj = nn.Conv2d(self.image_encoder.feature_dim, d_model, kernel_size=1)
        
        # State projection
        self.state_proj = nn.Linear(self.state_dim, d_model)
        
        # Latent projection
        self.latent_proj = nn.Linear(self.latent_dim, d_model)
        
        # CVAE encoder (training)
        self.cvae_encoder = CVAEEncoder(
            self.state_dim, self.action_dim, self.latent_dim,
            d_model=d_model, nhead=nhead, num_layers=num_encoder_layers
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )
        
        # Action queries (learned, one per timestep in chunk)
        self.action_queries = nn.Embedding(self.chunk_size, d_model)
        
        # Output head
        self.action_head = nn.Linear(d_model, self.action_dim)
        
        # KL weight (beta)
        self.kl_weight = config.get('kl_weight', 10.0)
    
    def encode_images(self, images):
        """
        images: (B, num_cams, 3, H, W)
        Returns: (B, num_patches, d_model) sequence for transformer memory
        """
        B, N, C, H, W = images.shape
        
        # Encode each camera
        all_patches = []
        for cam_idx in range(N):
            cam_imgs = images[:, cam_idx]           # (B, 3, H, W)
            features = self.image_encoder(cam_imgs)  # (B, 512, h, w)
            
            # Add 2D positional embedding
            pos = self.pos_embed_2d(features)
            features = features + pos
            
            # Add camera embedding
            cam_emb = self.cam_embed(torch.tensor(cam_idx, device=features.device))
            features = features + cam_emb.view(1, -1, 1, 1)
            
            # Project to d_model
            features = self.img_proj(features)       # (B, d_model, h, w)
            
            # Flatten spatial dims
            h, w = features.shape[2:]
            features = features.flatten(2).permute(0, 2, 1)  # (B, h*w, d_model)
            all_patches.append(features)
        
        return torch.cat(all_patches, dim=1)  # (B, N*h*w, d_model)
    
    def forward(self, images, qpos, actions=None, is_pad=None):
        """
        Training: pass actions → compute ELBO loss
        Inference: actions=None → predict action chunk
        
        Args:
            images:  (B, num_cams, 3, H, W)
            qpos:    (B, state_dim)
            actions: (B, chunk_size, action_dim) [training only]
            is_pad:  (B, chunk_size) bool [training only]
        
        Returns:
            Training: (loss, kl_loss, reconstruction_loss)
            Inference: (B, chunk_size, action_dim)
        """
        B = images.shape[0]
        
        # Encode images → memory for transformer decoder
        img_memory = self.encode_images(images)       # (B, S, d_model)
        
        # State token
        state_token = self.state_proj(qpos).unsqueeze(1)  # (B, 1, d_model)
        
        if actions is not None:
            # ─── Training: use CVAE encoder ───
            mu, logvar = self.cvae_encoder(qpos, actions)
            z = CVAEEncoder.reparameterize(mu, logvar)
        else:
            # ─── Inference: sample from prior ───
            mu = logvar = torch.zeros(B, self.latent_dim, device=images.device)
            z = mu  # deterministic at test time (use mean)
        
        latent_token = self.latent_proj(z).unsqueeze(1)   # (B, 1, d_model)
        
        # Build memory: [state_token, latent_token, image_patches]
        memory = torch.cat([state_token, latent_token, img_memory], dim=1)
        
        # Action queries
        queries = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        
        # Decode
        decoded = self.transformer_decoder(queries, memory)  # (B, chunk_size, d_model)
        pred_actions = self.action_head(decoded)              # (B, chunk_size, action_dim)
        
        if actions is None:
            return pred_actions
        
        # ─── Loss ───
        # Reconstruction loss (L1, masked)
        l1_loss = F.l1_loss(pred_actions, actions, reduction='none')  # (B, T, action_dim)
        if is_pad is not None:
            mask = ~is_pad  # (B, T) — True where not padded
            l1_loss = l1_loss[mask.unsqueeze(-1).expand_as(l1_loss)].mean()
        else:
            l1_loss = l1_loss.mean()
        
        # KL divergence loss
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        
        total_loss = l1_loss + self.kl_weight * kl_loss
        
        return total_loss, kl_loss.detach(), l1_loss.detach()
    
    @classmethod
    def from_config(cls, config: dict) -> 'ACTPolicy':
        return cls(config)