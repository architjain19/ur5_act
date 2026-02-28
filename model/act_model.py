"""
model/act_model.py

Action Chunking with Transformers (ACT) — adapted for UR5.
Based on: https://github.com/tonyzhaozh/act

Architecture:
  ┌─────────────────────────────────────────┐
  │  TRAINING                               │
  │  [qpos, action_chunk] → CVAE encoder   │
  │                         → mu, logvar   │
  │                         → z (sampled)  │
  └─────────────────────────────────────────┘
  ┌─────────────────────────────────────────┐
  │  BOTH                                   │
  │  [image(s)] → ResNet18 → patches        │
  │  [qpos]     → Linear   → state token   │
  │  [z]        → Linear   → latent token  │
  │                                         │
  │  memory = [state, latent, img_patches]  │
  │  queries = learned (chunk_size,)        │
  │                                         │
  │  Transformer decoder → action_head      │
  │                      → (chunk, action_dim)│
  └─────────────────────────────────────────┘
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


# ── Vision backbone ──────────────────────────────────────────────────────────

class ResNetBackbone(nn.Module):
    """ResNet-18 without the final pooling/fc — returns (B, 512, h, w)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        # Strip avgpool and fc
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.out_channels = 512

        # Freeze BatchNorm running stats (recommended for fine-tuning)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)   # (B, 512, H/32, W/32)


class PositionEmbedding2D(nn.Module):
    """Sinusoidal 2D PE added to ResNet feature maps.
    
    Allocates C//2 dims to y-axis and C//2 dims to x-axis.
    Each axis uses C//4 sin + C//4 cos frequencies.
    Works for any spatial size (H, W) and any even C.
    """

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sinusoidal PE"
        self.d_model = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dev = x.device
        # Each axis gets C//2 channels; within that, C//4 frequencies
        n_freq = C // 4

        dim_t = torch.arange(n_freq, device=dev, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / n_freq)   # (n_freq,)

        # y positions: (H,)  /  x positions: (W,)
        y_pos = torch.arange(H, device=dev, dtype=torch.float32)  # (H,)
        x_pos = torch.arange(W, device=dev, dtype=torch.float32)  # (W,)

        # Outer product with frequencies → (H, n_freq) and (W, n_freq)
        y_ang = y_pos[:, None] / dim_t[None, :]   # (H, n_freq)
        x_ang = x_pos[:, None] / dim_t[None, :]   # (W, n_freq)

        # Interleave sin/cos → (H, C//2) and (W, C//2)
        y_emb = torch.stack([y_ang.sin(), y_ang.cos()], dim=-1).flatten(-2)  # (H, C//2)
        x_emb = torch.stack([x_ang.sin(), x_ang.cos()], dim=-1).flatten(-2)  # (W, C//2)

        # Broadcast to (H, W, C//2) each, then concat → (H, W, C)
        y_emb = y_emb[:, None, :].expand(H, W, C // 2)   # (H, W, C//2)
        x_emb = x_emb[None, :, :].expand(H, W, C // 2)   # (H, W, C//2)
        pe = torch.cat([y_emb, x_emb], dim=-1)            # (H, W, C)

        pe = pe.permute(2, 0, 1).unsqueeze(0)             # (1, C, H, W)
        return x + pe


# ── CVAE encoder ─────────────────────────────────────────────────────────────

class CVAEEncoder(nn.Module):
    """
    Encodes (current_qpos, target_action_chunk) → Gaussian latent z.
    Only used during training. At inference, z ~ N(0,I).
    """

    def __init__(self, state_dim: int, action_dim: int, latent_dim: int,
                 d_model: int = 256, nhead: int = 8, num_layers: int = 4,
                 chunk_size: int = 40):
        super().__init__()
        self.latent_dim = latent_dim

        self.state_proj  = nn.Linear(state_dim,  d_model)
        self.action_proj = nn.Linear(action_dim, d_model)
        self.pos_embed   = nn.Embedding(chunk_size + 2, d_model)  # +2 for cls+state tokens

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True,   # pre-norm is more stable
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.to_mu     = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

    def forward(self, qpos: torch.Tensor, actions: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        qpos:    (B, state_dim)
        actions: (B, T, action_dim)
        → mu, logvar  each (B, latent_dim)
        """
        B, T, _ = actions.shape
        state_tok  = self.state_proj(qpos).unsqueeze(1)   # (B, 1, d)
        action_tok = self.action_proj(actions)              # (B, T, d)
        cls        = self.cls_token.expand(B, -1, -1)      # (B, 1, d)

        seq = torch.cat([cls, state_tok, action_tok], dim=1)  # (B, 2+T, d)
        positions = torch.arange(seq.size(1), device=qpos.device)
        seq = seq + self.pos_embed(positions).unsqueeze(0)

        out     = self.norm(self.transformer(seq))
        cls_out = out[:, 0]   # (B, d) — CLS token

        return self.to_mu(cls_out), self.to_logvar(cls_out)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)


# ── Full ACT policy ──────────────────────────────────────────────────────────

class ACTPolicy(nn.Module):

    def __init__(self, cfg: dict):
        super().__init__()
        state_dim  = cfg['state_dim']    # 7 (arm6 + gripper1)
        action_dim = cfg['action_dim']   # 7
        chunk_size = cfg['chunk_size']   # 40
        latent_dim = cfg['latent_dim']   # 32
        d_model    = cfg.get('d_model', 256)
        nhead      = cfg.get('nhead', 8)
        n_enc      = cfg.get('num_encoder_layers', 4)
        n_dec      = cfg.get('num_decoder_layers', 7)
        pretrained = cfg.get('pretrained_vision', True)

        self.chunk_size = chunk_size
        self.latent_dim = latent_dim
        self.kl_weight  = cfg.get('kl_weight', 10.0)

        # Vision
        self.backbone = ResNetBackbone(pretrained=pretrained)
        self.pe_2d    = PositionEmbedding2D(self.backbone.out_channels)
        self.img_proj = nn.Conv2d(self.backbone.out_channels, d_model, 1)

        # State / latent projections
        self.state_proj   = nn.Linear(state_dim,  d_model)
        self.latent_proj  = nn.Linear(latent_dim, d_model)

        # CVAE encoder
        self.cvae_enc = CVAEEncoder(state_dim, action_dim, latent_dim,
                                     d_model, nhead, n_enc, chunk_size)

        # Transformer decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=0.1,
            batch_first=True, norm_first=True,
        )
        self.transformer_dec = nn.TransformerDecoder(dec_layer, num_layers=n_dec)
        self.dec_norm = nn.LayerNorm(d_model)

        # Learned action queries (one per chunk step)
        self.action_queries = nn.Embedding(chunk_size, d_model)

        # Output head
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Image encoding ───────────────────────────────────────────────────────

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, num_cams, 3, H, W)
        → memory: (B, num_cams * h * w, d_model)
        """
        B, N, C, H, W = images.shape
        patches_list = []

        for cam in range(N):
            feat = self.backbone(images[:, cam])     # (B, 512, h, w)
            feat = self.pe_2d(feat)                  # add 2D PE
            feat = self.img_proj(feat)               # (B, d_model, h, w)
            h, w = feat.shape[2:]
            feat = feat.flatten(2).permute(0, 2, 1) # (B, h*w, d_model)
            patches_list.append(feat)

        return torch.cat(patches_list, dim=1)        # (B, N*h*w, d_model)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        images:  torch.Tensor,                    # (B, 1, 3, H, W)
        qpos:    torch.Tensor,                    # (B, state_dim)
        actions: torch.Tensor | None = None,      # (B, chunk_size, action_dim) training only
        is_pad:  torch.Tensor | None = None,      # (B, chunk_size) bool training only
    ):
        B = images.shape[0]

        # Encode images → memory
        img_mem = self.encode_images(images)          # (B, S, d_model)

        # State token
        state_tok = self.state_proj(qpos).unsqueeze(1)   # (B, 1, d_model)

        if actions is not None:
            # Training: encode (qpos, actions) → latent z
            mu, logvar = self.cvae_enc(qpos, actions)
            z = CVAEEncoder.reparameterize(mu, logvar)
        else:
            # Inference: z = 0 (mean of prior N(0,I))
            mu = logvar = torch.zeros(B, self.latent_dim, device=images.device)
            z = mu

        latent_tok = self.latent_proj(z).unsqueeze(1)    # (B, 1, d_model)

        # Memory = state + latent + image patches
        memory = torch.cat([state_tok, latent_tok, img_mem], dim=1)  # (B, 2+S, d_model)

        # Action queries → decoder
        queries  = self.action_queries.weight.unsqueeze(0).expand(B, -1, -1)
        decoded  = self.dec_norm(self.transformer_dec(queries, memory))  # (B, T, d_model)
        pred_act = self.action_head(decoded)              # (B, T, action_dim)

        if actions is None:
            return pred_act   # inference path

        # ── Loss computation ─────────────────────────────────────────────────
        # L1 reconstruction loss (masked over padded steps)
        l1 = F.l1_loss(pred_act, actions, reduction='none')  # (B, T, action_dim)
        if is_pad is not None:
            # is_pad: True where padded — mask those out
            mask = (~is_pad).float().unsqueeze(-1)            # (B, T, 1)
            l1   = (l1 * mask).sum() / (mask.sum() * l1.shape[-1] + 1e-8)
        else:
            l1 = l1.mean()

        # KL divergence: D_KL(q(z|x) || N(0,I))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        loss = l1 + self.kl_weight * kl
        return loss, kl.detach(), l1.detach()

    @classmethod
    def from_config(cls, cfg: dict) -> ACTPolicy:
        return cls(cfg)