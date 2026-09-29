from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel1ch(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False),
            nn.InstanceNorm2d(out_c, affine=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm at every spatial location."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).square().mean(dim=1, keepdim=True)
        return (x - mu) / torch.sqrt(var + self.eps) * self.weight + self.bias


class SharedModalityEncoder(nn.Module):
    def __init__(self, widths=(64,128,256,512)):
        super().__init__()
        self.sobel = Sobel1ch()
        layers = []
        in_c = 2
        for w in widths:
            layers.append(ConvBlock(in_c, w))
            in_c = w
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x [N,1,H,W]
        y = torch.cat([x, self.sobel(x)], dim=1)
        feats = []
        for layer in self.layers:
            y = layer(y)
            feats.append(y)
        return feats


class DynamicSetFusion(nn.Module):
    def __init__(self, channels: int, token_dim: int = 128):
        super().__init__()
        hidden = max(channels // 2, 32)
        self.gate = nn.Sequential(
            nn.Linear(channels + token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.norm = LayerNorm2d(channels)
        self.src_mask_proj = nn.Sequential(
            nn.Linear(3, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, feats: torch.Tensor, src_tokens: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # feats [B,M,C,H,W], src_tokens [B,M,D], src_mask [B,M]
        B, M, C, H, W = feats.shape
        pooled = feats.mean(dim=(-2,-1))
        gate_in = torch.cat([pooled, src_tokens], dim=-1)
        gates = torch.sigmoid(self.gate(gate_in)).view(B, M, C, 1, 1)
        m = src_mask[:, :, None, None, None].float()
        fused = (feats * gates * m).sum(dim=1)
        fused = self.norm(fused)
        cond = self.src_mask_proj(src_mask.float()).view(B, C, 1, 1)
        return fused + cond


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=32, gc=16):
        super().__init__()
        self.c1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.c2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.c3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.c4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.c5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x,x1],1)))
        x3 = self.act(self.c3(torch.cat([x,x1,x2],1)))
        x4 = self.act(self.c4(torch.cat([x,x1,x2,x3],1)))
        x5 = self.c5(torch.cat([x,x1,x2,x3,x4],1))
        return x + 0.2 * x5


class RRDB(nn.Module):
    def __init__(self, nf=32, gc=16):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf,gc)
        self.rdb2 = ResidualDenseBlock(nf,gc)
        self.rdb3 = ResidualDenseBlock(nf,gc)

    def forward(self, x):
        y = self.rdb3(self.rdb2(self.rdb1(x)))
        return x + 0.2 * (y - x)


class UpConvBlockAdaIN(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, style_dim: int = 512):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(in_c + skip_c, out_c, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_c, affine=False)
        self.style_fc = nn.Linear(style_dim, out_c * 2)
        self.act = nn.ReLU(inplace=True)
        nn.init.zeros_(self.style_fc.weight)
        nn.init.zeros_(self.style_fc.bias)

    def forward(self, x, skip, style):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        gamma, beta = self.style_fc(style).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(self.norm(x) * (1 + gamma) + beta)


class PhysicsDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(512,256,3,1,1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(256,128,3,1,1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(128,64,3,1,1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(64,32,3,1,1), nn.GELU(),
            nn.Conv2d(32,3,3,1,1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class SubsetSynthV2(nn.Module):
    def __init__(self, img_size=512, token_dim=128, transformer_layers=4, transformer_heads=8, ff_dim=1024, physics_enabled=False):
        super().__init__()
        self.img_size = int(img_size)
        self.token_dim = int(token_dim)
        self.physics_enabled = bool(physics_enabled)
        self.modality_emb = nn.Embedding(3, token_dim)
        self.role_src = nn.Parameter(torch.zeros(token_dim))
        self.role_tgt = nn.Parameter(torch.zeros(token_dim))
        self.tgt_to_style = nn.Linear(token_dim, 512)

        self.encoder = SharedModalityEncoder((64,128,256,512))
        self.fusions = nn.ModuleList([
            DynamicSetFusion(64, token_dim),
            DynamicSetFusion(128, token_dim),
            DynamicSetFusion(256, token_dim),
            DynamicSetFusion(512, token_dim),
        ])

        bottleneck_hw = max(self.img_size // 16, 1)
        self.pos_embed_2d = nn.Parameter(torch.randn(1,512,bottleneck_hw,bottleneck_hw) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=512, nhead=transformer_heads, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)

        self.dec = nn.ModuleList([
            UpConvBlockAdaIN(512,256,256),
            UpConvBlockAdaIN(256,128,128),
            UpConvBlockAdaIN(128,64,64),
            UpConvBlockAdaIN(64,0,32),
        ])
        self.rrdb = RRDB(32,16)
        self.out = nn.Sequential(nn.Conv2d(32,1,3,1,1), nn.Sigmoid())
        self.physics_decoder = PhysicsDecoder() if self.physics_enabled else None

    def _source_tokens(self, B: int, device) -> torch.Tensor:
        ids = torch.arange(3, device=device)
        base = self.modality_emb(ids) + self.role_src
        return base.unsqueeze(0).expand(B,-1,-1)

    def encode(self, images: torch.Tensor, src_mask: torch.Tensor):
        # images [B,3,H,W]. Hidden targets are zeroed before shared encoder.
        B, M, H, W = images.shape
        x = images * src_mask[:, :, None, None].float()
        flat = x.reshape(B*M,1,H,W)
        flat_feats = self.encoder(flat)
        src_tokens = self._source_tokens(B, images.device)
        fused_scales = []
        for feat, fusion in zip(flat_feats, self.fusions):
            _, C, h, w = feat.shape
            feat = feat.view(B,M,C,h,w)
            # Explicitly kill biased features for unobserved slots after the encoder stage.
            feat = feat * src_mask[:, :, None, None, None].float()
            fused_scales.append(fusion(feat, src_tokens, src_mask))

        feat = fused_scales[-1]
        B,C,h,w = feat.shape
        if self.pos_embed_2d.shape[-2:] != (h,w):
            pos = F.interpolate(self.pos_embed_2d, size=(h,w), mode="bilinear", align_corners=False)
        else:
            pos = self.pos_embed_2d
        seq = (feat + pos).flatten(2).transpose(1,2)
        seq = self.transformer(seq)
        z = seq.transpose(1,2).reshape(B,C,h,w)
        return z, fused_scales[:-1]

    def target_style(self, target_idx: torch.Tensor) -> torch.Tensor:
        token = self.modality_emb(target_idx) + self.role_tgt
        return self.tgt_to_style(token)

    def decode(self, z: torch.Tensor, skips: List[torch.Tensor], target_idx: torch.Tensor) -> torch.Tensor:
        style = self.target_style(target_idx)
        x = self.dec[0](z, skips[2], style)
        x = self.dec[1](x, skips[1], style)
        x = self.dec[2](x, skips[0], style)
        x = self.dec[3](x, None, style)
        return self.out(self.rrdb(x))

    def physics_maps(self, z: torch.Tensor) -> torch.Tensor:
        if self.physics_decoder is None:
            raise RuntimeError("Physics branch is disabled in this model instance")
        return self.physics_decoder(z)

    def forward(self, images: torch.Tensor, src_mask: torch.Tensor, target_idx: torch.Tensor):
        z, skips = self.encode(images, src_mask)
        pred = self.decode(z, skips, target_idx)
        return pred, z
