"""ViT backbone + GPU-batched pathology augmentations.

Two augmentation choices are domain decisions, not defaults copied from ImageNet:

1. HED stain-space jitter instead of RGB colour jitter. H&E variation is
   physically a variation in *stain concentration*, so perturbing the
   haematoxylin/eosin channels after colour deconvolution generates realistic
   inter-lab appearance shifts. RGB jitter generates colours no scanner makes.
   (Tellez et al., stain augmentation.)

2. Full dihedral group (flips + 90 degree rotations). Histology has no canonical
   up. Natural-image SSL recipes deliberately avoid vertical flips; here they are
   free, label-preserving augmentation.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ruifrok & Johnston HED <-> RGB optical-density basis
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29],
                             [0.07, 0.99, 0.11],
                             [0.27, 0.57, 0.78]], dtype=torch.float32)


# ----------------------------------------------------------------- augmentation
class PathologyAug(nn.Module):
    """Batched, on-device. Input uint8 [B,H,W,3] -> float [B,3,S,S] normalised."""

    def __init__(self, out: int, scale=(0.4, 1.0), hed_sigma=0.10, blur_p=0.3):
        super().__init__()
        self.out, self.scale, self.hed_sigma, self.blur_p = out, scale, hed_sigma, blur_p
        self.register_buffer("rgb_from_hed", RGB_FROM_HED)
        self.register_buffer("hed_from_rgb", torch.linalg.inv(RGB_FROM_HED))
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        k = (k[:, None] * k[None, :]) / 256.0
        self.register_buffer("blur_k", k[None, None].repeat(3, 1, 1, 1))

    def hed_jitter(self, x):
        """x in [0,1], [B,3,H,W]. Perturb stain concentrations multiplicatively."""
        B = x.shape[0]
        od = -torch.log10(x.clamp(min=1e-6))
        c = torch.einsum("kc,bchw->bkhw", self.hed_from_rgb, od)
        a = 1.0 + (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1) * self.hed_sigma
        b = (torch.rand(B, 3, 1, 1, device=x.device) * 2 - 1) * self.hed_sigma
        c = c * a + b
        od = torch.einsum("ck,bkhw->bchw", self.rgb_from_hed, c)
        return torch.pow(10.0, -od).clamp(0, 1)

    def forward(self, u8):
        x = u8.permute(0, 3, 1, 2).float() / 255.0
        B, _, H, W = x.shape
        # random resized crop via affine grid (one sampling op for the whole batch)
        s = torch.empty(B, device=x.device).uniform_(*self.scale).sqrt()
        cx = (torch.rand(B, device=x.device) * 2 - 1) * (1 - s)
        cy = (torch.rand(B, device=x.device) * 2 - 1) * (1 - s)
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 1, 1] = s, s
        theta[:, 0, 2], theta[:, 1, 2] = cx, cy
        grid = F.affine_grid(theta, (B, 3, self.out, self.out), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        # dihedral group: 4 rotations x 2 flips, applied per sub-batch
        r = torch.randint(0, 4, (1,)).item()
        if r:
            x = torch.rot90(x, r, dims=(2, 3))
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=(3,))
        x = self.hed_jitter(x)
        if torch.rand(1).item() < self.blur_p:
            x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.blur_k, groups=3)
        return (x - 0.5) / 0.25


# -------------------------------------------------------------------------- ViT
class Block(nn.Module):
    def __init__(self, d, heads, mlp=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp)), nn.GELU(), nn.Linear(int(d * mlp), d))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class ViT(nn.Module):
    """Small ViT whose positional embedding interpolates, so one model handles
    both the large global crops and the cheap small local crops (real multi-crop)."""

    def __init__(self, img=96, patch=8, d=192, depth=6, heads=3):
        super().__init__()
        self.patch, self.d, self.grid0 = patch, d, img // patch
        self.proj = nn.Conv2d(3, d, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, self.grid0 ** 2 + 1, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(d)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def _pos(self, g):
        if g == self.grid0:
            return self.pos
        cls, grid = self.pos[:, :1], self.pos[:, 1:]
        grid = grid.reshape(1, self.grid0, self.grid0, self.d).permute(0, 3, 1, 2)
        grid = F.interpolate(grid, size=(g, g), mode="bicubic", align_corners=False)
        return torch.cat([cls, grid.permute(0, 2, 3, 1).reshape(1, g * g, self.d)], dim=1)

    def forward(self, x):
        x = self.proj(x)
        g = x.shape[-1]
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], dim=1) + self._pos(g)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)[:, 0]                       # CLS embedding


class DINOHead(nn.Module):
    """MLP -> L2 normalise -> weight-normalised prototype layer."""

    def __init__(self, d_in, d_hidden=512, d_bottleneck=64, n_proto=4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_bottleneck))
        self.last = nn.utils.parametrizations.weight_norm(
            nn.Linear(d_bottleneck, n_proto, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1, p=2))


def cosine(step, total, base, final, warmup=0):
    if step < warmup:
        return base * step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return final + 0.5 * (base - final) * (1 + math.cos(math.pi * p))
