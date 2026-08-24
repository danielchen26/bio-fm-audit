"""Shared transformer primitives: RMSNorm, RoPE, QK-norm attention, SwiGLU."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def rope_cache(seq: int, head_dim: int, device, base: float = 10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq, device=device).float()
    f = torch.outer(t, inv)
    return torch.cos(f)[None, None], torch.sin(f)[None, None]


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    """Multi-head attention with RoPE + QK-norm (QK-norm kills the loss spikes)."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)

    def forward(self, x, pad_mask, rope, bias=None):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        cos, sin = rope
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        m = pad_mask[:, None, None, :].expand(B, self.h, L, L).clone()      # [B,h,L,L] bool keep
        attn_bias = torch.zeros(B, self.h, L, L, device=x.device, dtype=q.dtype)
        attn_bias.masked_fill_(~m, float("-inf"))
        if bias is not None:
            attn_bias = attn_bias + bias
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        return self.o(out.transpose(1, 2).reshape(B, L, -1))


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 8 / 3):
        super().__init__()
        hidden = int(d * mult) // 8 * 8
        self.w12 = nn.Linear(d, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, -1)
        return self.w3(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d)

    def forward(self, x, pad_mask, rope, bias=None):
        x = x + self.attn(self.n1(x), pad_mask, rope, bias)
        return x + self.ff(self.n2(x))


class Trunk(nn.Module):
    def __init__(self, d: int, heads: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = RMSNorm(d)
        self.dh = d // heads

    def forward(self, x, pad_mask, bias=None):
        rope = rope_cache(x.shape[1], self.dh, x.device)
        for i, blk in enumerate(self.blocks):
            # geometric bias injected in the first block only (ESM3-style)
            x = blk(x, pad_mask, rope, bias if i == 0 else None)
        return self.norm(x)


# --------------------------------------------------------------- geometry utils
def local_frames(coords: torch.Tensor):
    """Gram-Schmidt backbone frames. coords [...,L,4,3] (N,CA,C,O) -> R [...,L,3,3], t [...,L,3]."""
    N, CA, C = coords[..., 0, :], coords[..., 1, :], coords[..., 2, :]
    e1 = F.normalize(C - CA, dim=-1)
    u = N - CA
    e2 = F.normalize(u - (u * e1).sum(-1, keepdim=True) * e1, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1), CA        # R columns = basis


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation parameterisation -> SO(3). x [...,6]."""
    a, b = x[..., :3], x[..., 3:]
    e1 = F.normalize(a, dim=-1)
    e2 = F.normalize(b - (b * e1).sum(-1, keepdim=True) * e1, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-1)


def fape(pred_R, pred_t, true_R, true_t, pred_p, true_p, mask, clamp: float = 10.0, scale: float = 10.0):
    """Frame-Aligned Point Error: rigid-motion invariant, no alignment needed."""
    # x[b,i,j] = R_i^T (p_j - t_i)
    def project(R, t, p):
        # R columns are the frame basis, so R^T @ rel == einsum("bick,bijc->bijk")
        rel = p[:, None, :, :] - t[:, :, None, :]                 # [B,i,j,3]
        return torch.einsum("bick,bijc->bijk", R, rel)
    d = (project(pred_R, pred_t, pred_p) - project(true_R, true_t, true_p)).norm(dim=-1)
    d = d.clamp(max=clamp) / scale
    m = mask[:, :, None] & mask[:, None, :]
    return (d * m).sum() / m.sum().clamp(min=1)


def kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """RMSD after optimal superposition. P,Q [L,3]."""
    P, Q = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = torch.linalg.svd(P.T @ Q)
    d = torch.sign(torch.det(U @ Vt))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
    return ((P @ (U @ D @ Vt) - Q).pow(2).sum(-1).mean()).sqrt()


def rbf(d: torch.Tensor, n: int = 16, dmin: float = 2.0, dmax: float = 22.0):
    mu = torch.linspace(dmin, dmax, n, device=d.device)
    sigma = (dmax - dmin) / n
    return torch.exp(-((d[..., None] - mu) / sigma) ** 2)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, std=0.02)
