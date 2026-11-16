"""Neighbourhood Transformer + conditional rectified-flow head, as used by the November 2026 analyses.

This is the model code of the revised cross-modal script: compared with `src/xmodal.py` it keeps the
expression of held-out spots out of both the conditioning context and the training targets (`Bags`
takes an `observed` mask, and normalisation is fitted on observed spots only). The architecture and
the training recipe are otherwise the same: tokens = morphology + expression + relative position for
the centre spot and its 8 nearest neighbours, a 4-layer Transformer, and a 3-layer MLP velocity field
trained with a straight-line (rectified) flow-matching loss and sampled with 32 Euler steps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K_NEIGH = 8

def normalize_expression(expr, tr_mask):
    """Fit normalization only on measured training spots; keep truth separate."""
    tr_mask = np.asarray(tr_mask, dtype=bool)
    if tr_mask.shape != (len(expr),) or not tr_mask.any():
        raise ValueError("normalization requires a nonempty training mask")
    mu, sd = expr[tr_mask].mean(0), expr[tr_mask].std(0) + 1e-6
    return ((expr - mu) / sd).astype(np.float32), mu, sd


def spatial_blocks(coords, n_blocks=6, seed=0):
    lo, hi = coords.min(0), coords.max(0)
    ij = np.clip(((coords - lo) / (hi - lo + 1e-6) * n_blocks).astype(int), 0, n_blocks - 1)
    bid = ij[:, 0] * n_blocks + ij[:, 1]
    order = np.random.default_rng(seed).permutation(n_blocks * n_blocks)
    test_blocks = set(order[:max(1, len(order) // 5)].tolist())
    return np.array([b in test_blocks for b in bid]), bid


def knn_graph(coords, k=K_NEIGH):
    d = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :min(k, max(0, len(coords) - 1))]


class Trunk(nn.Module):
    """Shared multimodal trunk over a spatial neighbourhood of spots."""

    def __init__(self, n_genes, d_morph, d=256, layers=4, heads=8):
        super().__init__()
        self.morph_in = nn.Sequential(nn.LayerNorm(d_morph), nn.Linear(d_morph, d))
        self.expr_in = nn.Linear(n_genes, d)
        self.pos = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.morph_mask = nn.Parameter(torch.zeros(d))
        self.expr_mask = nn.Parameter(torch.zeros(d))
        enc = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True,
                                         norm_first=True, dropout=0.0)
        self.net = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        nn.init.trunc_normal_(self.morph_mask, std=0.02)
        nn.init.trunc_normal_(self.expr_mask, std=0.02)

    def forward(self, morph, expr, rel, m_morph, m_expr):
        m_expr = m_expr.clone()
        m_expr[:, 0] = True                  # target expression is never conditioning
        m = self.morph_in(morph.masked_fill(m_morph[..., None], 0))
        m = torch.where(m_morph[..., None], self.morph_mask.expand_as(m), m)
        # Multiplication does not hide NaNs, and can poison parameter gradients.
        e = self.expr_in(expr.masked_fill(m_expr[..., None], 0))
        e = torch.where(m_expr[..., None], self.expr_mask.expand_as(e), e)
        return self.net(m + e + self.pos(rel))


class MSEHead(nn.Module):
    def __init__(self, d, n_genes):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_genes))

    def loss(self, cond, target):
        return F.mse_loss(self.net(cond), target)

    @torch.no_grad()
    def sample(self, cond, n):
        pred = self.net(cond)
        return pred[:, None].expand(-1, n, -1)          # point mass


class FlowHead(nn.Module):
    """Rectified flow: v_theta(x_t, t, cond) trained to match u = x_1 - x_0."""

    def __init__(self, d, n_genes, width=512, depth=3, t_dim=64):
        super().__init__()
        self.t_dim = t_dim
        layers, din = [], n_genes + t_dim + d
        for _ in range(depth):
            layers += [nn.Linear(din, width), nn.GELU()]
            din = width
        self.net = nn.Sequential(*layers, nn.Linear(width, n_genes))

    def t_embed(self, t):
        f = torch.exp(torch.linspace(0, 6, self.t_dim // 2, device=t.device))
        a = t[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def v(self, x_t, t, cond):
        return self.net(torch.cat([x_t, self.t_embed(t), cond], dim=-1))

    def loss(self, cond, target):
        B = target.shape[0]
        x0 = torch.randn_like(target)
        t = torch.rand(B, device=target.device)
        x_t = (1 - t)[:, None] * x0 + t[:, None] * target
        return F.mse_loss(self.v(x_t, t, cond), target - x0)

    @torch.no_grad()
    def sample(self, cond, n, steps=32):
        B, d = cond.shape
        c = cond[:, None].expand(-1, n, -1).reshape(B * n, d)
        x = torch.randn(B * n, self.net[-1].out_features, device=cond.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B * n,), i * dt, device=cond.device)
            x = x + self.v(x, t, c) * dt              # Euler; rectified paths are near-straight
        return x.view(B, n, -1)


class CrossModal(nn.Module):
    def __init__(self, n_genes, d_morph, d=256, layers=4, heads=8, head="mse"):
        super().__init__()
        self.trunk = Trunk(n_genes, d_morph, d, layers, heads)
        self.head = FlowHead(d, n_genes) if head == "flow" else MSEHead(d, n_genes)

    def cond(self, *a):
        return self.trunk(*a)[:, 0]                     # centre spot representation


class Bags:
    """Conditioning store containing only expression observed in training."""

    def __init__(self, morph, expr, coords, nbr, scale, observed):
        self.observed = np.asarray(observed, dtype=bool).copy()
        if self.observed.shape != (len(expr),) or not self.observed.any():
            raise ValueError("Bags requires a nonempty observed training mask")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Bags scale must be positive and finite")
        self.morph, self.coords, self.nbr, self.scale = morph, coords, nbr, scale
        self.expr = np.zeros_like(expr)
        self.expr[self.observed] = expr[self.observed]
        if not np.isfinite(self.expr).all():
            raise ValueError("observed training expression must be finite")
        self.expr.setflags(write=False)
        self.observed.setflags(write=False)

    def expression_mask(self, idx, device):
        masked = ~self.observed[idx] | (idx == idx[:, :1])
        return torch.from_numpy(masked).to(device)

    def targets(self, centers, device):
        if not self.observed[centers].all():
            raise ValueError("held-out expression cannot be a training target")
        return torch.from_numpy(self.expr[centers]).to(device)

    def build(self, centers, device):
        idx = np.concatenate([centers[:, None], self.nbr[centers]], axis=1)
        rel = (self.coords[idx] - self.coords[centers][:, None]) / self.scale
        context = self.expr[idx].copy()
        context[idx == centers[:, None]] = 0
        t = lambda a: torch.from_numpy(a).to(device)
        return t(self.morph[idx]), t(context), t(rel).float(), idx


def training_loss(model, bags, centers, device):
    m_, e_, rel, idx = bags.build(centers, device)
    B, K = idx.shape
    m_expr = bags.expression_mask(idx, device)
    m_expr |= torch.rand(B, K, device=device) < torch.rand(B, 1, device=device)
    m_morph = torch.rand(B, K, device=device) < torch.rand(B, 1, device=device) * 0.5
    m_expr[:, 1:] |= torch.rand(B, 1, device=device) < 0.25
    m_morph |= torch.rand(B, 1, device=device) < 0.15
    cond = model.cond(m_, e_, rel, m_morph, m_expr)
    return model.head.loss(cond, bags.targets(centers, device))


@torch.no_grad()
def draw(model, bags, centers, device, target_morph=True, ctx_expr=True, n=32):
    """Draw with train-observed context; target_morph=False hides all morphology."""
    was_training = model.training
    model.eval()
    outs = []
    try:
        for i in range(0, len(centers), 128):
            m_, e_, rel, idx = bags.build(centers[i:i + 128], device)
            m_expr = bags.expression_mask(idx, device)
            m_morph = torch.zeros(idx.shape, dtype=torch.bool, device=device)
            if not target_morph:
                m_morph[:] = True
            if not ctx_expr:
                m_expr[:] = True
            cond = model.cond(m_, e_, rel, m_morph, m_expr)
            outs.append(model.head.sample(cond, n).cpu().numpy())
    finally:
        model.train(was_training)
    return np.concatenate(outs)


def pearson_per_gene(pred, true):
    p, t = pred - pred.mean(0, keepdims=True), true - true.mean(0, keepdims=True)
    return (p * t).sum(0) / (np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0)) + 1e-12)


def crps(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS from an ensemble. samples [N,S,G], y [N,G] -> per-gene mean CRPS.

    CRPS = E|X - y| - 0.5 E|X - X'|. Strictly proper: a point forecast cannot win
    by being over-confident, and a wide forecast cannot win by hedging.
    """
    term1 = np.abs(samples - y[:, None, :]).mean(1)
    s = np.sort(samples, axis=1)
    S = s.shape[1]
    # E|X - X'| via the sorted-sample identity, O(S) instead of O(S^2)
    w = (2 * np.arange(1, S + 1) - S - 1).reshape(1, S, 1)
    term2 = 2.0 * (w * s).sum(1) / (S * S)
    return (term1 - 0.5 * term2).mean(0)


def coverage(samples, y, lo=5, hi=95):
    a, b = np.percentile(samples, [lo, hi], axis=1)
    return float(((y >= a) & (y <= b)).mean())


def neighbour_stats(expr, nbr, tr_mask, te_idx):
    """Probabilistic smoothing baseline: Gaussian from measured neighbours."""
    mu = np.zeros((len(te_idx), expr.shape[1]), np.float32)
    sd = np.zeros_like(mu)
    fb_mu, fb_sd = expr[tr_mask].mean(0), expr[tr_mask].std(0)
    for r, i in enumerate(te_idx):
        ok = [j for j in nbr[i] if tr_mask[j]]
        if ok:
            mu[r], sd[r] = expr[ok].mean(0), expr[ok].std(0) + 1e-3
        else:
            mu[r], sd[r] = fb_mu, fb_sd
    return mu, sd


def morphology_knn(feat, expr, tr_idx, te_idx, k=10):
    nrm = lambda a: a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    sim = nrm(feat[te_idx]) @ nrm(feat[tr_idx]).T
    top = np.argsort(-sim, axis=1)[:, :k]
    return expr[tr_idx][top]                            # [n_test, k, G] -> an ensemble
