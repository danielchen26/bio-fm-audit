"""Cross-modal FM: tissue morphology <-> spatial transcriptome.

Same machinery as the protein model earlier in this project -- one trunk, several
tracks, independent per-track masking, whole-track dropout for any-to-all
conditioning. Only the modality pair changed:

    protein:   sequence  <-> structure tokens <-> SS8/SASA
    pathology: H&E patch <-> gene expression  <-> spatial position

THE QUESTION
The H&E-to-expression literature reports correlations without separating two
sources of signal: (a) morphology genuinely predicts expression, (b) neighbouring
spots are measured and expression is spatially autocorrelated, so smoothing alone
scores well. Only (a) is useful on a slide with no transcriptomics. So we
evaluate under conditions that isolate them:

    morph+ctx  target morphology given, neighbours measured   (both sources)
    morph-only target morphology given, neighbours masked     (morphology only)
    ctx-only   target morphology masked, neighbours measured  (spatial only)

TWO HEADS, AND THIS COMPARISON IS THE EXPERIMENT
    mse   deterministic regression. Its optimum is E[x|c], and the conditional
          mean of a spatially autocorrelated field IS the smoothed field. So an
          MSE head structurally cannot show morphology contributing anything that
          smoothing does not already give -- even if morphology fully determines
          the conditional DISTRIBUTION.
    flow  rectified-flow matching. Learns p(x|c), so morphology can pay off by
          sharpening or splitting the distribution rather than shifting its mean.

Scored with CRPS (a strictly proper scoring rule computable from samples and a
single observation), plus interval coverage and sharpness. Pearson r of a point
estimate is reported too, for continuity with the literature, but it is the weaker
metric and it is the one that hides the effect.

MORPHOLOGY ENCODERS (all frozen, precomputed)
    random   randomly initialised conv -- the null control.
    imagenet ViT-S/16 ImageNet. Did we need pathology pretraining at all?
    phikon   Owkin Phikon, ViT-B iBOT on TCGA. A real pathology foundation model.

SPLIT: spatially blocked. Adjacent Visium spots sit ~100 um apart and are
strongly correlated, so a random spot split leaks exactly the way a random patch
split leaks across a patient. Whole tiles of the slide are held out.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathdata import DATA, ROOT

K_NEIGH = 8
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ data prep
def load_paired(path="hest_sample.npz"):
    z = np.load(DATA / path, allow_pickle=False)
    return z["patches"], z["expr"].astype(np.float32), z["coords"].astype(np.float32), z["gene_names"]


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
    return np.argsort(d, axis=1)[:, :k]


# ----------------------------------------------- frozen morphology embeddings
class RandomConv(nn.Module):
    """Untrained conv stem. Null control -- random features are a strong baseline."""

    def __init__(self, d=384):
        super().__init__()
        ch = [3, 32, 64, 128, d]
        layers = []
        for a, b in zip(ch[:-1], ch[1:]):
            layers += [nn.Conv2d(a, b, 3, 2, 1), nn.GroupNorm(8, b), nn.GELU()]
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def precompute_morph(patches, kind, device, batch=64, tag=""):
    # tag MUST include the patch size: otherwise a 96px cache is silently
    # reused for a 224px run and the whole field-of-view experiment is void
    cache = DATA / f"morph_{kind}{tag}.npz"
    if cache.exists():
        f = np.load(cache)["feat"]
        print(f"[morph] cached {kind}: {f.shape}")
        return f
    dev = torch.device(device)
    if kind == "random":
        model, size, norm = RandomConv().to(dev).eval(), 96, "half"
    elif kind == "imagenet":
        import timm
        model = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k",
                                  pretrained=True, num_classes=0).to(dev).eval()
        size, norm = 224, "imagenet"
    elif kind == "phikon":
        from transformers import AutoModel
        model = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(dev).eval()
        size, norm = 224, "imagenet"
    else:
        raise ValueError(kind)
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    out = []
    for i in range(0, len(patches), batch):
        x = torch.from_numpy(patches[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != size:
            x = F.interpolate(x, (size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - mean) / std if norm == "imagenet" else (x - 0.5) / 0.25
        z = model(x)
        out.append((z if torch.is_tensor(z) else z.last_hidden_state[:, 0]).float().cpu().numpy())
    feat = np.concatenate(out).astype(np.float32)
    np.savez_compressed(cache, feat=feat)
    print(f"[morph] computed {kind}: {feat.shape} -> {cache.name}")
    return feat


# ----------------------------------------------------------------------- model
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
        m = self.morph_in(morph)
        m = torch.where(m_morph[..., None], self.morph_mask.expand_as(m), m)
        e = self.expr_in(expr * (~m_expr[..., None]))
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
    def __init__(self, morph, expr, coords, nbr, scale):
        self.morph, self.expr, self.coords, self.nbr, self.scale = morph, expr, coords, nbr, scale

    def build(self, centers, device):
        idx = np.concatenate([centers[:, None], self.nbr[centers]], axis=1)
        rel = (self.coords[idx] - self.coords[centers][:, None]) / self.scale
        t = lambda a: torch.from_numpy(a).to(device)
        return t(self.morph[idx]), t(self.expr[idx]), t(rel).float(), idx


# ---------------------------------------------------------------------- scoring
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


# ------------------------------------------------------------------- baselines
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


# ----------------------------------------------------------------------- train
def run(a):
    dev = torch.device(a.device)
    patches, expr, coords, genes = load_paired(a.src)
    G = expr.shape[1]
    morph = precompute_morph(patches, a.morph, a.device,
                             tag=f"_p{patches.shape[1]}")

    is_test, bid = spatial_blocks(coords, a.blocks, a.seed)
    tr_mask = ~is_test
    mu_g, sd_g = expr[tr_mask].mean(0), expr[tr_mask].std(0) + 1e-6
    exprz = ((expr - mu_g) / sd_g).astype(np.float32)

    nbr = knn_graph(coords, K_NEIGH)
    scale = float(np.linalg.norm(coords.max(0) - coords.min(0)) / 10.0)
    bags = Bags(morph, exprz, coords, nbr, scale)
    tr_idx, te_idx = np.where(tr_mask)[0], np.where(is_test)[0]
    print(f"[xmodal] {len(expr)} spots, {G} genes, patch={patches.shape[1]}px, "
          f"morph={a.morph}({morph.shape[1]}d), "
          f"head={a.head} | blocks {a.blocks}x{a.blocks} | train {len(tr_idx)} test {len(te_idx)} "
          f"| {len(set(bid[is_test].tolist()))} held-out tiles")

    model = CrossModal(G, morph.shape[1], d=a.d, layers=a.layers, head=a.head).to(dev)
    print(f"[xmodal] params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    t0 = time.time()

    for step in range(1, a.steps + 1):
        centers = rng.choice(tr_idx, a.batch)
        m_, e_, rel, idx = bags.build(centers, dev)
        B, K = idx.shape
        m_expr = torch.rand(B, K, device=dev) < torch.rand(B, 1, device=dev)
        m_morph = torch.rand(B, K, device=dev) < torch.rand(B, 1, device=dev) * 0.5
        m_expr[:, 0] = True                                        # always predict the centre
        m_expr[:, 1:] |= torch.rand(B, 1, device=dev) < 0.25       # sometimes hide all context
        m_morph |= torch.rand(B, 1, device=dev) < 0.15             # sometimes hide all morphology
        cond = model.cond(m_, e_, rel, m_morph, m_expr)
        loss = model.head.loss(cond, e_[:, 0])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 500 == 0 or step == 1:
            print(f"[{step:5d}] loss {loss.item():.4f} gn {gn:.2f} {time.time() - t0:6.1f}s", flush=True)

    @torch.no_grad()
    def draw(centers, target_morph=True, ctx_expr=True, n=a.samples):
        model.eval()
        outs = []
        for i in range(0, len(centers), 128):
            m_, e_, rel, _ = bags.build(centers[i:i + 128], dev)
            B, K = m_.shape[:2]
            m_expr = torch.zeros(B, K, dtype=torch.bool, device=dev)
            m_morph = torch.zeros(B, K, dtype=torch.bool, device=dev)
            m_expr[:, 0] = True
            if not target_morph:
                m_morph[:, 0] = True
            if not ctx_expr:
                m_expr[:, 1:] = True
            cond = model.cond(m_, e_, rel, m_morph, m_expr)
            outs.append(model.head.sample(cond, n).cpu().numpy())
        model.train()
        return np.concatenate(outs)                     # [n_test, n, G]

    truth = exprz[te_idx]
    print("\n" + "=" * 94)
    print(f"spatially held-out tiles | morphology = {a.morph} | head = {a.head} | "
          f"{a.samples} samples/spot")
    print("=" * 94)
    print(f"  {'':38s} {'CRPS':>8} {'r(mean)':>9} {'cov90':>7} {'sharp':>7}")

    def report(tag, samples):
        c = float(np.nanmean(crps(samples, truth)))
        r = float(np.nanmean(pearson_per_gene(samples.mean(1), truth)))
        print(f"  {tag:38s} {c:8.4f} {r:+9.4f} {coverage(samples, truth):7.3f} "
              f"{float(samples.std(1).mean()):7.3f}")
        return c, r

    scores = {}
    nb_mu, nb_sd = neighbour_stats(exprz, nbr, tr_mask, te_idx)
    gauss = nb_mu[:, None, :] + nb_sd[:, None, :] * np.random.default_rng(0).standard_normal(
        (len(te_idx), a.samples, G)).astype(np.float32)
    scores["baseline neighbour-Gaussian"] = report("baseline  neighbour-Gaussian (smoothing)", gauss)
    scores["baseline neighbour-point"] = report("baseline  neighbour-mean (point)",
                                                nb_mu[:, None, :].repeat(a.samples, 1))
    scores[f"baseline morph-kNN"] = report(f"baseline  morphology-kNN ({a.morph})",
                                           morphology_knn(morph, exprz, tr_idx, te_idx, 10))
    for tag, kw in (("morph+ctx  (both sources)", dict(target_morph=True, ctx_expr=True)),
                    ("morph-only (no neighbours)", dict(target_morph=True, ctx_expr=False)),
                    ("ctx-only   (no morphology)", dict(target_morph=False, ctx_expr=True))):
        scores[tag] = report(f"model     {tag}", draw(te_idx, **kw))

    c_both = scores["morph+ctx  (both sources)"][0]
    c_ctx = scores["ctx-only   (no morphology)"][0]
    c_morph = scores["morph-only (no neighbours)"][0]
    c_smooth = scores["baseline neighbour-Gaussian"][0]
    print("\n[decomposition]  (CRPS, lower is better -- negative delta means it helped)")
    print(f"  morphology beyond spatial context : {c_both - c_ctx:+.4f}")
    print(f"  spatial context beyond morphology : {c_both - c_morph:+.4f}")
    print(f"  model vs probabilistic smoothing  : {c_both - c_smooth:+.4f}")
    # NO automated verdict here, deliberately. An earlier version declared
    # "morphology helps" whenever the delta beat zero, and duly declared it for
    # RANDOM features -- because any extra per-spot vector buys a little. The
    # null hypothesis for "our encoder learned something" is the random-feature
    # control, never zero. So print the numbers a control run must be compared
    # against and let the comparison happen across runs.
    print(f"\n[compare]  morph={a.morph} head={a.head} patch={patches.shape[1]}px")
    print(f"  morph-only CRPS (deployment condition, no measured neighbours) : {c_morph:.4f}")
    print(f"  morphology-beyond-context delta                               : {c_both - c_ctx:+.4f}")
    print("  -> compare BOTH against the same numbers from --morph random.")
    print("     If they do not beat the random-feature control, the encoder")
    print("     contributed nothing, whatever the pretraining loss said.")

    out = ROOT / "ckpt" / f"xmodal_{a.morph}_{a.head}_p{patches.shape[1]}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "genes": genes, "mu": mu_g, "sd": sd_g,
                "morph": a.morph, "head": a.head, "scores": scores}, out)
    print(f"[xmodal] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--src", default="hest_sample.npz")
    ap.add_argument("--morph", default="phikon", choices=["random", "imagenet", "phikon"])
    ap.add_argument("--head", default="flow", choices=["mse", "flow"])
    ap.add_argument("--device", default="mps")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
