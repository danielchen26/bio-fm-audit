"""Step 2: structure tokenizer (VQ-VAE). Continuous 3D geometry -> discrete tokens.

Encoder : SE(3)-invariant local-neighbourhood features -> latent
Quantise: EMA codebook + dead-code revival (the two things that stop collapse)
Decoder : tokens -> backbone frames, trained with FAPE + distogram aux loss

Verification metric = CA RMSD after Kabsch superposition on a held-out,
30%-identity-disjoint test set.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ROOT, batches, load_splits
from nn import RMSNorm, Trunk, fape, init_weights, kabsch_rmsd, local_frames, rbf, rot6d_to_matrix

K_NEIGH = 16
LOCAL_W = [10.0]   # mutable so the CLI can ablate the local head to zero
N_DISTO = 32


# ------------------------------------------------------------------- encoder
def neighbor_geometry(coords, mask, k_max: int = K_NEIGH):
    """SE(3)-invariant local neighbourhood descriptors. Shared by encoder and loss."""
    B, L = mask.shape
    R, t = local_frames(coords)                                   # [B,L,3,3], [B,L,3]
    d = torch.cdist(t, t) + (~mask[:, None, :]) * 1e4
    d = d + torch.eye(L, device=d.device)[None] * 1e4              # exclude self
    k = min(k_max, L - 1) if L > 1 else 1
    nd, ni = torch.topk(d, k, dim=-1, largest=False)               # [B,L,k]
    bi = torch.arange(B, device=d.device)[:, None, None]
    rel = t[bi, ni] - t[:, :, None, :]
    # R columns are the basis => R^T rel expresses the neighbour inside frame i
    loc = torch.einsum("bick,bijc->bijk", R, rel) / 10.0
    rot = torch.einsum("bica,bijce->bijae", R, R[bi, ni]).flatten(-2)   # [B,L,k,9]
    off = torch.arange(L, device=d.device)[None, :, None] - ni
    valid = mask[bi, ni] & (nd < 1e3)
    return loc, rot, nd, off, valid


class GeometricEncoder(nn.Module):
    """Per-residue token from the local 3D neighbourhood. Invariant by construction."""

    def __init__(self, d: int, d_z: int, layers: int = 2, heads: int = 8):
        super().__init__()
        self.offset = nn.Embedding(65, 16)                       # signed seq offset, clamped +-32
        self.nbr = nn.Sequential(nn.Linear(3 + 9 + 16 + 16, d), nn.GELU(), nn.Linear(d, d))
        self.score = nn.Linear(d, 1)
        self.trunk = Trunk(d, heads, layers)
        self.out = nn.Sequential(RMSNorm(d), nn.Linear(d, d_z))

    def forward(self, coords, mask):
        loc, rot, nd, off, valid = neighbor_geometry(coords, mask)
        feat = torch.cat([loc, rot, rbf(nd), self.offset(off.clamp(-32, 32) + 32)], dim=-1)
        h = self.nbr(feat)
        # finite fill, not -inf: a fully-padded row would otherwise softmax to NaN
        w = self.score(h).masked_fill(~valid[..., None], -1e4).softmax(dim=-2)
        h = (h * w).sum(-2) * mask[..., None]                      # attention pool over neighbours
        h = self.trunk(h, mask)
        return self.out(h)


# ------------------------------------------------------------------ quantiser
class VectorQuantiserEMA(nn.Module):
    def __init__(self, size: int, dim: int, decay: float = 0.99, beta: float = 0.25):
        super().__init__()
        self.size, self.decay, self.beta = size, decay, beta
        self.register_buffer("codebook", torch.randn(size, dim) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(size))
        self.register_buffer("embed_avg", self.codebook.clone())
        self.register_buffer("inited", torch.zeros((), dtype=torch.bool))

    def forward(self, z, mask):
        flat = z[mask]                                             # [Nvalid, dim]
        if self.training and not bool(self.inited):
            # seed the codebook FROM DATA. Random init is the #1 cause of VQ collapse.
            with torch.no_grad():
                pick = torch.randint(len(flat), (self.size,), device=flat.device)
                self.codebook.copy_(flat[pick].detach())
                self.embed_avg.copy_(self.codebook)
                self.cluster_size.fill_(1.0)
                self.inited.fill_(True)
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.T + self.codebook.pow(2).sum(1)
        idx = d.argmin(1)
        e = self.codebook[idx]

        if self.training:
            with torch.no_grad():
                onehot = F.one_hot(idx, self.size).to(flat.dtype)
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = (self.cluster_size + 1e-5) / (n + self.size * 1e-5) * n
                self.codebook.copy_(self.embed_avg / cs[:, None])
                # dead-code revival: reseed unused codes from live encoder outputs
                dead = self.cluster_size < 1e-2
                if dead.any():
                    src = flat[torch.randint(len(flat), (int(dead.sum()),), device=flat.device)]
                    self.codebook[dead] = src
                    self.cluster_size[dead] = 1.0
                    self.embed_avg[dead] = src

        commit = F.mse_loss(flat, e.detach()) * self.beta
        q = flat + (e - flat).detach()                             # straight-through
        out = torch.zeros_like(z)
        out[mask] = q
        tok = torch.zeros(mask.shape, dtype=torch.long, device=z.device)
        tok[mask] = idx
        p = F.one_hot(idx, self.size).sum(0).float()               # bincount is patchy on MPS
        p = p / p.sum()
        ppl = torch.exp(-(p * (p + 1e-10).log()).sum())
        return out, tok, commit, ppl, (p > 0).float().sum()


# ------------------------------------------------------------------- decoder
class StructureDecoder(nn.Module):
    def __init__(self, d: int, d_z: int, layers: int = 4, heads: int = 8):
        super().__init__()
        self.inp = nn.Linear(d_z, d)
        self.trunk = Trunk(d, heads, layers)
        self.frame = nn.Linear(d, 9)                               # 6D rotation + translation
        self.pair_a, self.pair_b = nn.Linear(d, 64), nn.Linear(d, 64)
        self.disto = nn.Sequential(nn.GELU(), nn.Linear(64, N_DISTO))
        # the real autoencoder objective: reconstruct the LOCAL geometry that was
        # encoded. Without this the token is only pressured to help a badly
        # conditioned global-coordinate task, and carries almost no local signal.
        self.local = nn.Sequential(nn.GELU(), nn.Linear(d, K_NEIGH * 3))

    def forward(self, z, mask):
        h = self.trunk(self.inp(z), mask)
        f = self.frame(h)
        R = rot6d_to_matrix(f[..., :6])
        t = f[..., 6:] * 10.0
        pair = self.pair_a(h)[:, :, None] + self.pair_b(h)[:, None, :]
        loc = self.local(h).view(*h.shape[:2], K_NEIGH, 3)
        return R, t, self.disto(pair), loc


class StructTokenizer(nn.Module):
    def __init__(self, d=256, d_z=128, codebook=512):
        super().__init__()
        self.enc = GeometricEncoder(d, d_z)
        self.vq = VectorQuantiserEMA(codebook, d_z)
        self.dec = StructureDecoder(d, d_z)
        self.apply(init_weights)

    def encode(self, coords, mask):
        z = self.enc(coords, mask)
        _, tok, _, _, _ = self.vq(z, mask)
        return tok

    def forward(self, coords, mask):
        z = self.enc(coords, mask)
        zq, tok, commit, ppl, used = self.vq(z, mask)
        R, t, disto, loc = self.dec(zq, mask)
        return R, t, disto, loc, tok, commit, ppl, used


# ---------------------------------------------------------------------- losses
def disto_target(coords, mask):
    d = torch.cdist(coords[:, :, 1], coords[:, :, 1])
    edges = torch.linspace(2.0, 22.0, N_DISTO - 1, device=d.device)
    y = torch.bucketize(d, edges)
    pair = mask[:, :, None] & mask[:, None, :]
    return y.masked_fill(~pair, -100)


def forward_and_loss(model, batch):
    coords, mask = batch["coords"], batch["mask"]
    R, t, disto, loc, tok, commit, ppl, used = model(coords, mask)
    tR, tt = local_frames(coords)
    pair = mask[:, :, None] & mask[:, None, :]
    l_fape = fape(R, t, tR, tt, t, tt, mask)
    l_dist = F.cross_entropy(disto.flatten(0, 2), disto_target(coords, mask).flatten(), ignore_index=-100)
    # fully invariant distance-map loss: trains global coords far faster than FAPE alone
    dmap = (torch.cdist(t, t) - torch.cdist(tt, tt)).abs().clamp(max=10.0) / 10.0
    l_dmap = (dmap * pair).sum() / pair.sum().clamp(min=1)
    # local autoencoder term: reconstruct the neighbourhood the encoder consumed
    tgt_loc, _, _, _, valid = neighbor_geometry(coords, mask)
    k = tgt_loc.shape[2]
    err = F.smooth_l1_loss(loc[:, :, :k], tgt_loc, reduction="none").mean(-1)
    l_local = (err * valid).sum() / valid.sum().clamp(min=1)
    stats = {"fape": l_fape.item(), "dmap": l_dmap.item(), "disto": l_dist.item(),
             "local": l_local.item(), "commit": commit.item(), "ppl": ppl.item(), "used": used.item()}
    total = l_fape + l_dmap + 0.5 * l_dist + LOCAL_W[0] * l_local + commit
    return total, stats, t


def loss_fn(model, batch):
    loss, stats, _ = forward_and_loss(model, batch)
    return loss, stats


@torch.no_grad()
def evaluate(model, records, device, batch_tokens):
    model.eval()
    rmsds, stats, n = [], {}, 0
    for b in batches(records, batch_tokens, device, shuffle=False):
        _, s, t = forward_and_loss(model, b)
        for k, v in s.items():
            stats[k] = stats.get(k, 0.0) + v
        n += 1
        for i in range(len(t)):
            m = b["mask"][i]
            rmsds.append(kabsch_rmsd(t[i][m].float(), b["coords"][i][m][:, 1].float()).item())
    model.train()
    return {**{k: v / n for k, v in stats.items()}, "ca_rmsd": sum(rmsds) / len(rmsds)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-tokens", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--codebook", type=int, default=512)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="vqvae.pt")
    ap.add_argument("--local-weight", type=float, default=10.0)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    a = ap.parse_args()

    LOCAL_W[0] = a.local_weight
    dev = torch.device(a.device)
    splits = load_splits()
    model = StructTokenizer(codebook=a.codebook).to(dev)
    print(f"[vqvae] params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)

    step, t0 = 0, time.time()
    while step < a.steps:
        for b in batches(splits["train"], a.batch_tokens, dev, seed=step):
            if step >= a.steps:
                break
            loss, s = loss_fn(model, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 250 == 0 or step == 1:
                print(f"[{step:5d}] loss {loss.item():.4f} local {s['local']:.4f} dmap {s['dmap']:.4f} "
                      f"fape {s['fape']:.3f} disto {s['disto']:.3f} codes {int(s['used']):3d}/{a.codebook} "
                      f"ppl {s['ppl']:6.1f} gn {gn:.2f} {time.time() - t0:6.1f}s", flush=True)
            if step % a.ckpt_every == 0:
                ck = ROOT / "ckpt" / a.out
                ck.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "codebook": a.codebook,
                            "step": step, "opt": opt.state_dict()}, ck)
    for name in ("val", "test"):
        e = evaluate(model, splits[name], dev, a.batch_tokens)
        print(f"[vqvae/{name}] CA-RMSD {e['ca_rmsd']:.2f}A  local {e['local']:.4f}  "
              f"dmap {e['dmap']:.4f}  disto {e['disto']:.3f}  codes {int(e['used'])}/{a.codebook}")
    out = ROOT / "ckpt" / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "codebook": a.codebook}, out)
    print(f"[vqvae] saved {out}")


if __name__ == "__main__":
    main()
