"""Step 3: the multimodal foundation model (ESM3-style).

Four discrete tracks share one trunk and one latent space:
    seq    (20 aa + unk)   <- measured
    struct (VQ tokens)     <- frozen struct_tokenizer
    ss8    (DSSP 8-state)  <- derived from coords
    sasa   (4 burial bins) <- Shrake-Rupley from coords

Objective: generative masked modelling with an independent noise schedule per
track, plus whole-track dropout. That dropout is what buys any-to-all
conditioning -- inverse folding, forward folding and function-prompted
co-generation all come out of one set of weights.

LEAKAGE CONTROL: geometric attention conditions on true coordinates, so its
bias is gated by the structure mask. Otherwise `seq -> struct` evaluation would
read the answer off the coordinates.
"""
from __future__ import annotations

import argparse
import collections
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import ROOT, batches, load_splits
from nn import RMSNorm, Trunk, init_weights, rbf
from struct_tokenizer import StructTokenizer

N = {"seq": 21, "ss8": 8, "sasa": 4}          # struct filled in from the codebook
IGN = -100
LOSS_W = {"seq": 1.0, "struct": 1.0, "ss8": 0.5, "sasa": 0.5}


class MultimodalFM(nn.Module):
    def __init__(self, n_struct: int, tracks=("seq", "struct", "ss8", "sasa"),
                 d=384, heads=8, layers=10, geometric=True):
        super().__init__()
        self.tracks = tuple(tracks)
        self.n = {**N, "struct": n_struct}
        self.mask_id = {t: self.n[t] for t in self.tracks}          # highest id == MASK
        self.emb = nn.ModuleDict({t: nn.Embedding(self.n[t] + 1, d) for t in self.tracks})
        self.trunk = Trunk(d, heads, layers)
        self.head = nn.ModuleDict({t: nn.Sequential(RMSNorm(d), nn.Linear(d, self.n[t]))
                                   for t in self.tracks})
        self.geo = nn.Linear(16, heads, bias=False) if geometric and "struct" in tracks else None
        self.apply(init_weights)

    def forward(self, toks, mask, coords=None, struct_given=None):
        x = sum(self.emb[t](toks[t]) for t in self.tracks)
        bias = None
        if self.geo is not None and coords is not None:
            d = torch.cdist(coords[:, :, 1], coords[:, :, 1])
            bias = self.geo(rbf(d)).permute(0, 3, 1, 2)              # [B,heads,L,L]
            # gate: geometry is visible only between residues whose struct token is visible
            g = struct_given if struct_given is not None else mask
            vis = (g & mask)[:, None, :, None] & (g & mask)[:, None, None, :]
            bias = bias * vis
        h = self.trunk(x, mask, bias)
        return {t: self.head[t](h) for t in self.tracks}


# ------------------------------------------------------------- noise schedule
def corrupt(model, true, mask, rng, p_full_track: float = 0.15):
    """Per-example, per-track mask rate. Returns (inputs, labels, struct_given)."""
    B, L = mask.shape
    dev = mask.device
    toks, labels = {}, {}
    for t in model.tracks:
        u = torch.rand(B, 1, device=dev, generator=rng)
        rate = torch.cos(u * math.pi / 2).clamp(0.05, 1.0)           # mass at both extremes
        full = torch.rand(B, 1, device=dev, generator=rng) < p_full_track
        rate = torch.where(full, torch.ones_like(rate), rate)
        m = (torch.rand(B, L, device=dev, generator=rng) < rate) & mask
        # guarantee >=1 supervised position per example so the loss is defined
        first = torch.zeros_like(m).scatter_(1, mask.float().argmax(1, keepdim=True), True)
        m = m | (~m.any(1, keepdim=True) & mask.any(1, keepdim=True) & first)
        toks[t] = torch.where(m, torch.full_like(true[t], model.mask_id[t]), true[t])
        labels[t] = torch.where(m, true[t], torch.full_like(true[t], IGN))
    given = ~(toks["struct"] == model.mask_id["struct"]) if "struct" in model.tracks else mask
    return toks, labels, given


def track_loss(model, logits, labels):
    per, total = {}, 0.0
    for t in model.tracks:
        l = F.cross_entropy(logits[t].flatten(0, 1), labels[t].flatten(), ignore_index=IGN)
        per[t] = l.item()
        total = total + LOSS_W[t] * l
    return total, per


# -------------------------------------------------------- struct token caching
@torch.no_grad()
def add_struct_tokens(splits, tokenizer, device, batch_tokens=4000):
    """Freeze the tokenizer and materialise structure tokens once (ESM3 protocol)."""
    tokenizer.eval()
    for name, records in splits.items():
        order = sorted(range(len(records)), key=lambda i: len(records[i]["seq"]))
        ptr = 0
        for b in batches([records[i] for i in order], batch_tokens, device, shuffle=False):
            tok = tokenizer.encode(b["coords"], b["mask"])
            for i in range(len(tok)):
                n = int(b["mask"][i].sum())
                records[order[ptr]]["struct"] = tok[i, :n].cpu().numpy()
                ptr += 1
        assert ptr == len(records), (name, ptr, len(records))
    return splits


def tracks_of(batch, model):
    return {t: batch[t] for t in model.tracks}


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def conditional_eval(model, records, device, batch_tokens, target, given):
    """Fully mask `target`, reveal `given`, mask all other tracks. Returns accuracy."""
    model.eval()
    hit = tot = 0
    for b in batches(records, batch_tokens, device, shuffle=False):
        true = tracks_of(b, model)
        toks = {t: (true[t] if t in given else torch.full_like(true[t], model.mask_id[t]))
                for t in model.tracks}
        sg = b["mask"] if "struct" in given else torch.zeros_like(b["mask"])
        logits = model(toks, b["mask"], b["coords"], sg)
        pred = logits[target].argmax(-1)
        hit += ((pred == true[target]) & b["mask"]).sum().item()
        tot += b["mask"].sum().item()
    model.train()
    return hit / tot


def baselines(train_records, test_records):
    """The comparisons that MUST sit in the same table. Beat them or stop."""
    out = {}
    for t in ("seq", "struct", "ss8", "sasa"):
        cnt = collections.Counter()
        for r in train_records:
            cnt.update(r[t].tolist())
        top = cnt.most_common(1)[0][0]
        out[f"marginal -> {t}"] = (sum(int((r[t] == top).sum()) for r in test_records)
                                  / sum(len(r[t]) for r in test_records))
    # non-trivial baseline: most frequent residue per DSSP state
    tab = collections.defaultdict(collections.Counter)
    for r in train_records:
        for s, aa in zip(r["ss8"].tolist(), r["seq"].tolist()):
            tab[s][aa] += 1
    best = {s: c.most_common(1)[0][0] for s, c in tab.items()}
    hit = tot = 0
    for r in test_records:
        for s, aa in zip(r["ss8"].tolist(), r["seq"].tolist()):
            hit += int(best.get(s, -1) == aa)
            tot += 1
    out["ss8-conditional -> seq"] = hit / tot
    # non-trivial baseline: most frequent residue per structure token
    tab2 = collections.defaultdict(collections.Counter)
    for r in train_records:
        for s, aa in zip(r["struct"].tolist(), r["seq"].tolist()):
            tab2[s][aa] += 1
    best2 = {s: c.most_common(1)[0][0] for s, c in tab2.items()}
    hit = tot = 0
    for r in test_records:
        for s, aa in zip(r["struct"].tolist(), r["seq"].tolist()):
            hit += int(best2.get(s, -1) == aa)
            tot += 1
    out["structtoken-lookup -> seq"] = hit / tot
    return out


# ------------------------------------------------------------------ generation
@torch.no_grad()
def generate(model, length, device, prompt=None, steps=16, temp=1.0, seed=0):
    """MaskGIT / ESM3 iterative confidence-ordered unmasking."""
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    mask = torch.ones(1, length, dtype=torch.bool, device=device)
    toks = {t: torch.full((1, length), model.mask_id[t], dtype=torch.long, device=device)
            for t in model.tracks}
    frozen = {t: torch.zeros(1, length, dtype=torch.bool, device=device) for t in model.tracks}
    if prompt:
        for t, v in prompt.items():
            v = torch.as_tensor(v, device=device)[None]
            keep = v >= 0
            toks[t] = torch.where(keep, v.clamp(min=0), toks[t])
            frozen[t] = keep
    free = [t for t in model.tracks if not frozen[t].all()]
    for s in range(steps):
        logits = model(toks, mask, None, None)
        # cosine schedule: how many tokens stay masked after this step
        keep_masked = max(0, int(length * math.cos(math.pi / 2 * (s + 1) / steps)))
        for t in free:
            p = (logits[t] / temp).softmax(-1)[0].cpu()
            sampled = torch.multinomial(p, 1, generator=g).squeeze(-1)
            conf = p.gather(1, sampled[:, None]).squeeze(-1)
            cur = toks[t][0].cpu()
            is_masked = (cur == model.mask_id[t]) & ~frozen[t][0].cpu()
            cur = torch.where(is_masked, sampled, cur)
            conf = conf.masked_fill(~is_masked, float("inf"))        # already-decided stay
            if keep_masked > 0 and is_masked.any():
                k = min(keep_masked, int(is_masked.sum()))
                remask = conf.topk(k, largest=False).indices
                cur[remask] = model.mask_id[t]
            toks[t] = cur.to(device)[None]
    model.train()
    return {t: toks[t][0].cpu() for t in model.tracks}


# ----------------------------------------------------------------------- train
def run(a):
    dev = torch.device(a.device)
    splits = load_splits()
    ck = torch.load(ROOT / "ckpt" / a.vq, map_location=dev)
    tok = StructTokenizer(codebook=ck["codebook"]).to(dev)
    tok.load_state_dict(ck["model"])
    for p in tok.parameters():
        p.requires_grad_(False)
    splits = add_struct_tokens(splits, tok, dev)

    tracks = tuple(a.tracks.split(","))
    model = MultimodalFM(ck["codebook"], tracks=tracks, d=a.d, layers=a.layers,
                         geometric=not a.no_geo).to(dev)
    print(f"[fm] params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M  "
          f"tracks={tracks}  geometric={model.geo is not None}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    rng = torch.Generator(device=dev).manual_seed(0)

    step, t0 = 0, time.time()
    while step < a.steps:
        for b in batches(splits["train"], a.batch_tokens, dev, seed=step):
            if step >= a.steps:
                break
            toks, labels, given = corrupt(model, tracks_of(b, model), b["mask"], rng)
            logits = model(toks, b["mask"], b["coords"], given)
            loss, per = track_loss(model, logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 500 == 0 or step == 1:
                print(f"[{step:5d}] loss {loss.item():.4f} | " +
                      " ".join(f"{t} {per[t]:.3f}" for t in tracks) +
                      f" | gn {gn:.2f} {time.time() - t0:6.1f}s", flush=True)

    out = ROOT / "ckpt" / a.out
    torch.save({"model": model.state_dict(), "codebook": ck["codebook"], "tracks": tracks,
                "d": a.d, "layers": a.layers, "geo": not a.no_geo}, out)
    print(f"[fm] saved {out}")

    print("\n=== any-to-all conditional accuracy | test split, 30%-identity disjoint ===")
    probes = [
        ("inverse folding    struct+ss8+sasa -> seq", "seq", ("struct", "ss8", "sasa")),
        ("inverse folding    struct          -> seq", "seq", ("struct",)),
        ("forward folding    seq             -> struct", "struct", ("seq",)),
        ("secondary struct   seq             -> ss8", "ss8", ("seq",)),
        ("burial             seq             -> sasa", "sasa", ("seq",)),
        ("unconditional      -               -> seq", "seq", ()),
    ]
    for label, tgt, given in probes:
        if tgt not in tracks or any(g not in tracks for g in given):
            continue
        acc = conditional_eval(model, splits["test"], dev, a.batch_tokens, tgt, given)
        print(f"  {label:44s} {acc * 100:5.1f}%")
    print("\n=== baselines (same test split) ===")
    for k, v in baselines(splits["train"], splits["test"]).items():
        print(f"  {k:44s} {v * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-tokens", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--tracks", default="seq,struct,ss8,sasa")
    ap.add_argument("--no-geo", action="store_true", help="ablate geometric attention")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--vq", default="vqvae.pt")
    ap.add_argument("--out", default="fm.pt")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
