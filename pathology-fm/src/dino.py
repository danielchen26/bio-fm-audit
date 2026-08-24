"""Self-supervised pretraining of the patch encoder (DINO).

DINO's failure mode is collapse: every image maps to the same output
distribution, the loss looks fine, and the embedding is worthless. It is exactly
the failure we hit with the VQ codebook earlier, in a different mechanism. So
collapse is monitored as a FIRST-CLASS metric, not inferred from the loss:

  teacher_entropy  -- H(mean teacher output). Collapse drives this to 0.
  kl_to_uniform    -- how far the mean output is from uniform over prototypes.
  top1_share       -- fraction of the batch assigned to the single most popular
                      prototype. Collapse drives this to 1.

If teacher_entropy falls while the loss also falls, the run is collapsing and
the loss is lying to you.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import DINOHead, PathologyAug, ViT, cosine
from pathdata import DATA, ROOT, load_manifest, make_split


class DINO(nn.Module):
    def __init__(self, img=96, patch=8, d=192, depth=6, heads=3, n_proto=4096):
        super().__init__()
        self.student, self.teacher = ViT(img, patch, d, depth, heads), ViT(img, patch, d, depth, heads)
        self.s_head, self.t_head = DINOHead(d, n_proto=n_proto), DINOHead(d, n_proto=n_proto)
        self.teacher.load_state_dict(self.student.state_dict())
        self.t_head.load_state_dict(self.s_head.state_dict())
        for p in list(self.teacher.parameters()) + list(self.t_head.parameters()):
            p.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, n_proto))
        self.n_proto = n_proto

    @torch.no_grad()
    def ema(self, m: float):
        for a, b in ((self.student, self.teacher), (self.s_head, self.t_head)):
            for ps, pt in zip(a.parameters(), b.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1 - m)


def dino_loss(model, s_out, t_out, t_temp, s_temp, center_m=0.9):
    """s_out: list over all crops. t_out: list over global crops only."""
    t = [F.softmax((o - model.center) / t_temp, dim=-1).detach() for o in t_out]
    loss, n = 0.0, 0
    for i, tt in enumerate(t):
        for j, so in enumerate(s_out):
            if i == j:                                   # skip the identical view
                continue
            loss = loss - (tt * F.log_softmax(so / s_temp, dim=-1)).sum(-1).mean()
            n += 1
    with torch.no_grad():
        batch_center = torch.cat(t_out).mean(0, keepdim=True)
        model.center.mul_(center_m).add_(batch_center, alpha=1 - center_m)
        p = torch.cat(t).mean(0)
        ent = -(p * (p + 1e-12).log()).sum()
        kl = np.log(model.n_proto) - ent.item()
        top1 = p.max().item()
    return loss / n, {"t_entropy": ent.item(), "kl_uniform": kl, "top1_share": top1}


def run(a):
    dev = torch.device(a.device)
    df = load_manifest()
    # SSL sees ONLY the training patients. Pretraining on test patients is
    # the most common silent leak in "self-supervised so labels don't matter".
    sp = make_split(df, "patient", seed=0)
    train_idx = sp.train
    cache = np.load(DATA / a.cache)["patches"]
    assert len(cache) == len(df), (len(cache), len(df))
    pool = torch.from_numpy(cache[train_idx])
    print(f"[dino] pretrain pool {len(pool)} patches from {df['patient_id'].values[train_idx].__len__()} "
          f"images / {len(set(df['patient_id'].values[train_idx]))} patients")

    model = DINO(img=a.img, patch=a.patch, d=a.d, depth=a.depth, heads=a.heads,
                 n_proto=a.protos).to(dev)
    g_aug = PathologyAug(a.img, scale=(0.4, 1.0), hed_sigma=a.hed).to(dev)
    l_aug = PathologyAug(a.img // 2, scale=(0.08, 0.4), hed_sigma=a.hed).to(dev)
    params = list(model.student.parameters()) + list(model.s_head.parameters())
    print(f"[dino] student params {sum(p.numel() for p in params) / 1e6:.2f}M  protos {a.protos}")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.04)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = cosine(step, a.steps, a.lr, a.lr * 0.01, warmup=a.steps // 20)
        for gp in opt.param_groups:
            gp["lr"] = lr
        t_temp = cosine(step, a.steps // 2, 0.04, 0.07) if step < a.steps // 2 else 0.07
        m = cosine(step, a.steps, a.momentum, 1.0)

        sel = torch.randint(0, len(pool), (a.batch,))
        u8 = pool[sel].to(dev)
        globals_ = [g_aug(u8) for _ in range(2)]
        locals_ = [l_aug(u8) for _ in range(a.n_local)]
        with torch.no_grad():
            t_out = [model.t_head(model.teacher(v)) for v in globals_]
        s_out = [model.s_head(model.student(v)) for v in globals_ + locals_]
        loss, mon = dino_loss(model, s_out, t_out, t_temp, a.s_temp)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 3.0)
        opt.step()
        model.ema(m)

        if step % 200 == 0 or step == 1:
            print(f"[{step:5d}] loss {loss.item():.4f} | t_ent {mon['t_entropy']:.3f}"
                  f"/{np.log(a.protos):.3f} kl_unif {mon['kl_uniform']:.3f} "
                  f"top1 {mon['top1_share']:.4f} | lr {lr:.2e} m {m:.4f} gn {gn:.2f} "
                  f"{time.time() - t0:6.1f}s", flush=True)
            if mon["t_entropy"] < 0.5 * np.log(a.protos) and step > a.steps // 4:
                print("[dino] WARNING teacher entropy below half of uniform -- collapsing")
        # run() is also called programmatically (frontier.py builds its own
        # Namespace), so tolerate a missing knob rather than crashing
        every = getattr(a, "ckpt_every", 250)
        if every and (step % every == 0 or step == a.steps):
            ck = ROOT / "ckpt" / a.out
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"backbone": model.teacher.state_dict(), "step": step,
                        "cfg": {"img": a.img, "patch": a.patch, "d": a.d,
                                "depth": a.depth, "heads": a.heads}}, ck)

    out = ROOT / "ckpt" / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"backbone": model.teacher.state_dict(),
                "cfg": {"img": a.img, "patch": a.patch, "d": a.d,
                        "depth": a.depth, "heads": a.heads}}, out)
    print(f"[dino] saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--img", type=int, default=96)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--protos", type=int, default=4096)
    ap.add_argument("--n-local", type=int, default=4)
    ap.add_argument("--s-temp", type=float, default=0.1)
    ap.add_argument("--hed", type=float, default=0.10)
    ap.add_argument("--momentum", type=float, default=0.996)
    ap.add_argument("--cache", default="patches96.npz")
    ap.add_argument("--out", default="dino.pt")
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--device", default="mps")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
