"""Weakly supervised patient-level classification (gated-attention MIL).

This is the "weakly supervised" line in the JD. Real pathology labels arrive at
the level of a patient or a slide, never per patch: nobody annotates every cell.
A bag is one patient, instances are that patient's patches, and the label is
attached to the bag only.

Statistical honesty note: BreakHis has 82 patients. A 20% test split is ~16
patients, so a single patient-level AUC is close to meaningless. Everything here
is reported as mean +- std over repeated seeds, and the number of test patients
is printed next to it so the reader can see how thin the evidence is.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audit import roc_auc
from pathdata import make_split


class GatedAttentionMIL(nn.Module):
    """Ilse et al. gated attention pooling over instance embeddings."""

    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(dropout))
        self.V = nn.Linear(d_hidden, d_hidden)
        self.U = nn.Linear(d_hidden, d_hidden)
        self.w = nn.Linear(d_hidden, 1)
        self.head = nn.Linear(d_hidden, 2)

    def forward(self, bag):                              # bag: [n_instances, d_in]
        h = self.proj(bag)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))   # [n,1]
        a = a.softmax(0)
        return self.head((a * h).sum(0)), a.squeeze(-1)


def bags_for(df, emb, indices):
    """Group the given rows into per-patient bags."""
    pats = df["patient_id"].values[indices]
    y = df["y"].values[indices]
    out = []
    for p in sorted(set(pats)):
        m = pats == p
        labels = set(y[m].tolist())
        assert len(labels) == 1, f"patient {p} has mixed labels {labels}"
        out.append((p, torch.from_numpy(emb[indices][m]).float(), int(labels.pop())))
    return out


def train_mil(df, emb, regime: str, seeds=(0, 1, 2, 3, 4), epochs=60, lr=1e-3,
              device="cpu", verbose=False):
    dev = torch.device(device)
    aucs, accs, n_test = [], [], []
    for seed in seeds:
        sp = make_split(df, regime, seed=seed)
        tr = bags_for(df, emb, sp.train)
        te = bags_for(df, emb, sp.test)
        if len({b[2] for b in te}) < 2:
            continue                                     # degenerate test fold
        model = GatedAttentionMIL(emb.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            model.train()
            for i in rng.permutation(len(tr)):
                _, bag, label = tr[i]
                logit, _ = model(bag.to(dev))
                loss = F.cross_entropy(logit[None], torch.tensor([label], device=dev))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            score, truth = [], []
            for _, bag, label in te:
                logit, _ = model(bag.to(dev))
                score.append(logit.softmax(-1)[1].item())
                truth.append(label)
        score, truth = np.array(score), np.array(truth)
        aucs.append(roc_auc(score, truth))
        accs.append(float(((score > 0.5).astype(int) == truth).mean()))
        n_test.append(len(te))
        if verbose:
            print(f"    seed {seed}: {len(tr)} train bags / {len(te)} test bags "
                  f"AUC {aucs[-1]:.3f} acc {accs[-1]:.3f}")
    return {"auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "n_test_patients": float(np.mean(n_test)), "n_seeds": len(aucs)}


def format_mil(name: str, r: dict) -> str:
    return (f"  {name:26s} patient-AUC {r['auc_mean']:.3f} +- {r['auc_std']:.3f}   "
            f"acc {r['acc_mean']:.3f} +- {r['acc_std']:.3f}   "
            f"(n_test_patients~{r['n_test_patients']:.0f}, {r['n_seeds']} seeds)")
