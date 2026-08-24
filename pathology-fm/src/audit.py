"""Frozen-embedding audit: what does the representation actually encode?

This is the pathology transfer of the parameter-free probe that caught our
protein structure tokenizer carrying only 23% of the SS8 entropy. Same idea,
different modality: never trust a pretraining loss, measure the representation
directly, and always probe the CONFOUNDS alongside the biology.

Three probes run on the SAME frozen embeddings:

  biology   -- benign vs malignant. What we want.
  patient   -- which patient this patch came from. A confound: if this is highly
               decodable the encoder has memorised slide appearance (stain lot,
               section thickness, scanner), which is exactly what fails to
               transfer to a new hospital.
  magnif    -- 40x/100x/200x/400x. A nuisance variable.

The headline number is not the biology AUC. It is the biology AUC together with
how much confound the embedding carries, and how far the biology AUC falls when
you move from a leaky random split to an honest patient-disjoint split.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pathdata import Split


# ------------------------------------------------------------ linear probing
def logistic_probe(Xtr, ytr, Xte, yte, n_cls, epochs=300, lr=0.1, wd=1e-4, device="cpu"):
    """Multinomial logistic regression, full-batch Adam. Deterministic, no sklearn."""
    dev = torch.device(device)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32, device=dev)
    Xte = torch.as_tensor(Xte, dtype=torch.float32, device=dev)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=dev)
    yte_t = torch.as_tensor(yte, dtype=torch.long, device=dev)

    W = torch.zeros(Xtr.shape[1], n_cls, device=dev, requires_grad=True)
    b = torch.zeros(n_cls, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(Xtr @ W + b, ytr_t).backward()
        opt.step()

    with torch.no_grad():
        logit = Xte @ W + b
        acc = (logit.argmax(1) == yte_t).float().mean().item()
        auc = roc_auc(logit.softmax(1)[:, 1].cpu().numpy(), yte) if n_cls == 2 else float("nan")
    chance = np.bincount(yte, minlength=n_cls).max() / len(yte)
    return {"acc": acc, "auc": auc, "chance": chance, "lift": acc / chance}


def roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC. Handles ties by average rank (Mann-Whitney U)."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ------------------------------------------------------------- the audit table
def audit_embedding(emb: np.ndarray, df, sp: Split, device="cpu") -> dict:
    """Run the biology probe and the confound probes on one frozen embedding."""
    y = df["y"].values
    mag = df["magnification"].values
    mag_cls = {m: i for i, m in enumerate(sorted(set(mag.tolist())))}
    mag_i = np.array([mag_cls[m] for m in mag])

    out = {}
    out["biology"] = logistic_probe(emb[sp.train], y[sp.train], emb[sp.test], y[sp.test],
                                    2, device=device)
    out["magnif"] = logistic_probe(emb[sp.train], mag_i[sp.train], emb[sp.test], mag_i[sp.test],
                                   len(mag_cls), device=device)

    # patient identity is only meaningful WITHIN a partition -- a patient-disjoint
    # split has no shared identities to decode. So we probe it inside train,
    # holding out patches: "can the encoder tell slides apart at all?"
    pats = df["patient_id"].values[sp.train]
    codes = {p: i for i, p in enumerate(sorted(set(pats)))}
    pid = np.array([codes[p] for p in pats])
    rs = np.random.default_rng(0).permutation(len(pid))
    cut = int(0.7 * len(rs))
    a, b = rs[:cut], rs[cut:]
    Etr = emb[sp.train]
    out["patient"] = logistic_probe(Etr[a], pid[a], Etr[b], pid[b], len(codes), device=device)
    return out


def format_audit(name: str, res: dict) -> str:
    b, m, p = res["biology"], res["magnif"], res["patient"]
    return (f"  {name:26s} bio-AUC {b['auc']:.3f}  bio-acc {b['acc']:.3f} "
            f"(chance {b['chance']:.3f})   || confounds: patient-ID {p['acc']:.3f} "
            f"(chance {p['chance']:.4f}, lift {p['lift']:6.1f}x)  magnif {m['acc']:.3f} "
            f"(chance {m['chance']:.3f})")


def leak_report(res_random: dict, res_patient: dict) -> str:
    """The number that matters: how much of the random-split score was leakage."""
    a, b = res_random["biology"]["auc"], res_patient["biology"]["auc"]
    return (f"    random-split AUC {a:.3f} -> patient-disjoint AUC {b:.3f}   "
            f"leak = {a - b:+.3f} AUC ({(a - b) / max(a - 0.5, 1e-6) * 100:+.1f}% of headroom)")
