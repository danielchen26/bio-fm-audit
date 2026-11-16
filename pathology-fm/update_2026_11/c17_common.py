"""Shared loaders and probe for the CAMELYON17 grid (recipe identical to audit.logistic_probe)."""
import sys, glob, os, json
sys.dont_write_bytecode = True
import numpy as np, torch, torch.nn.functional as F
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("C17_TILESET", f"{PROJ}/data/camelyon17_tileset")   # tiles/<slide>.npz plus the two reference embeddings
HERE = os.path.dirname(os.path.abspath(__file__))
GRADES = ("negative", "itc", "micro", "macro")
STORED = {"phikon": "ref_emb_phikon",
          "raw": "ref_emb_raw"}


def load_meta():
    y, patient, slide, grade, centre, pats, sids = [], [], [], [], [], [], []
    for si, f in enumerate(sorted(glob.glob(f"{ROOT}/tiles/*.npz"))):
        with np.load(f, allow_pickle=False) as z:
            lab, pid = z["labels"], str(z["patient_id"])
            if pid not in pats:
                pats.append(pid)
            sids.append(str(z["slide_id"]) if "slide_id" in z.files else os.path.basename(f)[:-4])
            n = len(lab)
            y.append(lab); patient.append(np.full(n, pats.index(pid))); slide.append(np.full(n, si))
            grade.append(np.full(n, GRADES.index(str(z["grade"])))); centre.append(np.full(n, int(z["centre"])))
    y, patient, slide, grade, centre = map(np.concatenate, (y, patient, slide, grade, centre))
    keep = np.isin(grade, [0, 2, 3])
    return {"keep": keep, "y": y[keep].astype(np.int64), "patient": patient[keep], "slide": slide[keep], "grade": grade[keep],
            "centre": centre[keep], "sids": sids, "pats": pats}


def load_emb(name, keep):
    if name in STORED:
        with np.load(f"{ROOT}/{STORED[name]}.npz", allow_pickle=False) as z:
            return z["emb"][keep].astype(np.float32)
    return np.load(f"{HERE}/c17_emb_{name}.npy")[keep].astype(np.float32)


def group_split(key, seed, frac=0.25):
    g = np.unique(key); pick = np.random.default_rng(seed).permutation(g)
    m = np.isin(key, pick[:max(1, int(len(g) * frac))])
    return np.where(~m)[0], np.where(m)[0]


def probe_scores(Xtr, ytr, Xte, w=None, epochs=300, lr=0.1, wd=1e-4):
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32); Xte = torch.as_tensor(Xte, dtype=torch.float32)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    yt = torch.as_tensor(ytr, dtype=torch.long)
    wt = None if w is None else torch.as_tensor(w, dtype=torch.float32)
    W = torch.zeros(Xtr.shape[1], 2, requires_grad=True); b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        if wt is None:
            loss = F.cross_entropy(Xtr @ W + b, yt)
        else:
            loss = (F.cross_entropy(Xtr @ W + b, yt, reduction="none") * wt).sum() / wt.sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        return (Xte @ W + b).softmax(1)[:, 1].numpy()


def fast_auc(s, yy):
    yy = np.asarray(yy)
    if yy.min() == yy.max():
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = yy.sum(); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
