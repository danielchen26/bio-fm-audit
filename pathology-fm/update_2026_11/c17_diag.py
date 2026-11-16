"""CAMELYON17 leave-one-hospital-out, diagnosis only: composition per hospital, per-slide breakdown,
patient-level bootstrap intervals for Phikon minus ImageNet. No method is tuned here."""
import sys, json, glob, os
sys.dont_write_bytecode = True
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ + "/src")
import numpy as np, torch, torch.nn.functional as F
from audit import roc_auc
ROOT = os.environ.get("C17_TILESET", f"{PROJ}/data/camelyon17_tileset")   # tiles/<slide>.npz plus the two reference embeddings
GRADES = ("negative", "itc", "micro", "macro")
files = sorted(glob.glob(f"{ROOT}/tiles/*.npz"))
y, patient, slide, grade, centre, pats, sids = [], [], [], [], [], [], []
for si, f in enumerate(files):
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
emb = {"imagenet": np.load("c17_emb_imagenet.npy")[keep]}
with np.load(f"{ROOT}/ref_emb_phikon.npz", allow_pickle=False) as z:
    emb["phikon"] = z["emb"][keep]
y, patient, slide, grade, centre = y[keep], patient[keep], slide[keep], grade[keep], centre[keep]


def probe_scores(Xtr, ytr, Xte, epochs=300, lr=0.1, wd=1e-4):          # same recipe as audit.logistic_probe, returns P(tumour)
    Xtr = torch.as_tensor(Xtr, dtype=torch.float32); Xte = torch.as_tensor(Xte, dtype=torch.float32)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    yt = torch.as_tensor(ytr, dtype=torch.long)
    W = torch.zeros(Xtr.shape[1], 2, requires_grad=True); b = torch.zeros(2, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); F.cross_entropy(Xtr @ W + b, yt).backward(); opt.step()
    with torch.no_grad():
        return (Xte @ W + b).softmax(1)[:, 1].numpy()


def fast_auc(s, yy):
    if yy.min() == yy.max():
        return float("nan")
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    r = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1 = yy.sum(); n0 = len(yy) - n1
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

stored = json.load(open("c17_labeleff3.json"))
out = {"composition": {}, "loco": {}, "per_slide": {}, "bootstrap": {}}
score = {k: np.full(len(y), np.nan, np.float32) for k in emb}
for c in range(5):
    te = centre == c
    out["composition"][c] = {"slides": int(len(set(slide[te].tolist()))), "patients": int(len(set(patient[te].tolist()))),
                             "tiles": int(te.sum()), "tumour_tiles": int(y[te].sum()),
                             "slides_with_tumour": int(len(set(slide[te & (y == 1)].tolist()))),
                             "grades": {GRADES[g]: int(len(set(slide[te & (grade == g)].tolist()))) for g in (0, 2, 3)}}
    for k in emb:
        score[k][te] = probe_scores(emb[k][~te], y[~te], emb[k][te])
        a = fast_auc(score[k][te], y[te]); ref = roc_auc(score[k][te], y[te])
        assert abs(a - ref) < 1e-9 and abs(a - stored[k]["loco"][c]) < 1e-6, (k, c, a, ref, stored[k]["loco"][c])
        out["loco"].setdefault(k, []).append(round(a, 4))
    print(f"hospital {c}: {out['composition'][c]}  AUC phikon {out['loco']['phikon'][c]:.3f} imagenet {out['loco']['imagenet'][c]:.3f}", flush=True)
print("sanity: leave-one-hospital-out AUCs reproduce the stored values exactly", flush=True)

rng = np.random.default_rng(0)
for c in range(5):
    te = np.where(centre == c)[0]
    rows = []
    for s in sorted(set(slide[te].tolist())):
        m = te[slide[te] == s]
        rows.append({"slide": sids[s], "grade": GRADES[int(grade[m][0])], "tiles": int(len(m)), "tumour": int(y[m].sum()),
                     "auc_phikon": round(fast_auc(score["phikon"][m], y[m]), 3), "auc_imagenet": round(fast_auc(score["imagenet"][m], y[m]), 3),
                     "mean_p_normal_phikon": round(float(score["phikon"][m][y[m] == 0].mean()), 3) if (y[m] == 0).any() else None,
                     "mean_p_normal_imagenet": round(float(score["imagenet"][m][y[m] == 0].mean()), 3) if (y[m] == 0).any() else None})
    out["per_slide"][c] = rows
    by_pat = [te[patient[te] == p] for p in sorted(set(patient[te].tolist()))]
    d = []
    for _ in range(2000):
        idx = np.concatenate([by_pat[i] for i in rng.integers(0, len(by_pat), len(by_pat))])
        a, b = fast_auc(score["phikon"][idx], y[idx]), fast_auc(score["imagenet"][idx], y[idx])
        if np.isfinite(a) and np.isfinite(b):
            d.append(a - b)
    d = np.array(d)
    out["bootstrap"][c] = {"patients": len(by_pat), "diff": round(out["loco"]["phikon"][c] - out["loco"]["imagenet"][c], 4),
                           "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                           "frac_phikon_higher": round(float((d > 0).mean()), 3), "valid_resamples": int(len(d))}
    print(f"hospital {c}: phikon - imagenet {out['bootstrap'][c]}", flush=True)
by_pat = [np.where(patient == p)[0] for p in sorted(set(patient.tolist()))]
d, pa, pb = [], [], []
for _ in range(2000):
    idx = np.concatenate([by_pat[i] for i in rng.integers(0, len(by_pat), len(by_pat))])
    a, b = fast_auc(score["phikon"][idx], y[idx]), fast_auc(score["imagenet"][idx], y[idx])
    d.append(a - b); pa.append(a); pb.append(b)
d = np.array(d)
out["pooled"] = {"auc_phikon": round(fast_auc(score["phikon"], y), 4), "auc_imagenet": round(fast_auc(score["imagenet"], y), 4),
                 "diff_ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                 "phikon_ci95": [round(float(np.percentile(pa, 2.5)), 4), round(float(np.percentile(pa, 97.5)), 4)],
                 "imagenet_ci95": [round(float(np.percentile(pb, 2.5)), 4), round(float(np.percentile(pb, 97.5)), 4)]}
print("pooled over all tiles, each scored by the model that never saw its hospital:", out["pooled"], flush=True)
flip = [c for c in range(5) if out["loco"]["phikon"][c] < out["loco"]["imagenet"][c]]
for c in flip:
    print(f"per-slide breakdown, hospital {c}:")
    for r in out["per_slide"][c]:
        print("  ", r)
json.dump(out, open("c17_diag.json", "w"), indent=1)
print("done", flush=True)
