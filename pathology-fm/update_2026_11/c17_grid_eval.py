"""Pre-registered CAMELYON17 evaluation for one encoder: E1 label efficiency, E2 all-patient out-of-fold,
E3 leave-one-hospital-out, E4 hospital-shift methods with nested selection, E5 per-slide AUC. See PREREG_phase1.md."""
import sys, json, time
from c17_common import *
name = sys.argv[1]; torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
M = load_meta(); y, patient, slide, centre, grade = M["y"], M["patient"], M["slide"], M["centre"], M["grade"]
E = load_emb(name, M["keep"]); N = len(y); t0 = time.time()
out = {"encoder": name, "dim": int(E.shape[1]), "tiles": int(N)}
stored = json.load(open(f"{HERE}/c17_labeleff3.json"))
auc = lambda tr, te: fast_auc(probe_scores(E[tr], y[tr], E[te]), y[te])

# ---- E1 label efficiency (same subsets as c17_labeleff3.py)
NS = (1, 2, 4, 8, 16, "all"); le = {str(n): [] for n in NS}
for s in range(5):
    tr, te = group_split(patient, s); pool = np.unique(patient[tr])
    for n in NS:
        vals = []
        for rep in range(1 if n == "all" else 4):
            rng = np.random.default_rng(1000 * s + 10 * rep + (0 if n == "all" else n))
            for _ in range(50):
                pick = pool if n == "all" else rng.choice(pool, n, replace=False)
                sub = tr[np.isin(patient[tr], pick)]
                if len(set(y[sub].tolist())) == 2:
                    break
            vals.append(auc(sub, te))
        le[str(n)].append(float(np.mean(vals)))
out["E1_label_eff"] = le
if name in ("phikon", "raw"):
    for n in NS:
        assert np.allclose(le[str(n)], stored[name]["label_eff"][str(n)], atol=1e-6), (n, le[str(n)], stored[name]["label_eff"][str(n)])
    print(f"[{name}] sanity: E1 reproduces the stored label-efficiency row exactly", flush=True)
print(f"[{name}] E1 means", {n: round(float(np.mean(v)), 4) for n, v in le.items()}, f"{time.time() - t0:.0f}s", flush=True)

# ---- E2 all-patient out-of-fold, 5 repeats of grouped 5-fold
pats = np.unique(patient); oof = np.zeros((5, N), np.float32); fold_mean = []
for r in range(5):
    perm = np.random.default_rng(100 + r).permutation(pats); fa = []
    for k in range(5):
        m = np.isin(patient, perm[k::5]); tr, te = np.where(~m)[0], np.where(m)[0]
        oof[r, te] = probe_scores(E[tr], y[tr], E[te]); fa.append(fast_auc(oof[r, te], y[te]))
    fold_mean.append(float(np.mean(fa)))
out["E2_pooled_oof_auc_per_repeat"] = [fast_auc(oof[r], y) for r in range(5)]
out["E2_fold_mean_auc_per_repeat"] = fold_mean
print(f"[{name}] E2 pooled {np.round(out['E2_pooled_oof_auc_per_repeat'], 4).tolist()} fold-mean {np.round(fold_mean, 4).tolist()} {time.time() - t0:.0f}s", flush=True)

# ---- E3 / E4 leave one hospital out, four methods, nested selection
METH = ("M0", "M1", "M2", "M3")


def prep(method, tr, te):
    Xtr, Xte, w = E[tr], E[te], None
    if method == "M1":
        Xtr = Xtr.copy()
        for h in np.unique(centre[tr]):
            m = centre[tr] == h; Xtr[m] = (Xtr[m] - Xtr[m].mean(0)) / (Xtr[m].std(0) + 1e-6)
        Xte = (Xte - Xte.mean(0)) / (Xte.std(0) + 1e-6)
    elif method == "M2":
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        hm = np.stack([0.5 * (Xtr[(centre[tr] == h) & (y[tr] == 0)].mean(0) + Xtr[(centre[tr] == h) & (y[tr] == 1)].mean(0))
                       for h in np.unique(centre[tr])])
        hm = hm - hm.mean(0); _, sv, Vt = np.linalg.svd(hm, full_matrices=False); Q = Vt[sv > 1e-6 * sv.max()]
        Xtr, Xte = Xtr - (Xtr @ Q.T) @ Q, Xte - (Xte @ Q.T) @ Q
    elif method == "M3":
        cell = centre[tr] * 2 + y[tr]; cnt = np.bincount(cell, minlength=10).astype(float)
        w = 1.0 / cnt[cell]; w = w / w.sum() * len(w)
    return Xtr, Xte, w

sc = {m: np.zeros(N, np.float32) for m in METH}; sel = np.zeros(N, np.float32); chosen = []
for c in range(5):
    te, tr = np.where(centre == c)[0], np.where(centre != c)[0]
    for m in METH:
        Xtr, Xte, w = prep(m, tr, te); sc[m][te] = probe_scores(Xtr, y[tr], Xte, w)
    inner = {m: np.zeros(N, np.float32) for m in METH}
    for v in [h for h in range(5) if h != c]:
        te2, tr2 = np.where(centre == v)[0], np.where((centre != c) & (centre != v))[0]
        for m in METH:
            Xtr, Xte, w = prep(m, tr2, te2); inner[m][te2] = probe_scores(Xtr, y[tr2], Xte, w)
    pooled = {m: fast_auc(inner[m][tr], y[tr]) for m in METH}
    best = max(METH, key=lambda m: pooled[m]); sel[te] = sc[best][te]
    chosen.append({"held_out": c, "inner_pooled_auc": {m: round(v, 4) for m, v in pooled.items()}, "selected": best})
    print(f"[{name}] hospital {c} done, selected {best} {time.time() - t0:.0f}s", flush=True)


def summarise(s):
    per = [fast_auc(s[centre == c], y[centre == c]) for c in range(5)]
    pred = s >= 0.5
    return {"per_hospital_auc": [round(a, 4) for a in per], "mean_auc": round(float(np.mean(per)), 4), "worst_auc": round(float(np.min(per)), 4),
            "pooled_auc": round(fast_auc(s, y), 4),
            "sens_at_0.5": round(float(pred[y == 1].mean()), 4), "spec_at_0.5": round(float((~pred[y == 0]).mean()), 4),
            "per_hospital_sens": [round(float(pred[(centre == c) & (y == 1)].mean()), 3) for c in range(5)],
            "per_hospital_spec": [round(float((~pred[(centre == c) & (y == 0)]).mean()), 3) for c in range(5)]}

out["E3_E4_loco"] = {m: summarise(sc[m]) for m in METH}; out["E3_E4_loco"]["selected"] = summarise(sel)
out["E4_selection"] = chosen
if name in ("phikon", "raw"):
    assert np.allclose(out["E3_E4_loco"]["M0"]["per_hospital_auc"], np.round(stored[name]["loco"], 4), atol=1e-4)
    print(f"[{name}] sanity: E3 reproduces the stored leave-one-hospital-out row", flush=True)

# ---- E5 per-slide AUC under leave-one-hospital-out (no correction)
ps = {}
for s in np.unique(slide):
    m = slide == s
    ps[M["sids"][int(s)]] = {"hospital": int(centre[m][0]), "grade": GRADES[int(grade[m][0])], "tiles": int(m.sum()), "tumour": int(y[m].sum()),
                             "auc": None if y[m].min() == y[m].max() else round(fast_auc(sc["M0"][m], y[m]), 4),
                             "mean_p_normal": round(float(sc["M0"][m][y[m] == 0].mean()), 4) if (y[m] == 0).any() else None,
                             "mean_p_tumour": round(float(sc["M0"][m][y[m] == 1].mean()), 4) if (y[m] == 1).any() else None}
out["E5_per_slide"] = ps; out["seconds"] = round(time.time() - t0)
np.savez_compressed(f"{HERE}/c17_grid_scores_{name}.npz", oof=oof, sel=sel, **{f"loco_{m}": sc[m] for m in METH})
json.dump(out, open(f"{HERE}/c17_grid_{name}.json", "w"), indent=1)
for m in list(METH) + ["selected"]:
    r = out["E3_E4_loco"][m]
    print(f"[{name}] {m:8s} per-hospital {r['per_hospital_auc']} mean {r['mean_auc']} worst {r['worst_auc']} pooled {r['pooled_auc']} sens {r['sens_at_0.5']} spec {r['spec_at_0.5']}", flush=True)
print(f"[{name}] done in {time.time() - t0:.0f}s", flush=True)
