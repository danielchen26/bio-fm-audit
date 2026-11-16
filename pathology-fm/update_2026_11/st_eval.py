"""Pre-registered within-slide evaluation with a ridge head (PREREG_phase1.md S1-S6): baselines incl. coordinates-only,
encoder ladder, noise ceiling by count thinning, block-size sensitivity. 5-fold spatial-block CV, folds as in st_cv.py."""
import sys, os, json, glob
sys.dont_write_bytecode = True
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, PROJ + "/src")
import numpy as np, torch
import xmodal_v2 as X
torch.set_num_threads(4)
sid = sys.argv[1]
with np.load(f"st_slide_{sid}.npz", allow_pickle=False) as z:
    expr, coords, genes = z["expr"].astype(np.float32), z["coords"].astype(np.float32), z["gene_names"].astype(str)
    counts, rest = z["counts"].astype(np.int64), z["rest"].astype(np.int64)
ENC = ["in_sup_vits", "in_dino_vits", "path_dino_vits_lunit", "in_sup_vitb", "in_dino_vitb", "path_dino_vitb_kaiko", "phikon", "path_dinov2_vitl_phikon2"]
feats = {k: np.load(f"st_feat_{sid}_{k}.npy") for k in ENC if os.path.exists(f"st_feat_{sid}_{k}.npy")}
if sid == "V1LN":
    with np.load(f"{PROJ}/data/morph_random_p224.npz", allow_pickle=False) as z:
        feats = {"random_conv": z["feat"], **feats}
n, G = expr.shape


def pearson(p, t):
    p, t = p - p.mean(0, keepdims=True), t - t.mean(0, keepdims=True)
    return (p * t).sum(0) / (np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0)) + 1e-12)


def make_folds(nb):
    _, bid = X.spatial_blocks(coords, nb, 0)
    return bid, np.array_split(np.random.default_rng(2026).permutation(np.unique(bid)), 5)


def pca_ridge(Ftr, Ytr, Fte, alphas):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    U, s, Vt = np.linalg.svd((Ftr - mu) / sd, full_matrices=False)
    p = min(256, int((s > 1e-6 * s[0]).sum())); U, s, Vt = U[:, :p], s[:p], Vt[:p]
    UY, Zte = U.T @ Ytr, ((Fte - mu) / sd) @ Vt.T
    return [Zte @ ((s / (s ** 2 + a))[:, None] * UY) for a in alphas]


def ridge_oof(F, bid, folds):
    pred, picked = np.zeros_like(expr), []
    for k in range(5):
        te = np.isin(bid, folds[k]); tr = ~te
        mu, sd = expr[tr].mean(0), expr[tr].std(0) + 1e-6
        Ytr = (expr[tr] - mu) / sd; ntr = int(tr.sum()); alphas = [ntr * a for a in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)]
        blocks = np.unique(bid[tr]); grp = np.array_split(np.random.default_rng(k).permutation(blocks), 3); score = np.zeros(len(alphas))
        for g in grp:                                           # inner CV grouped by tissue block
            va = np.isin(bid[tr], g)
            for i, pv in enumerate(pca_ridge(F[tr][~va], Ytr[~va], F[tr][va], alphas)):
                score[i] += np.median(pearson(pv, Ytr[va]))
        best = int(np.argmax(score)); picked.append(alphas[best] / ntr)
        pred[te] = pca_ridge(F[tr], Ytr, F[te], [alphas[best]])[0] * sd + mu
    return pred, picked


def coords_knn_oof(bid, folds, k=8):
    pred = np.zeros_like(expr)
    for f in range(5):
        te = np.isin(bid, folds[f]); tr = np.where(~te)[0]
        d = ((coords[te][:, None, :] - coords[tr][None, :, :]) ** 2).sum(-1)
        pred[te] = expr[tr][np.argsort(d, 1)[:, :k]].mean(1)
    return pred


def feat_knn_oof(F, bid, folds):
    pred = np.zeros_like(expr)
    for f in range(5):
        te = np.isin(bid, folds[f]); tr = ~te; mu, sd = expr[tr].mean(0), expr[tr].std(0) + 1e-6
        pred[te] = X.morphology_knn(F, (expr - mu) / sd, np.where(tr)[0], np.where(te)[0], 10).mean(1) * sd + mu
    return pred


def score(pred, bid, folds):
    r = pearson(pred, expr); fm, fmean, t50 = [], [], []
    for k in range(5):
        te = np.isin(bid, folds[k]); rk = pearson(pred[te], expr[te]); fm.append(float(np.median(rk))); fmean.append(float(np.mean(rk)))
        t50.append(float(np.mean(rk[np.argsort(expr[~te].var(0))[::-1][:50]])))
    return {"gene_r_mean": round(float(r.mean()), 4), "gene_r_median": round(float(np.median(r)), 4), "n_gt_0.5": int((r > 0.5).sum()),
            "n_gt_0.3": int((r > 0.3).sum()), "per_fold_median_r": [round(v, 4) for v in fm], "per_fold_mean_r": [round(v, 4) for v in fmean],
            "top50_hvg_r_per_fold": [round(v, 4) for v in t50], "top50_hvg_r": round(float(np.mean(t50)), 4),
            "mae_log": round(float(np.abs(pred - expr).mean()), 4)}, r

out = {"slide": sid, "spots": n, "genes": G, "encoders": list(feats)}
bid, folds = make_folds(6)
if sid == "V1LN":                                              # gate: identical folds to the stored cross-validation
    for k in range(5):
        with np.load(f"st_cv_fold{k}.npz", allow_pickle=False) as z:
            assert (np.sort(z["te_idx"]) == np.where(np.isin(bid, folds[k]))[0]).all()
    out["gate"] = "folds identical to st_cv_fold*.npz"
res, R, keep_pred = {}, {}, {}
p = coords_knn_oof(bid, folds); res["coords_only_knn"], R["coords_only_knn"] = score(p, bid, folds); keep_pred["coords_only_knn"] = p
if "phikon" in feats:
    p = feat_knn_oof(feats["phikon"], bid, folds); res["phikon_feature_knn"], R["phikon_feature_knn"] = score(p, bid, folds)
for kname, F in feats.items():
    p, picked = ridge_oof(F.astype(np.float64), bid, folds)
    res[f"ridge:{kname}"], R[f"ridge:{kname}"] = score(p, bid, folds); res[f"ridge:{kname}"]["alpha_over_n"] = picked
    if kname == "phikon":
        keep_pred["ridge_phikon"] = p
    print(f"[{sid}] ridge {kname:26s} median r {res[f'ridge:{kname}']['gene_r_median']:.3f} mean {res[f'ridge:{kname}']['gene_r_mean']:.3f} "
          f">0.5: {res[f'ridge:{kname}']['n_gt_0.5']:3d} top50 {res[f'ridge:{kname}']['top50_hvg_r']:.3f} per-fold median {res[f'ridge:{kname}']['per_fold_median_r']}", flush=True)
if sid == "V1LN":                                              # stored neighbourhood-Transformer + flow predictions, same folds
    for enc in ("phikon", "random"):
        p = np.zeros_like(expr)
        for k in range(5):
            with np.load(f"st_cv_fold{k}.npz", allow_pickle=False) as z:
                p[z["te_idx"]] = z[f"pred_{enc}_log"]
        res[f"flow:{enc}"], R[f"flow:{enc}"] = score(p, bid, folds)
for k in ("coords_only_knn", "phikon_feature_knn", "flow:phikon", "flow:random"):
    if k in res:
        print(f"[{sid}] {k:32s} median r {res[k]['gene_r_median']:.3f} mean {res[k]['gene_r_mean']:.3f} >0.5: {res[k]['n_gt_0.5']:3d} top50 {res[k]['top50_hvg_r']:.3f} per-fold median {res[k]['per_fold_median_r']}", flush=True)
out["blocks_6x6"] = res

# ---- S5 noise ceiling: binomial thinning of the raw counts, split-half reliability, Spearman-Brown
rng = np.random.default_rng(0); rh = np.zeros(G)
for _ in range(10):
    a = rng.binomial(counts, 0.5); b = counts - a; ra = rng.binomial(rest, 0.5); rb = rest - ra
    na = np.log1p(a / np.maximum(a.sum(1) + ra, 1)[:, None] * 1e4); nb = np.log1p(b / np.maximum(b.sum(1) + rb, 1)[:, None] * 1e4)
    rh += pearson(na, nb) / 10
ceil = np.sqrt(np.clip(2 * rh / (1 + rh), 1e-6, 1.0))
out["noise_ceiling"] = {"median_ceiling_r": round(float(np.median(ceil)), 4), "mean_ceiling_r": round(float(ceil.mean()), 4),
                        "genes_with_ceiling_gt_0.5": int((ceil > 0.5).sum()), "genes_with_ceiling_gt_0.8": int((ceil > 0.8).sum()),
                        "median_fraction_of_ceiling": {k: round(float(np.median(v / ceil)), 4) for k, v in R.items()},
                        "corr_gene_r_vs_ceiling": {k: round(float(np.corrcoef(v, ceil)[0, 1]), 3) for k, v in R.items()},
                        "median_r_among_genes_with_ceiling_gt_0.8": {k: round(float(np.median(v[ceil > 0.8])), 4) if (ceil > 0.8).any() else None
                                                                    for k, v in R.items()}}
print(f"[{sid}] noise ceiling", out["noise_ceiling"], flush=True)

# ---- S6 block-size sensitivity: 3 x 3 blocks (larger held-out regions, less spatial interpolation)
bid3, folds3 = make_folds(3); res3 = {}
res3["coords_only_knn"], _ = score(coords_knn_oof(bid3, folds3), bid3, folds3)
for kname in [k for k in ("in_dino_vitb", "phikon", "path_dino_vitb_kaiko") if k in feats]:
    res3[f"ridge:{kname}"], _ = score(ridge_oof(feats[kname].astype(np.float64), bid3, folds3)[0], bid3, folds3)
out["blocks_3x3"] = res3
print(f"[{sid}] 3x3 blocks:", {k: (v["gene_r_median"], v["per_fold_median_r"]) for k, v in res3.items()}, flush=True)
json.dump(out, open(f"st_eval_{sid}.json", "w"), indent=1)
np.savez_compressed(f"st_eval_{sid}.npz", ceiling=ceil.astype(np.float32), genes=genes, coords=coords, truth=expr,
                    **{f"r__{k.replace(':', '_')}": v.astype(np.float32) for k, v in R.items()}, **{f"pred__{k}": v.astype(np.float32) for k, v in keep_pred.items()})
print("done", flush=True)
