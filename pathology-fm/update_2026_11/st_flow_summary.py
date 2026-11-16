"""Head ladder: neighbourhood Transformer + flow (Phikon features) vs ridge on the same features and folds, three slides."""
import json, numpy as np


def pearson(p, t):
    p, t = p - p.mean(0, keepdims=True), t - t.mean(0, keepdims=True)
    return (p * t).sum(0) / (np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0)) + 1e-12)

out = {}
for sid in ("V1LN", "BRCA", "V2LN"):
    with np.load(f"st_slide_{sid}.npz", allow_pickle=False) as z:
        expr = z["expr"].astype(np.float32)
    ceil = np.load(f"st_eval_{sid}.npz")["ceiling"]; E = json.load(open(f"st_eval_{sid}.json"))["blocks_6x6"]
    pred = np.zeros_like(expr); fm, t50, crel, cov_raw, cov_conf = [], [], [], [], []; c_all, k_all = [], []
    for k in range(5):
        f = f"st_cv_fold{k}.npz" if sid == "V1LN" else f"st_cv2_{sid}_phikon_fold{k}.npz"
        with np.load(f, allow_pickle=False) as z:
            te, b, y = z["te_idx"], z["blocks"], z["truth_log"]
            mu, sd = (z["pred_phikon_log"], z["sd_phikon_log"]) if sid == "V1LN" else (z["pred_log"], z["sd_log"])
            c, cc = (z["crps_phikon"], z["crps_clim"]) if sid == "V1LN" else (z["crps"], z["crps_clim"])
        assert np.allclose(y, expr[te]); pred[te] = mu
        tr = np.ones(len(expr), bool); tr[te] = False
        rk = pearson(mu, y); fm.append(float(np.median(rk))); t50.append(float(np.mean(rk[np.argsort(expr[tr].var(0))[::-1][:50]])))
        crel.append(float(c.mean() / cc.mean() - 1)); c_all.append(c); k_all.append(cc)
        s = np.abs(y - mu) / (sd + 1e-6); ub = np.sort(np.unique(b)); halves = (np.isin(b, ub[0::2]), np.isin(b, ub[1::2]))
        for cal, ev in (halves, halves[::-1]):
            n = int(cal.sum()); q = np.quantile(s[cal], min(1.0, np.ceil((n + 1) * 0.9) / n), axis=0, method="higher")
            cov_raw.append(float((s[ev] <= 1.645).mean())); cov_conf.append(float((s[ev] <= q[None]).mean()))
    r = pearson(pred, expr); rp = E["ridge:phikon"]["per_fold_median_r"]; cp = E["coords_only_knn"]["per_fold_median_r"]
    out[sid] = {"gene_r_median": round(float(np.median(r)), 4), "gene_r_mean": round(float(r.mean()), 4), "n_gt_0.5": int((r > 0.5).sum()),
                "n_gt_0.3": int((r > 0.3).sum()), "per_fold_median_r": [round(v, 4) for v in fm], "top50_hvg_r": round(float(np.mean(t50)), 4),
                "median_fraction_of_ceiling": round(float(np.median(r / ceil)), 4),
                "crps_rel_vs_climatology": round(float(np.concatenate(c_all).mean() / np.concatenate(k_all).mean() - 1), 4),
                "crps_rel_per_fold": [round(v, 4) for v in crel],
                "ridge_same_features": {"gene_r_median": E["ridge:phikon"]["gene_r_median"], "per_fold_median_r": rp, "n_gt_0.5": E["ridge:phikon"]["n_gt_0.5"]},
                "folds_flow_above_ridge": int(sum(a > b for a, b in zip(fm, rp))), "folds_flow_above_coords_only": int(sum(a > b for a, b in zip(fm, cp))),
                "coverage90_uncalibrated": round(float(np.mean(cov_raw)), 4), "coverage90_conformal": round(float(np.mean(cov_conf)), 4),
                "coverage90_conformal_range": [round(float(np.min(cov_conf)), 4), round(float(np.max(cov_conf)), 4)]}
    np.save(f"st_flow_gene_r_{sid}.npy", r.astype(np.float32))
    print(sid, json.dumps(out[sid]), flush=True)
json.dump(out, open("st_flow_summary.json", "w"), indent=1)
