"""S7: split-conformal calibration of the flow head's 90% intervals on held-out tissue blocks (V1 lymph node, stored CV outputs)."""
import json, numpy as np
rows = {"gauss_uncal": [], "conformal_per_gene": [], "conformal_global": [], "width_ratio_per_gene": [], "q_median": []}
for k in range(5):
    with np.load(f"st_cv_fold{k}.npz", allow_pickle=False) as z:
        b, y, mu, sd = z["blocks"], z["truth_log"], z["pred_phikon_log"], z["sd_phikon_log"] + 1e-6
    ub = np.sort(np.unique(b)); halves = (np.isin(b, ub[0::2]), np.isin(b, ub[1::2]))
    s = np.abs(y - mu) / sd
    for cal, ev in (halves, halves[::-1]):
        n = int(cal.sum()); lvl = min(1.0, np.ceil((n + 1) * 0.9) / n)
        q = np.quantile(s[cal], lvl, axis=0, method="higher"); qg = np.quantile(s[cal], min(1.0, np.ceil((s[cal].size + 1) * 0.9) / s[cal].size), method="higher")
        rows["gauss_uncal"].append(float((s[ev] <= 1.645).mean())); rows["conformal_per_gene"].append(float((s[ev] <= q[None]).mean()))
        rows["conformal_global"].append(float((s[ev] <= qg).mean())); rows["width_ratio_per_gene"].append(float(np.median(q) / 1.645)); rows["q_median"].append(float(np.median(q)))
out = {k: {"mean": round(float(np.mean(v)), 4), "min": round(float(np.min(v)), 4), "max": round(float(np.max(v)), 4)} for k, v in rows.items()}
out["note"] = "nominal 0.90; 10 evaluations = 5 folds x 2 halves of held-out blocks"
json.dump(out, open("st_conformal.json", "w"), indent=1); print(json.dumps(out))
