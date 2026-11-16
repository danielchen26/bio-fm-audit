"""Phase 4 summary: both training budgets chosen on a validation strip of the training slides, three replicates.
(A) test score of the four-fifths model at its selected budget; (B) the selected budget re-used with all training spots."""
import json, glob, os, sys, numpy as np
HEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hest")
TASKS = ["LUNG", "PAAD", "SKCM", "COAD", "READ", "LYMPH_IDC"]; REPS = [0, 1, 2]
P3 = {0: "results", 1: "results_s2", 2: "results_s3"}          # Phase 3 full-data STFlow runs: replicate r = seed r + 1


def rule(v, patience=20):                                       # the released rule, pointed at any curve
    best, wait, stop, arg = -1.0, 0, len(v), 0
    for i, x in enumerate(v):
        if x > best:
            best, wait, arg = x, 0, i
        else:
            wait += 1
            if wait >= patience:
                stop = i + 1; break
    return arg + 1, stop

out, miss = {}, []
for t in TASKS:
    O3 = {r: json.load(open(f"{HEST}/hest_flow_{t}_phikon_s{r}.json")) for r in REPS}; n = len(O3[0]["splits"]); sp = []
    OV = {}
    for r in REPS:
        f = f"{HEST}/hest_flowval_{t}_phikon_r{r}.json"
        OV[r] = {x["split"]: x for x in json.load(open(f))["splits"]} if os.path.exists(f) else {}
    for i in range(n):
        row = {"split": i, "ridge": round(O3[0]["splits"][i]["ridge"], 4), "reps": []}
        for r in REPS:
            f = glob.glob(f"results_val/rep{r}/*/{t}/split{i}/curve.json"); c = json.load(open(f[-1])) if f else None
            if c is None or len(c["test_pearson_per_epoch"]) < 100 or i not in OV[r] or "selected_budget" not in OV[r][i]:
                miss.append((t, i, r)); continue
            tc, vc = c["test_pearson_per_epoch"], c["validation_pearson_per_epoch"]; e, stop = rule(vc); o = OV[r][i]
            p3 = json.load(open(glob.glob(f"{P3[r]}/*/{t}/split{i}/curve.json")[0]))["test_pearson_per_epoch"]
            row["reps"].append({"rep": r,
                "stflow": {"selected_epoch": e, "A": round(tc[e - 1], 4), "B": round(p3[e - 1], 4), "selected_epoch_argmax100": int(np.argmax(vc)) + 1,
                           "A_argmax100": round(tc[int(np.argmax(vc))], 4), "hindsight": round(max(tc), 4), "hindsight_epoch": int(np.argmax(tc)) + 1,
                           "epoch100_four_fifths": round(tc[-1], 4), "phase3_epoch100": round(p3[-1], 4), "phase3_hindsight": round(max(p3), 4),
                           "test_curve": [round(float(x), 4) for x in tc], "val_curve": [round(float(x), 4) for x in vc]},
                "ours": {"selected_budget": o["selected_budget"], "A": o["test_at_selected"], "B": o["refit_all_spots"], "hindsight": max(o["test_by_budget"]),
                         "hindsight_budget": o["oracle_budget"], "steps4000_four_fifths": o["test_by_budget"][4], "phase3_steps4000": round(O3[r]["splits"][i]["flow"], 4),
                         "test_by_budget": o["test_by_budget"], "val_by_budget": o["val"]}})
        if len(row["reps"]) == len(REPS):
            for m in ("ours", "stflow"):
                for k in ("A", "B", "hindsight"):
                    row[f"{m}_{k}"] = round(float(np.mean([x[m][k] for x in row["reps"]])), 4)
            row["ours_default"] = round(float(np.mean([x["ours"]["phase3_steps4000"] for x in row["reps"]])), 4)
            row["stflow_default"] = round(float(np.mean([x["stflow"]["phase3_epoch100"] for x in row["reps"]])), 4)
        sp.append(row)
    out[t] = {"splits": sp}
if miss:
    print("incomplete:", len(miss), "of 45 split-replicates missing, e.g.", miss[:6]); json.dump(out, open("phase4_partial.json", "w")); sys.exit()
KEYS = ["ours_A", "stflow_A", "ours_B", "stflow_B", "ours_hindsight", "stflow_hindsight", "ours_default", "stflow_default", "ridge"]
for t in TASKS:
    for k in KEYS:
        out[t][k] = round(float(np.mean([x[k] for x in out[t]["splits"]])), 4)
    for m in ("ours", "stflow"):
        for k in ("A", "B"):
            out[t][f"{m}_{k}_by_rep"] = [round(float(np.mean([x["reps"][r][m][k] for x in out[t]["splits"]])), 4) for r in REPS]
rows = [x for t in TASKS for x in out[t]["splits"]]
tm = lambda k: float(np.mean([out[t][k] for t in TASKS])); br = lambda k: [round(float(np.mean([out[t][k + "_by_rep"][r] for t in TASKS])), 4) for r in REPS]


def read(a, b):
    d = tm(a) - tm(b); w = sum(int(x[a] > x[b]) for x in rows)
    v = "on par" if abs(d) < 0.01 else ("ahead" if d > 0 and w >= 10 else ("behind" if d < 0 and 15 - w >= 10 else "mixed"))
    return {"verdict": v, "six_organ_mean_diff": round(d, 4), "splits_ours_higher": w, "organs_ours_higher": sum(int(out[t][a] > out[t][b]) for t in TASKS),
            "per_organ": {t: round(out[t][a] - out[t][b], 4) for t in TASKS}}

allr = [x for row in rows for x in row["reps"]]
G = {"six_organ_mean": {k: round(tm(k), 4) for k in KEYS},
     "six_organ_mean_by_replicate": {k: br(k) for k in ("ours_A", "stflow_A", "ours_B", "stflow_B")},
     "primary_A_ours_vs_stflow": read("ours_A", "stflow_A"), "secondary_B_ours_vs_stflow": read("ours_B", "stflow_B"),
     "hindsight_ours_vs_stflow": read("ours_hindsight", "stflow_hindsight"),
     "regret_mean": {"ours": round(float(np.mean([x["ours"]["hindsight"] - x["ours"]["A"] for x in allr])), 4),
                     "stflow": round(float(np.mean([x["stflow"]["hindsight"] - x["stflow"]["A"] for x in allr])), 4)},
     "gain_from_selection_vs_default_all_spots": {"ours_B_minus_default": round(tm("ours_B") - tm("ours_default"), 4), "stflow_B_minus_default": round(tm("stflow_B") - tm("stflow_default"), 4)},
     "selected_budgets_ours": {str(b): int(sum(x["ours"]["selected_budget"] == b for x in allr)) for b in (250, 500, 1000, 2000, 4000, 8000)},
     "selected_epochs_stflow": {"median": float(np.median([x["stflow"]["selected_epoch"] for x in allr])), "min": int(min(x["stflow"]["selected_epoch"] for x in allr)),
                                "max": int(max(x["stflow"]["selected_epoch"] for x in allr)), "quartiles": [float(v) for v in np.percentile([x["stflow"]["selected_epoch"] for x in allr], [25, 75])]},
     "stflow_alt_rule_argmax_over_100_epochs_A": round(float(np.mean([np.mean([np.mean([x["stflow"]["A_argmax100"] for x in row["reps"]]) for row in out[t]["splits"]]) for t in TASKS])), 4),
     "vs_ridge_all_spots": {"ours_B": round(tm("ours_B") - tm("ridge"), 4), "stflow_B": round(tm("stflow_B") - tm("ridge"), 4),
                            "organs_ours_B_above": sum(int(out[t]["ours_B"] > out[t]["ridge"]) for t in TASKS), "organs_stflow_B_above": sum(int(out[t]["stflow_B"] > out[t]["ridge"]) for t in TASKS)}}
out["_overall"] = G
for t in TASKS:
    print(t, {k: out[t][k] for k in KEYS})
print(json.dumps(G, indent=1)); json.dump(out, open("phase4_summary.json", "w"), indent=1)
