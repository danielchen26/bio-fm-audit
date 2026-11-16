"""Three seeds per method. Public STFlow code (seeds 1-3; test curve per epoch) vs our flow model (seeds 0-2) vs official ridge,
identical Phikon features and official HEST splits. Reading rule of the pre-registration applied to seed-averaged values."""
import json, glob, os, numpy as np
HEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hest")
TASKS = ["LUNG", "PAAD", "SKCM", "COAD", "READ", "LYMPH_IDC"]
SDIR = {1: "results", 2: "results_s2", 3: "results_s3"}; OSEED = [0, 1, 2]


def released_rule(c, patience=20):
    best, wait, stop = -1.0, 0, len(c)
    for i, v in enumerate(c):
        if v > best:
            best, wait = v, 0
        else:
            wait += 1
            if wait >= patience:
                stop = i + 1; break
    return best, stop

out = {}
for t in TASKS:
    O = {s: json.load(open(f"{HEST}/hest_flow_{t}_phikon_s{s}.json")) for s in OSEED}; n = len(O[0]["splits"]); sp = []
    for i in range(n):
        C = {}
        for s, d in SDIR.items():
            f = glob.glob(f"{d}/*/{t}/split{i}/curve.json"); assert len(f) == 1, (t, i, s, f)
            C[s] = json.load(open(f[0]))["test_pearson_per_epoch"]; assert len(C[s]) == 100
        rel = {s: released_rule(C[s]) for s in C}
        ridge = [O[s]["splits"][i]["ridge"] for s in OSEED]; assert max(ridge) - min(ridge) < 1e-6
        sp.append({"split": i, "test": O[0]["splits"][i]["test"], "ridge": round(ridge[0], 4),
                   "ours_by_seed": [round(O[s]["splits"][i]["flow"], 4) for s in OSEED],
                   "stflow_epoch100_by_seed": [round(C[s][-1], 4) for s in C],
                   "stflow_released_by_seed": [round(rel[s][0], 4) for s in C], "released_stop_epoch_by_seed": [rel[s][1] for s in C],
                   "stflow_best_epoch_by_seed": [int(np.argmax(C[s][:rel[s][1]])) + 1 for s in C],
                   "curves": {str(s): [round(float(v), 4) for v in C[s]] for s in C}})
        for k in ("ours", "stflow_epoch100", "stflow_released"):
            sp[-1][k] = round(float(np.mean(sp[-1][k + "_by_seed"])), 4)
    out[t] = {"splits": sp}
    for k in ("ours", "stflow_epoch100", "stflow_released", "ridge"):
        out[t][k] = round(float(np.mean([x[k] for x in sp])), 4)
    for k in ("ours", "stflow_epoch100", "stflow_released"):
        out[t][k + "_by_seed"] = [round(float(np.mean([x[k + "_by_seed"][j] for x in sp])), 4) for j in range(3)]
    print(t, {k: out[t][k] for k in ("ridge", "ours", "stflow_epoch100", "stflow_released")}, "| ours seeds", out[t]["ours_by_seed"], "| stflow e100 seeds", out[t]["stflow_epoch100_by_seed"])
rows = [x for t in TASKS for x in out[t]["splits"]]
tm = lambda k: float(np.mean([out[t][k] for t in TASKS]))
bs = lambda k: [round(float(np.mean([out[t][k + "_by_seed"][j] for t in TASKS])), 4) for j in range(3)]
d = tm("ours") - tm("stflow_epoch100"); w = sum(int(x["ours"] > x["stflow_epoch100"]) for x in rows)
verdict = "on par" if abs(d) < 0.01 else ("ahead" if d > 0 and w >= 10 else ("behind" if d < 0 and 15 - w >= 10 else "mixed"))
curve = np.mean([np.mean([np.mean([x["curves"][s] for s in ("1", "2", "3")], 0) for x in out[t]["splits"]], 0) for t in TASKS], 0)
curve_seed = {s: np.mean([np.mean([x["curves"][s] for x in out[t]["splits"]], 0) for t in TASKS], 0) for s in ("1", "2", "3")}
ab = np.where(curve > tm("ours"))[0]
pairs = [(a, b) for a in bs("ours") for b in bs("stflow_epoch100")]
out["_overall"] = {
    "seeds": {"ours": OSEED, "stflow": [1, 2, 3]}, "verdict_preregistered_rule_on_seed_means": verdict,
    "six_organ_mean": {k: round(tm(k), 4) for k in ("ours", "stflow_epoch100", "stflow_released", "ridge")},
    "six_organ_mean_by_seed": {k: bs(k) for k in ("ours", "stflow_epoch100", "stflow_released")},
    "six_organ_seed_sd": {k: round(float(np.std(bs(k), ddof=1)), 4) for k in ("ours", "stflow_epoch100", "stflow_released")},
    "ours_minus_stflow_epoch100": {"mean": round(d, 4), "splits_ours_higher": w, "organs_ours_higher": sum(int(out[t]["ours"] > out[t]["stflow_epoch100"]) for t in TASKS),
                                   "per_organ": {t: round(out[t]["ours"] - out[t]["stflow_epoch100"], 4) for t in TASKS},
                                   "seed_pairings_ours_higher": sum(int(a > b) for a, b in pairs), "seed_pairings": len(pairs),
                                   "range_over_seed_pairings": [round(min(a - b for a, b in pairs), 4), round(max(a - b for a, b in pairs), 4)]},
    "ours_minus_stflow_released": {"mean": round(tm("ours") - tm("stflow_released"), 4), "splits_ours_higher": sum(int(x["ours"] > x["stflow_released"]) for x in rows)},
    "selection_effect": {"mean_over_organs": round(tm("stflow_released") - tm("stflow_epoch100"), 4),
                         "by_seed": [round(a - b, 4) for a, b in zip(bs("stflow_released"), bs("stflow_epoch100"))],
                         "per_organ": {t: round(out[t]["stflow_released"] - out[t]["stflow_epoch100"], 4) for t in TASKS},
                         "largest_single_run": round(max(a - b for x in rows for a, b in zip(x["stflow_released_by_seed"], x["stflow_epoch100_by_seed"])), 4)},
    "budget_sensitivity_post_hoc": {"stflow_six_organ_mean_by_epoch": [round(float(v), 4) for v in curve], "peak_value": round(float(curve.max()), 4), "peak_epoch": int(curve.argmax()) + 1,
                                    "peak_by_seed": {s: [round(float(curve_seed[s].max()), 4), int(curve_seed[s].argmax()) + 1] for s in curve_seed},
                                    "epochs_stflow_mean_above_ours": [int(ab.min()) + 1, int(ab.max()) + 1] if len(ab) else None, "n_epochs_above": int(len(ab))},
    "vs_ridge": {"ours": round(tm("ours") - tm("ridge"), 4), "stflow_epoch100": round(tm("stflow_epoch100") - tm("ridge"), 4),
                 "splits_ours_above_ridge": sum(int(x["ours"] > x["ridge"]) for x in rows), "splits_stflow_epoch100_above_ridge": sum(int(x["stflow_epoch100"] > x["ridge"]) for x in rows),
                 "organs_ours_above_ridge": sum(int(out[t]["ours"] > out[t]["ridge"]) for t in TASKS), "organs_stflow_above_ridge": sum(int(out[t]["stflow_epoch100"] > out[t]["ridge"]) for t in TASKS)}}
o = dict(out["_overall"]); o["budget_sensitivity_post_hoc"] = {k: v for k, v in o["budget_sensitivity_post_hoc"].items() if k != "stflow_six_organ_mean_by_epoch"}
print(json.dumps(o, indent=1))
json.dump(out, open("h2h_seeds_summary.json", "w"), indent=1)
