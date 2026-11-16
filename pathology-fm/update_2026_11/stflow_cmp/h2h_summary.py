"""Head-to-head on identical Phikon features and official HEST splits: public STFlow code vs our flow model vs the official ridge probe.
STFlow is read from the per-epoch test curve: (a) released rule = best test epoch with patience 20, (b) fixed budget = last epoch."""
import json, glob, os, sys, numpy as np
HEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hest")
TASKS = ["LUNG", "PAAD", "SKCM", "COAD", "READ", "LYMPH_IDC"]
TAG = sys.argv[1] if len(sys.argv) > 1 else "results"


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

out, rows = {}, []
for t in TASKS:
    ours = json.load(open(f"{HEST}/hest_flow_{t}_phikon_s0.json"))
    files = sorted(glob.glob(f"{TAG}/*/{t}/split*/curve.json"))
    sp = []
    for f in files:
        i = int(os.path.basename(os.path.dirname(f))[5:]); c = json.load(open(f))["test_pearson_per_epoch"]
        if len(c) < 100:
            continue
        a, stop = released_rule(c)
        s = ours["splits"][i]
        sp.append({"split": i, "stflow_released_rule": round(a, 4), "released_rule_stop_epoch": stop, "stflow_epoch100": round(c[-1], 4),
                   "stflow_best_of_100": round(max(c), 4), "stflow_mean_last10": round(float(np.mean(c[-10:])), 4),
                   "ours_flow": round(s["flow"], 4), "ridge": round(s["ridge"], 4), "test": s["test"], "n_test": s["n_test"],
                   "curve": [round(float(v), 4) for v in c]})
    if len(sp) != len(ours["splits"]):
        print(t, "incomplete:", len(sp), "of", len(ours["splits"])); out.setdefault("_partial", {})[t] = sp; continue
    m = lambda k: round(float(np.mean([x[k] for x in sp])), 4)
    out[t] = {"splits": sp, "mean": {k: m(k) for k in ("stflow_released_rule", "stflow_epoch100", "stflow_best_of_100", "stflow_mean_last10", "ours_flow", "ridge")}}
    rows += [(t, x) for x in sp]
    print(t, out[t]["mean"])
if rows:
    d = np.array([x["ours_flow"] - x["stflow_epoch100"] for _, x in rows]); e = np.array([x["ours_flow"] - x["stflow_released_rule"] for _, x in rows])
    sel = np.array([x["stflow_released_rule"] - x["stflow_epoch100"] for _, x in rows])
    done = [t for t in out if not t.startswith("_")]
    tm = lambda k: float(np.mean([out[t]["mean"][k] for t in done]))
    out["_overall"] = {"tasks": done, "n_splits": len(rows),
                       "task_mean": {k: round(tm(k), 4) for k in ("ours_flow", "stflow_epoch100", "stflow_released_rule", "stflow_mean_last10", "ridge")},
                       "ours_minus_stflow_epoch100": {"task_mean_diff": round(tm("ours_flow") - tm("stflow_epoch100"), 4), "splits_ours_higher": int((d > 0).sum())},
                       "ours_minus_stflow_released_rule": {"task_mean_diff": round(tm("ours_flow") - tm("stflow_released_rule"), 4), "splits_ours_higher": int((e > 0).sum())},
                       "selection_effect_released_minus_epoch100": {"mean_over_splits": round(float(sel.mean()), 4), "max": round(float(sel.max()), 4), "min": round(float(sel.min()), 4)},
                       "stflow_epoch100_minus_ridge": {"task_mean_diff": round(tm("stflow_epoch100") - tm("ridge"), 4),
                                                       "splits_stflow_higher": int(sum(x["stflow_epoch100"] > x["ridge"] for _, x in rows))}}
    print(json.dumps(out["_overall"], indent=1))
json.dump(out, open(f"h2h_summary_{os.path.basename(TAG)}.json", "w"), indent=1)
