"""Phase 4, our model: leak-free choice of the training budget. A contiguous strip of every training slide (val_strip.py) is held out;
each budget is a full run with its own one-cycle schedule; the budget with the best validation score is selected and the test patients
are read at that budget. Recipe identical to hest_flow.py; --mode full is hest_flow.py itself (all training spots)."""
import sys, json, time, argparse, os
sys.dont_write_bytecode = True
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, PROJ + "/src"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import xmodal_v2 as X
from hest_common import *
from val_strip import val_strip
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--enc", default="phikon")
ap.add_argument("--rep", type=int, required=True)            # replicate = seed, and picks the validation strip
ap.add_argument("--budgets", default="250,500,1000,2000,4000,8000"); ap.add_argument("--mode", default="val", choices=["val", "full"])
ap.add_argument("--splits", default=""); ap.add_argument("--tag", default="")
a = ap.parse_args(); seed = a.rep; BUD = [int(b) for b in a.budgets.split(",")]
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
gl = genes(a.task); sp_ = splits(a.task); S = {}
for s in sorted({s for tr, te in sp_ for s in tr + te}):
    d = load_sample(a.task, s, gl); d["feat"] = emb(a.task, s, a.enc).astype(np.float32)
    c = d["coords"]; nn = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1) + np.eye(len(c)) * 1e18).min(1)
    d["c"] = (c / np.median(nn)).astype(np.float32); d["nbr"] = X.knn_graph(d["c"], X.K_NEIGH)
    va = val_strip(d["coords"], a.rep); nb = np.zeros_like(d["nbr"])
    for part in (np.where(~va)[0], np.where(va)[0]):         # the two parts of a training slide never see each other
        nb[part] = part[X.knn_graph(d["c"][part], X.K_NEIGH)]
    d["va"], d["nbr_val"] = va, nb; S[s] = d


def assemble(i, mode):
    tr, te = sp_[i]; order = tr + te; n = [len(S[s]["feat"]) for s in order]; off = np.cumsum([0] + n)
    D = {"morph": np.concatenate([S[s]["feat"] for s in order]), "y": np.concatenate([S[s]["y"] for s in order]).astype(np.float32),
         "coords": np.concatenate([S[s]["c"] for s in order])}
    role = [np.full(n[j], 2, np.int8) if j >= len(tr) else (S[s]["va"].astype(np.int8) if mode == "val" else np.zeros(n[j], np.int8)) for j, s in enumerate(order)]
    D["nbr"] = np.concatenate([(S[s]["nbr_val"] if (mode == "val" and j < len(tr)) else S[s]["nbr"]) + off[j] for j, s in enumerate(order)])
    role = np.concatenate(role); D["observed"] = role == 0
    D["tr_idx"], D["va_idx"], D["te_idx"] = np.where(role == 0)[0], np.where(role == 1)[0], np.where(role == 2)[0]
    D["exprz"], _, _ = X.normalize_expression(D["y"], D["observed"])
    for idx in (D["tr_idx"], D["va_idx"]):                   # no edge may leave its own part
        if len(idx):
            assert np.isin(D["nbr"][idx], idx).all()
    return D


def fit(D, steps, i):
    torch.manual_seed(seed * 100 + i)
    bags = X.Bags(D["morph"], D["exprz"], D["coords"], D["nbr"], 12.0, observed=D["observed"])
    model = X.CrossModal(len(gl), D["morph"].shape[1], d=256, layers=4, head="flow").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.1)
    rg = np.random.default_rng(seed * 100 + i)
    for _ in range(steps):
        loss = X.training_loss(model, bags, rg.choice(D["tr_idx"], 64), dev)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return model, bags


def score(model, bags, D, idx, i):
    torch.manual_seed(seed * 100 + i)
    sm = np.concatenate([X.draw(model, bags, idx, dev, n=32, target_morph=True, ctx_expr=False) for _ in range(2)], 1)
    return pearson(sm.mean(1).astype(np.float64), D["exprz"][idx].astype(np.float64))

name = f"hest_flowval_{a.task}_{a.enc}_r{a.rep}{a.tag}.json"
out = {"task": a.task, "encoder": a.enc, "rep": a.rep, "mode": a.mode, "budgets": BUD, "splits": []}
for i in ([int(x) for x in a.splits.split(",")] if a.splits else range(len(sp_))):
    t0 = time.time(); D = assemble(i, a.mode); row = {"split": i, "train": sp_[i][0], "test": sp_[i][1], "n_train": int(len(D["tr_idx"])),
                                                       "n_val": int(len(D["va_idx"])), "n_test": int(len(D["te_idx"])), "val": [], "test_by_budget": []}
    best = None
    for b in BUD:
        model, bags = fit(D, b, i)
        rt = score(model, bags, D, D["te_idx"], i); row["test_by_budget"].append(round(float(np.nanmean(rt)), 4))
        if a.mode == "val":
            row["val"].append(round(float(np.nanmean(score(model, bags, D, D["va_idx"], i))), 4))
            if best is None or row["val"][-1] > best[0]:
                best = (row["val"][-1], b, rt)
        del model, bags
    if a.mode == "val":
        row["selected_budget"] = best[1]; row["test_at_selected"] = round(float(np.nanmean(best[2])), 4)
        row["r_test_at_selected"] = [round(float(v), 4) for v in best[2]]
        row["oracle_budget"] = BUD[int(np.argmax(row["test_by_budget"]))]; row["regret"] = round(max(row["test_by_budget"]) - row["test_at_selected"], 4)
        p3 = f"hest_flow_{a.task}_{a.enc}_s{seed}.json"                      # (B) the selected budget with all training spots
        if best[1] == 4000 and os.path.exists(p3):
            row["refit_all_spots"] = json.load(open(p3))["splits"][i]["flow"]; row["refit_source"] = "phase 3 run"
        else:
            DF = assemble(i, "full"); model, bags = fit(DF, best[1], i)
            row["refit_all_spots"] = round(float(np.nanmean(score(model, bags, DF, DF["te_idx"], i))), 4); row["refit_source"] = "new run"; del model, bags
    row["seconds"] = round(time.time() - t0); out["splits"].append(row)
    print(f"[{a.task} r{a.rep} {a.mode}] split {i}: budgets {BUD} | val {row['val']} | test {row['test_by_budget']} | "
          f"selected {row.get('selected_budget')} -> test {row.get('test_at_selected')} | refit {row.get('refit_all_spots')} | {row['seconds']}s", flush=True)
    json.dump(out, open(name, "w"), indent=1)
print("DONE", name, flush=True)
