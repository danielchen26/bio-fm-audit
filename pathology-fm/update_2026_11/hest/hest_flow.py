"""Cross-patient test of the neighbourhood Transformer + flow head (clean xmodal code, unchanged recipe) on HEST-Benchmark:
official patient-stratified splits and gene lists; inference from morphology only, no expression from test patients."""
import sys, json, time, argparse, os
sys.dont_write_bytecode = True
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))); sys.path.insert(0, PROJ + "/src"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import xmodal_v2 as X
from hest_common import *
ap = argparse.ArgumentParser(); ap.add_argument("--task", required=True); ap.add_argument("--enc", default="phikon")
ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
gl = genes(a.task); sp_ = splits(a.task); S = {}
for s in sorted({s for tr, te in sp_ for s in tr + te}):
    d = load_sample(a.task, s, gl); d["feat"] = emb(a.task, s, a.enc).astype(np.float32)
    c = d["coords"]; nn = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1) + np.eye(len(c)) * 1e18).min(1)
    d["c"] = (c / np.median(nn)).astype(np.float32); d["nbr"] = X.knn_graph(d["c"], X.K_NEIGH); S[s] = d


def ens_crps(samples, y):
    t1 = np.abs(samples - y[:, None, :]).mean(1)
    s = np.sort(samples, axis=1); n = s.shape[1]
    w = (2 * np.arange(1, n + 1) - n - 1).reshape(1, n, 1)
    return (t1 - 0.5 * 2.0 * (w * s).sum(1) / (n * (n - 1)))

out = {"task": a.task, "encoder": a.enc, "genes": gl, "splits": []}; dump = {}
for i, (tr, te) in enumerate(sp_):
    order = tr + te; off = np.cumsum([0] + [len(S[s]["feat"]) for s in order])
    morph = np.concatenate([S[s]["feat"] for s in order]); y = np.concatenate([S[s]["y"] for s in order]).astype(np.float32)
    coords = np.concatenate([S[s]["c"] for s in order]); nbr = np.concatenate([S[s]["nbr"] + off[j] for j, s in enumerate(order)])
    observed = np.zeros(len(y), bool); observed[:off[len(tr)]] = True
    exprz, mu, sd = X.normalize_expression(y, observed); tr_idx, te_idx = np.where(observed)[0], np.where(~observed)[0]
    torch.manual_seed(a.seed * 100 + i)
    bags = X.Bags(morph, exprz, coords, nbr, 12.0, observed=observed)
    model = X.CrossModal(len(gl), morph.shape[1], d=256, layers=4, head="flow").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.1)
    rg, t0 = np.random.default_rng(a.seed * 100 + i), time.time()
    for _ in range(a.steps):
        loss = X.training_loss(model, bags, rg.choice(tr_idx, 64), dev)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    torch.manual_seed(a.seed * 100 + i)
    sm = np.concatenate([X.draw(model, bags, te_idx, dev, n=32, target_morph=True, ctx_expr=False) for _ in range(2)], 1)
    truth = exprz[te_idx]; r_flow = pearson(sm.mean(1).astype(np.float64), truth.astype(np.float64))
    clim = exprz[tr_idx][np.random.default_rng(5000 + i).integers(0, len(tr_idx), (len(te_idx), 256))]
    crps, crps_clim = float(ens_crps(sm[:, :32], truth).mean()), float(ens_crps(clim, truth).mean()); del clim
    F = morph.astype(np.float64); Fc = np.concatenate([F, F[nbr].mean(1)], 1)
    r_ridge = pearson(official_ridge(F[tr_idx], y[tr_idx].astype(np.float64), F[te_idx]), y[te_idx].astype(np.float64))
    r_ctx = pearson(official_ridge(Fc[tr_idx], y[tr_idx].astype(np.float64), Fc[te_idx]), y[te_idx].astype(np.float64))
    row = {"split": i, "train": tr, "test": te, "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
           "flow": round(float(np.nanmean(r_flow)), 4), "ridge": round(float(np.nanmean(r_ridge)), 4), "ridge_context": round(float(np.nanmean(r_ctx)), 4),
           "genes_flow_above_ridge": int(np.nansum(r_flow > r_ridge)), "crps_rel_vs_climatology": round(crps / crps_clim - 1, 4),
           "r_flow": [round(float(v), 4) for v in r_flow], "r_ridge": [round(float(v), 4) for v in r_ridge], "r_ridge_context": [round(float(v), 4) for v in r_ctx],
           "train_seconds": round(time.time() - t0)}
    out["splits"].append(row)
    dump[f"pred_{i}"] = (sm.mean(1) * sd + mu).astype(np.float32); dump[f"sd_{i}"] = (sm.std(1) * sd).astype(np.float32)
    dump[f"truth_{i}"] = y[te_idx]; dump[f"coords_{i}"] = np.concatenate([S[s]["coords"] for s in te]).astype(np.float32)
    print(f"[{a.task} {a.enc}] split {i} test {te}: flow {row['flow']:.4f} | ridge {row['ridge']:.4f} | ridge+context {row['ridge_context']:.4f} | "
          f"genes flow>ridge {row['genes_flow_above_ridge']}/{len(gl)} | CRPS vs no-skill {row['crps_rel_vs_climatology']:+.3f} | {row['train_seconds']}s", flush=True)
    json.dump(out, open(f"hest_flow_{a.task}_{a.enc}_s{a.seed}.json", "w"), indent=1)
for k in ("flow", "ridge", "ridge_context"):
    out[f"mean_{k}"] = round(float(np.mean([r[k] for r in out["splits"]])), 4)
json.dump(out, open(f"hest_flow_{a.task}_{a.enc}_s{a.seed}.json", "w"), indent=1)
np.savez_compressed(f"hest_flow_{a.task}_{a.enc}_s{a.seed}.npz", **dump)
print(f"[{a.task} {a.enc}] mean over splits: flow {out['mean_flow']:.4f} ridge {out['mean_ridge']:.4f} ridge+context {out['mean_ridge_context']:.4f}", flush=True)
