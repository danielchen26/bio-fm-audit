"""S8: the neighbourhood Transformer + flow head (clean xmodal code, unchanged) under the same 5-fold spatial-block CV,
on any slide built by st_build.py. Morphology-only inference, as in st_cv.py."""
import sys, json, time, argparse
sys.dont_write_bytecode = True
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ + "/src")
import numpy as np, torch
import xmodal_v2 as X
ap = argparse.ArgumentParser(); ap.add_argument("--slide", required=True); ap.add_argument("--enc", default="phikon"); ap.add_argument("--steps", type=int, default=4000)
a = ap.parse_args()
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
with np.load(f"st_slide_{a.slide}.npz", allow_pickle=False) as z:
    expr, coords, genes = z["expr"].astype(np.float32), z["coords"].astype(np.float32), z["gene_names"].astype(str)
morph = np.load(f"st_feat_{a.slide}_{a.enc}.npy")
_, bid = X.spatial_blocks(coords, 6, 0)
folds = np.array_split(np.random.default_rng(2026).permutation(np.unique(bid)), 5)
nbr = X.knn_graph(coords, X.K_NEIGH); scale = float(np.linalg.norm(coords.max(0) - coords.min(0)) / 10.0); G = expr.shape[1]


def ens_crps(samples, y):
    t1 = np.abs(samples - y[:, None, :]).mean(1)
    s = np.sort(samples, axis=1); n = s.shape[1]
    w = (2 * np.arange(1, n + 1) - n - 1).reshape(1, n, 1)
    return (t1 - 0.5 * 2.0 * (w * s).sum(1) / (n * (n - 1)))

for K in range(5):
    is_test = np.isin(bid, folds[K]); tr = ~is_test
    exprz, mu, sd = X.normalize_expression(expr, tr); tr_idx, te_idx = np.where(tr)[0], np.where(is_test)[0]; truth = exprz[te_idx]
    clim = exprz[tr_idx][np.random.default_rng(5000 + K).integers(0, len(tr_idx), (len(te_idx), 512))]
    out = {"te_idx": te_idx, "blocks": bid[te_idx], "truth_log": expr[te_idx], "crps_clim": ens_crps(clim, truth).mean(1)}; del clim
    torch.manual_seed(K)
    bags = X.Bags(morph, exprz, coords, nbr, scale, observed=tr)
    model = X.CrossModal(G, morph.shape[1], d=256, layers=4, head="flow").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=a.steps, pct_start=0.1)
    rg, t0 = np.random.default_rng(K), time.time()
    for _ in range(a.steps):
        loss = X.training_loss(model, bags, rg.choice(tr_idx, 64), dev)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    torch.manual_seed(K)
    sm = np.concatenate([X.draw(model, bags, te_idx, dev, n=32, target_morph=True, ctx_expr=False) for _ in range(4)], 1)
    out["pred_log"] = (sm.mean(1) * sd + mu).astype(np.float32); out["sd_log"] = (sm.std(1) * sd).astype(np.float32)
    out["crps"] = ens_crps(sm[:, :32], truth).mean(1)
    np.savez_compressed(f"st_cv2_{a.slide}_{a.enc}_fold{K}.npz", **out)
    print(f"[{a.slide} {a.enc}] fold {K}: CRPS {out['crps'].mean():.4f} (clim {out['crps_clim'].mean():.4f})  median gene r "
          f"{np.nanmedian(X.pearson_per_gene(sm.mean(1), truth)):.3f}  train {time.time() - t0:.0f}s", flush=True)
print("done", flush=True)
