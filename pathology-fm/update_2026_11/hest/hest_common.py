"""HEST-Benchmark loaders and the official regression protocol re-implemented with numpy (no sklearn/scanpy needed)."""
import os, json, glob, csv
import numpy as np, h5py, scipy.sparse as sp
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_data")
EMB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emb")


def splits(task):
    n = len(glob.glob(f"{ROOT}/{task}/splits/train_*.csv")); out = []
    for i in range(n):
        rd = lambda f: [r["sample_id"] for r in csv.DictReader(open(f))]
        out.append((rd(f"{ROOT}/{task}/splits/train_{i}.csv"), rd(f"{ROOT}/{task}/splits/test_{i}.csv")))
    return out


def genes(task):
    return json.load(open(f"{ROOT}/{task}/var_50genes.json"))["genes"]


def load_sample(task, sid, gl):
    with h5py.File(f"{ROOT}/{task}/patches/{sid}.h5", "r") as h:
        bc = [b.decode() if isinstance(b, bytes) else str(b) for b in h["barcode"][:].reshape(-1)]; coords = h["coords"][:].astype(np.float64)
    with h5py.File(f"{ROOT}/{task}/adata/{sid}.h5ad", "r") as h:
        obs = list(h["obs/_index"].asstr()[:]); var = list(h["var/_index"].asstr()[:])
        oi = {b: i for i, b in enumerate(obs)}; vi = {g: i for i, g in enumerate(var)}
        rows = np.array([oi[b] for b in bc]); cols = np.array([vi[g] for g in gl])
        if isinstance(h["X"], h5py.Group):
            cls = sp.csr_matrix if h["X"].attrs["encoding-type"] == "csr_matrix" else sp.csc_matrix
            X = cls((h["X/data"][:], h["X/indices"][:], h["X/indptr"][:]), shape=tuple(h["X"].attrs["shape"])).tocsr()[rows][:, cols].toarray()
        else:
            order = np.argsort(rows); X = np.empty((len(rows), len(cols)), np.float64)
            X[order] = h["X"][np.sort(rows)][:, cols] if len(np.unique(rows)) == len(rows) else h["X"][:][rows[order]][:, cols]
    return {"barcodes": bc, "coords": coords, "y": np.log1p(X.astype(np.float64))}


def emb(task, sid, enc):
    return np.load(f"{EMB}/{task}__{sid}__{enc}.npy").astype(np.float64)


def pearson(p, t):
    p, t = p - p.mean(0, keepdims=True), t - t.mean(0, keepdims=True)
    d = np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0))
    return np.where(d > 0, (p * t).sum(0) / np.maximum(d, 1e-300), np.nan)


def official_ridge(Xtr, ytr, Xte, latent=256):
    """StandardScaler -> PCA(latent) -> Ridge(alpha = 100 / (latent * n_targets), fit_intercept=False), as in hest.bench."""
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1.0
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    _, _, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False); p = min(latent, Vt.shape[0])
    S, St = Z @ Vt[:p].T, Zt @ Vt[:p].T
    B = np.linalg.solve(S.T @ S + (100.0 / (p * ytr.shape[1])) * np.eye(p), S.T @ ytr)
    return St @ B
