"""Data layer for the computational-pathology FM.

Three things live here, and each maps to a line in the Merck JD:

  split regimes        -> "QC standards and best practices"
  Macenko normalisation-> stain/scanner batch-effect control
  bag construction     -> "weakly supervised" patient-level learning

The split regimes are the load-bearing part. A random split over IMAGES puts
near-duplicate crops of the same slide on both sides of the boundary, which
inflates every downstream number. Patient-disjoint is the honest split; the
magnification split is an explicit domain shift. We measure all three so the
size of the leak is a reported quantity, not an unknown.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# Ruifrok & Johnston H&E reference stain matrix (3x2: RGB x [haematoxylin, eosin])
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])


# ------------------------------------------------------------------ manifest
def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(DATA / "breakhis_manifest.csv")
    need = {"path", "patient_id", "label", "subtype", "magnification", "seq"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    df["y"] = (df["label"] == "malignant").astype(np.int64)
    return df


def _hash_frac(key: str, salt: str) -> float:
    """Deterministic uniform in [0,1) from a string key. Stable across runs."""
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


@dataclass
class Split:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def sizes(self):
        return len(self.train), len(self.val), len(self.test)


def make_split(df: pd.DataFrame, regime: str, seed: int = 0) -> Split:
    """regime in {'random', 'patient', 'magnification'}.

    'random'        -- leaky on purpose. Splits images, ignoring patient identity.
    'patient'       -- honest. No patient appears in more than one partition.
    'magnification' -- domain shift. Train on 40X/100X, test on 400X.
    """
    idx = np.arange(len(df))
    if regime == "random":
        f = np.array([_hash_frac(p, f"img{seed}") for p in df["path"]])
        return Split(regime, idx[f >= 0.30], idx[(f >= 0.15) & (f < 0.30)], idx[f < 0.15])

    if regime == "patient":
        # stratify the PATIENT assignment by label so both classes appear everywhere
        train, val, test = [], [], []
        for label, grp in df.groupby("label"):
            pats = sorted(grp["patient_id"].unique())
            f = {p: _hash_frac(p, f"pat{seed}") for p in pats}
            for p in pats:
                bucket = test if f[p] < 0.20 else (val if f[p] < 0.35 else train)
                bucket.append(p)
        sel = lambda pats: idx[df["patient_id"].isin(set(pats)).values]
        return Split(regime, sel(train), sel(val), sel(test))

    if regime == "magnification":
        mag = df["magnification"].values
        # keep it patient-disjoint too, otherwise the shift is confounded by leakage
        pats = sorted(df["patient_id"].unique())
        f = {p: _hash_frac(p, f"pat{seed}") for p in pats}
        heldout = {p for p in pats if f[p] < 0.25}
        held = df["patient_id"].isin(heldout).values
        return Split(regime,
                     idx[np.isin(mag, [40, 100]) & ~held],
                     idx[np.isin(mag, [200]) & ~held],
                     idx[np.isin(mag, [400]) & held])

    raise ValueError(regime)


def assert_disjoint(df: pd.DataFrame, sp: Split) -> dict:
    """Quantify the leak: how many patients are shared across partitions."""
    pat = df["patient_id"].values
    tr, va, te = set(pat[sp.train]), set(pat[sp.val]), set(pat[sp.test])
    return {"n_train_patients": len(tr), "n_test_patients": len(te),
            "shared_train_test": len(tr & te), "shared_train_val": len(tr & va)}


# ------------------------------------------------- Macenko stain normalisation
def macenko_stain_matrix(rgb: np.ndarray, beta: float = 0.15, alpha: float = 1.0):
    """Estimate the 3x2 stain matrix and per-stain max concentration for one image."""
    od = -np.log10((rgb.reshape(-1, 3).astype(np.float64) + 1.0) / 256.0)
    od_hat = od[np.all(od > beta, axis=1)]
    if len(od_hat) < 32:
        return HE_REF, MAXC_REF
    # principal plane of the OD cloud
    _, v = np.linalg.eigh(np.cov(od_hat.T))
    plane = od_hat @ v[:, 1:3]                       # two largest eigenvectors
    phi = np.arctan2(plane[:, 1], plane[:, 0])
    lo, hi = np.percentile(phi, [alpha, 100 - alpha])
    v1 = v[:, 1:3] @ np.array([np.cos(lo), np.sin(lo)])
    v2 = v[:, 1:3] @ np.array([np.cos(hi), np.sin(hi)])
    # haematoxylin is the more blue/violet stain -> larger first OD component
    he = np.stack([v1, v2] if v1[0] > v2[0] else [v2, v1], axis=1)
    he = np.abs(he) / (np.linalg.norm(np.abs(he), axis=0, keepdims=True) + 1e-8)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]  # [2, npix]
    maxc = np.percentile(conc, 99, axis=1)
    return he, np.maximum(maxc, 1e-6)


def macenko_normalize(rgb: np.ndarray) -> np.ndarray:
    """Map an image onto the reference H&E appearance. Batch-effect control."""
    he, maxc = macenko_stain_matrix(rgb)
    h, w, _ = rgb.shape
    od = -np.log10((rgb.reshape(-1, 3).astype(np.float64) + 1.0) / 256.0)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    conc = conc * (MAXC_REF / maxc)[:, None]
    out = 256.0 * 10 ** (-(HE_REF @ conc)) - 1.0
    return np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)


# --------------------------------------------------------------- patch loading
def load_patch(path: str, size: int, normalize: bool, rng=None) -> np.ndarray:
    """Centre crop (or random crop when rng is given) resized to size x size."""
    im = Image.open(ROOT / path).convert("RGB")
    W, H = im.size
    side = min(W, H)
    if rng is None:
        left, top = (W - side) // 2, (H - side) // 2
    else:
        left, top = rng.integers(0, W - side + 1), rng.integers(0, H - side + 1)
    a = np.asarray(im.crop((left, top, left + side, top + side)), dtype=np.uint8)
    # stain-normalise at native resolution: estimating the stain matrix from a
    # 96x96 thumbnail is far less stable than from the full crop
    if normalize:
        a = macenko_normalize(a)
    return np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def build_cache(df: pd.DataFrame, size: int = 96, normalize: bool = False,
                out: str | None = None, workers: int = 12) -> np.ndarray:
    """Decode every image once into a uint8 array. 7909 * 96 * 96 * 3 ~= 219 MB."""
    import concurrent.futures as cf
    paths = df["path"].tolist()
    arr = np.zeros((len(paths), size, size, 3), np.uint8)
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, a in enumerate(ex.map(lambda p: load_patch(p, size, normalize), paths)):
            arr[i] = a
            if (i + 1) % 1000 == 0:
                print(f"[cache] {i + 1}/{len(paths)}", flush=True)
    if out:
        np.savez_compressed(DATA / out, patches=arr)
        print(f"[cache] wrote {DATA / out}  {arr.nbytes / 1e6:.0f} MB raw")
    return arr


if __name__ == "__main__":
    df = load_manifest()
    print(f"[data] {len(df)} images, {df['patient_id'].nunique()} patients")
    for regime in ("random", "patient", "magnification"):
        sp = make_split(df, regime)
        print(f"\n[{regime}] train/val/test = {sp.sizes()}")
        for k, v in assert_disjoint(df, sp).items():
            print(f"    {k:22s} {v}")
        for part, name in ((sp.train, "train"), (sp.test, "test")):
            frac = df["y"].values[part].mean() if len(part) else float("nan")
            print(f"    {name} malignant fraction {frac:.3f}")
