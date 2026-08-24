"""Step 1: real structural data with homology-aware splits.

Pipeline: RCSB 30%-identity cluster representatives -> backbone coords
-> DSSP-style SS8 -> Shrake-Rupley relative SASA -> cluster-level train/val/test.

Every track below is DERIVED FROM MEASURED COORDINATES. Nothing is synthetic.
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import pickle
import random
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CLUSTER_URL = "https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-30.txt"

AA3 = "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
AA1 = "ARNDCQEGHILKMFPSTWYV"
THREE2IDX = {t: i for i, t in enumerate(AA3)}
N_AA = 20  # 20 = unknown/other

SS8 = "HBEGITS-"  # DSSP alphabet; '-' = coil
SS8_IDX = {c: i for i, c in enumerate(SS8)}

# Tien et al. 2013 theoretical max per-residue solvent accessibility (A^2)
MAXACC = np.array([129, 274, 195, 193, 167, 225, 223, 104, 224, 197,
                   201, 236, 224, 240, 159, 155, 172, 285, 263, 174, 200.0])

VDW = {"N": 1.55, "C": 1.70, "O": 1.52, "S": 1.80}


# ---------------------------------------------------------------- fetch / parse
def cluster_representatives(limit: int) -> list[str]:
    """One entity per 30%-identity cluster => no homologs leak across splits."""
    cache = DATA / "clusters-30.txt"
    if not cache.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CLUSTER_URL, cache)
    reps = []
    with cache.open() as fh:
        for line in fh:
            members = line.split()
            if members:
                reps.append(members[0])  # cluster representative
    random.Random(0).shuffle(reps)
    return reps[:limit]


def fetch_pdb(entity: str) -> tuple[str, bytes] | None:
    pdb_id = entity.split("_")[0].lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return entity, gzip.decompress(r.read())
    except Exception:
        return None


def parse_backbone(raw: bytes, min_len: int, max_len: int) -> dict | None:
    """Longest contiguous complete-backbone fragment across chains of model 1."""
    chain_res: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    order: dict[str, list[int]] = {}
    for line in raw.decode("utf-8", "ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[16] not in (" ", "A"):        # altloc: keep primary only
            continue
        name, resn, ch = line[12:16].strip(), line[17:20].strip(), line[21]
        if name not in ("N", "CA", "C", "O", "CB") or resn not in THREE2IDX:
            continue
        try:
            key = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], np.float32)
        except ValueError:
            continue
        if line[26] != " ":                   # skip insertion codes
            continue
        res = chain_res.setdefault(ch, {})
        if key not in res:
            res[key] = {"aa": resn}
            order.setdefault(ch, []).append(key)
        res[key][name] = xyz

    best = None
    for ch, keys in order.items():
        res = chain_res[ch]
        keep = [k for k in keys if all(a in res[k] for a in ("N", "CA", "C", "O"))]
        if len(keep) < min_len:
            continue
        ca = np.stack([res[k]["CA"] for k in keep])
        # split on chain breaks: consecutive CA must be ~3.8 A apart and numbering contiguous
        gap = (np.linalg.norm(np.diff(ca, axis=0), axis=1) > 4.5) | (np.diff(keep) != 1)
        bounds = [0, *(np.where(gap)[0] + 1), len(keep)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= min_len and (best is None or b - a > best[1] - best[0]):
                best = (a, b, ch, keep)
    if best is None:
        return None
    a, b, ch, keys = best
    res = chain_res[ch]
    keep = keys[a:min(b, a + max_len)]           # cap length, keep contiguity
    seq = np.array([THREE2IDX[res[k]["aa"]] for k in keep], np.int8)
    coords = np.stack([[res[k][x] for x in ("N", "CA", "C", "O")] for k in keep]).astype(np.float32)
    cb = np.stack([res[k].get("CB", res[k]["CA"]) for k in keep]).astype(np.float32)
    return {"seq": seq, "coords": coords, "cb": cb}


# ------------------------------------------------------- DSSP-style SS8 (real)
def _hbond_energy(coords: np.ndarray) -> np.ndarray:
    """Kabsch-Sander electrostatic H-bond energy, kcal/mol. coords: [L,4,3] N,CA,C,O."""
    L = len(coords)
    N, CA, C, O = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    # amide H placed opposite the preceding C=O direction
    H = N.copy()
    d = N[1:] - C[:-1]
    co = O[:-1] - C[:-1]
    co /= np.linalg.norm(co, axis=-1, keepdims=True) + 1e-8
    H[1:] = N[1:] - co * 1.0
    del d

    def dist(a, b):
        return np.linalg.norm(a[:, None] - b[None], axis=-1) + 1e-8

    e = 0.084 * 332 * (1.0 / dist(O, N) + 1.0 / dist(C, H) - 1.0 / dist(O, H) - 1.0 / dist(C, N))
    e[:, 0] = 1e3                      # residue 0 has no real amide H
    idx = np.arange(L)
    for off in (-2, -1, 0, 1, 2):      # exclude near-diagonal self bonds
        i = idx[np.clip(idx + off, 0, L - 1) == idx + off]
        e[i, i + off] = 1e3
    return e


def assign_ss8(coords: np.ndarray) -> np.ndarray:
    """DSSP assignment with standard priority H > B > E > G > I > T > S > coil."""
    L = len(coords)
    e = _hbond_energy(coords)
    hb = e < -0.5                                   # hb[i,j]: C=O(i) -> H-N(j)

    def bond(i, j):                                 # N-H(i) ... O=C(j)
        return 0 <= i < L and 0 <= j < L and hb[j, i]

    ss = np.full(L, SS8_IDX["-"], np.int8)
    # n-turns
    turn = {n: np.array([bond(i + n, i) for i in range(L)]) for n in (3, 4, 5)}
    # beta bridges: parallel / antiparallel
    bridge = np.zeros((L, L), bool)
    for i in range(1, L - 1):
        for j in range(i + 3, L - 1):
            par = (bond(j, i - 1) and bond(i + 1, j)) or (bond(i, j - 1) and bond(j + 1, i))
            anti = (bond(j, i) and bond(i, j)) or (bond(j + 1, i - 1) and bond(i + 1, j - 1))
            if par or anti:
                bridge[i, j] = bridge[j, i] = True

    def put(mask, sym):
        m = mask & (ss == SS8_IDX["-"])
        ss[m] = SS8_IDX[sym]

    # priority order
    helix4 = np.array([turn[4][i] and turn[4][max(i - 1, 0)] for i in range(L)])
    h = np.zeros(L, bool)
    for i in range(1, L):
        if helix4[i]:
            h[i:i + 4] = True
    put(h, "H")
    lad = bridge.any(1)
    ladder_len = np.zeros(L, int)
    for i in range(L):
        if lad[i]:
            ladder_len[i] = sum(lad[max(i - 1, 0):i + 2])
    put(lad & (ladder_len >= 2), "E")
    put(lad, "B")
    g = np.zeros(L, bool)
    for i in range(1, L):
        if turn[3][i] and turn[3][max(i - 1, 0)]:
            g[i:i + 3] = True
    put(g, "G")
    pi = np.zeros(L, bool)
    for i in range(1, L):
        if turn[5][i] and turn[5][max(i - 1, 0)]:
            pi[i:i + 5] = True
    put(pi, "I")
    put(turn[3] | turn[4] | turn[5], "T")
    # bend: CA(i-2),CA(i),CA(i+2) angle > 70 deg
    ca = coords[:, 1]
    bend = np.zeros(L, bool)
    if L > 4:
        v1, v2 = ca[2:-2] - ca[:-4], ca[4:] - ca[2:-2]
        cos = np.sum(v1 * v2, 1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8)
        bend[2:-2] = np.degrees(np.arccos(np.clip(cos, -1, 1))) > 70
    put(bend, "S")
    return ss


# ------------------------------------------------- Shrake-Rupley relative SASA
def _sphere(n: int = 92) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)


_SPHERE = _sphere()


def relative_sasa(coords: np.ndarray, cb: np.ndarray, seq: np.ndarray, probe: float = 1.4) -> np.ndarray:
    """Per-residue SASA / theoretical max, binned into 4 levels."""
    atoms, radii, owner = [], [], []
    for i in range(len(coords)):
        for j, el in enumerate("NCCO"):
            atoms.append(coords[i, j]); radii.append(VDW[el]); owner.append(i)
        atoms.append(cb[i]); radii.append(VDW["C"]); owner.append(i)
    atoms = np.asarray(atoms, np.float64)
    radii = np.asarray(radii) + probe
    owner = np.asarray(owner)

    d = np.linalg.norm(atoms[:, None] - atoms[None], axis=-1)
    per_res = np.zeros(len(coords))
    for a in range(len(atoms)):
        near = np.where((d[a] < radii[a] + radii.max()) & (np.arange(len(atoms)) != a))[0]
        pts = atoms[a] + _SPHERE * radii[a]
        if len(near):
            occluded = (np.linalg.norm(pts[:, None] - atoms[near][None], axis=-1) < radii[near][None]).any(1)
        else:
            occluded = np.zeros(len(pts), bool)
        frac = 1.0 - occluded.mean()
        per_res[owner[a]] += 4 * np.pi * radii[a] ** 2 * frac
    rel = per_res / MAXACC[np.clip(seq, 0, 20)]
    return np.digitize(rel, [0.05, 0.25, 0.50]).astype(np.int8)  # 4 bins


# ------------------------------------------------------------------ entrypoint
def build(n_clusters: int = 1600, min_len: int = 40, max_len: int = 192, workers: int = 24):
    reps = cluster_representatives(n_clusters)
    print(f"[data] {len(reps)} cluster representatives (30% identity)")
    records = []
    with cf.ThreadPoolExecutor(workers) as ex:
        for k, got in enumerate(ex.map(fetch_pdb, reps)):
            if got is None:
                continue
            entity, raw = got
            parsed = parse_backbone(raw, min_len, max_len)
            if parsed is None:
                continue
            try:
                parsed["ss8"] = assign_ss8(parsed["coords"])
                parsed["sasa"] = relative_sasa(parsed["coords"], parsed.pop("cb"), parsed["seq"])
            except Exception:
                continue
            parsed["id"] = entity
            records.append(parsed)
            if len(records) % 50 == 0:
                print(f"[data] {len(records)} usable / {k + 1} fetched", flush=True)

    # cluster-level split: representatives are already 1-per-cluster, so a
    # random split here IS a homology-aware split at 30% identity.
    rng = random.Random(1234)
    rng.shuffle(records)
    n = len(records)
    n_test, n_val = max(1, n // 10), max(1, n // 10)
    splits = {"test": records[:n_test],
              "val": records[n_test:n_test + n_val],
              "train": records[n_test + n_val:]}
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "dataset.pkl").open("wb") as fh:
        pickle.dump(splits, fh)
    for k, v in splits.items():
        lens = [len(r["seq"]) for r in v]
        print(f"[data] {k:5s} n={len(v):4d} residues={sum(lens):6d} len={min(lens)}-{max(lens)}")
    return splits


if __name__ == "__main__":
    build()
