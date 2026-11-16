"""CAMELYON17 tile dataset + the split regimes that matter.

BreaKHis could only test patient-level leakage: one cohort, one lab, one scanner.
CAMELYON17 adds the variable that actually breaks deployed pathology models --
the CENTRE. Five hospitals, five scanners, five staining protocols, and
stages.csv states the centre explicitly so it never has to be inferred.

Four regimes, in increasing honesty:

  random  tiles shuffled. Maximally leaky: tiles from one lesion land on both
          sides, and 400 tiles from one slide are near-duplicates.
  slide   slide-disjoint. Removes near-duplicate leakage but a patient
          contributes several nodes, so a patient can still straddle.
  patient patient-disjoint. What the BreaKHis study measured.
  centre  LEAVE-ONE-CENTRE-OUT. Train on 4 hospitals, test on the 5th. This is
          the number that predicts whether a model survives being deployed
          somewhere it was not trained, and it is the one almost never reported.

The gap between `patient` and `centre` is the quantity of interest: it isolates
site/scanner/stain shift from ordinary biological variation.
"""
from __future__ import annotations

import os
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# CAMELYON17 slides, tiles and caches (about 180 GB). Set C17_ROOT to keep them on another disk.
TILES = Path(os.environ.get("C17_ROOT", Path(__file__).resolve().parent.parent / "data" / "camelyon17")) / "tiles"
GRADES = ("negative", "itc", "micro", "macro")


@dataclass
class C17:
    patches: np.ndarray        # [N,224,224,3] uint8
    y: np.ndarray              # [N] tumour 0/1
    centre: np.ndarray         # [N] 0-4
    slide: np.ndarray          # [N] slide index
    patient: np.ndarray        # [N] patient index
    grade: np.ndarray          # [N] index into GRADES
    cover: np.ndarray          # [N] tumour coverage of the tile
    slide_ids: list[str]
    patient_ids: list[str]

    def __len__(self):
        return len(self.y)

    def summary(self) -> str:
        lines = [f"tiles {len(self)}  tumour {int(self.y.sum())} ({self.y.mean():.1%})  "
                 f"slides {len(self.slide_ids)}  patients {len(self.patient_ids)}"]
        for c in range(5):
            m = self.centre == c
            if not m.any():
                continue
            lines.append(f"  centre {c}: {int(m.sum()):6d} tiles  "
                         f"{int(self.y[m].sum()):5d} tumour ({self.y[m].mean():.1%})  "
                         f"{len(set(self.slide[m].tolist())):2d} slides  "
                         f"{len(set(self.patient[m].tolist())):2d} patients")
        for gi, g in enumerate(GRADES):
            m = self.grade == gi
            if m.any():
                lines.append(f"  grade {g:9s}: {int(m.sum()):6d} tiles, "
                             f"{int(self.y[m].sum()):5d} tumour")
        return "\n".join(lines)


def load(limit_slides: int = 0, max_tiles_per_slide: int = 0) -> C17:
    files = sorted(glob.glob(str(TILES / "*.npz")))
    if limit_slides:
        files = files[:limit_slides]
    if not files:
        raise FileNotFoundError(f"no tile shards in {TILES}")

    P, Y, C, S, PT, G, CV = [], [], [], [], [], [], []
    slide_ids, patient_ids = [], []
    for si, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        sid = str(z["slide_id"])
        pid = "_".join(sid.split("_")[:2])            # patient_XXX
        if pid not in patient_ids:
            patient_ids.append(pid)
        slide_ids.append(sid)
        n = len(z["labels"])
        keep = np.arange(n)
        if max_tiles_per_slide and n > max_tiles_per_slide:
            # keep every positive, subsample negatives -- positives are the scarce class
            lab = z["labels"]
            pos = np.where(lab == 1)[0]
            neg = np.where(lab == 0)[0]
            room = max(0, max_tiles_per_slide - len(pos))
            rng = np.random.default_rng(si)
            neg = neg[rng.choice(len(neg), min(room, len(neg)), replace=False)] if room else neg[:0]
            keep = np.concatenate([pos, neg])
        P.append(z["patches"][keep])
        Y.append(z["labels"][keep])
        CV.append(z["cover"][keep])
        C.append(np.full(len(keep), int(z["centre"]), np.int64))
        S.append(np.full(len(keep), si, np.int64))
        PT.append(np.full(len(keep), patient_ids.index(pid), np.int64))
        g = str(z["grade"])
        G.append(np.full(len(keep), GRADES.index(g) if g in GRADES else 0, np.int64))

    return C17(np.concatenate(P), np.concatenate(Y), np.concatenate(C),
               np.concatenate(S), np.concatenate(PT), np.concatenate(G),
               np.concatenate(CV), slide_ids, patient_ids)


# ------------------------------------------------------------------- splits
@dataclass
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray
    note: str = ""

    def sizes(self):
        return len(self.train), len(self.test)


def make_split(d: C17, regime: str, seed: int = 0, held_centre: int | None = None) -> Split:
    idx = np.arange(len(d))
    rng = np.random.default_rng(seed)

    if regime == "random":
        p = rng.permutation(len(d))
        cut = int(0.75 * len(d))
        return Split(regime, p[:cut], p[cut:], "tiles shuffled -- maximally leaky")

    if regime in ("slide", "patient"):
        key = d.slide if regime == "slide" else d.patient
        groups = np.unique(key)
        pick = rng.permutation(groups)
        n_te = max(1, len(groups) // 4)
        te = set(pick[:n_te].tolist())
        m = np.isin(key, list(te))
        return Split(regime, idx[~m], idx[m], f"{len(te)}/{len(groups)} groups held out")

    if regime == "centre":
        cs = sorted(set(d.centre.tolist()))
        hc = held_centre if held_centre is not None else cs[seed % len(cs)]
        m = d.centre == hc
        return Split(regime, idx[~m], idx[m], f"leave-out centre {hc}")

    raise ValueError(regime)


def audit_split(d: C17, sp: Split) -> dict:
    """Verify disjointness instead of trusting it."""
    f = lambda a, k: set(k[a].tolist())
    return {
        "n_train": len(sp.train), "n_test": len(sp.test),
        "shared_slides": len(f(sp.train, d.slide) & f(sp.test, d.slide)),
        "shared_patients": len(f(sp.train, d.patient) & f(sp.test, d.patient)),
        "shared_centres": len(f(sp.train, d.centre) & f(sp.test, d.centre)),
        "train_pos": float(d.y[sp.train].mean()) if len(sp.train) else float("nan"),
        "test_pos": float(d.y[sp.test].mean()) if len(sp.test) else float("nan"),
    }


if __name__ == "__main__":
    d = load()
    print(d.summary())
    print()
    for regime in ("random", "slide", "patient", "centre"):
        sp = make_split(d, regime)
        a = audit_split(d, sp)
        print(f"[{regime:8s}] train/test {sp.sizes()}  ({sp.note})")
        print(f"           shared slides {a['shared_slides']:3d} | patients "
              f"{a['shared_patients']:3d} | centres {a['shared_centres']:2d} | "
              f"pos train {a['train_pos']:.3f} test {a['test_pos']:.3f}")
