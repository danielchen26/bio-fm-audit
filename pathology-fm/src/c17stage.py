"""Slide grading and pN staging: image features -> a real clinical endpoint.

CAMELYON17's clinical target is pN stage, and it is defined by METASTASIS SIZE,
not by a learned score:

    < 0.2 mm            isolated tumour cells (ITC)   -> contributes pN0(i+)
    0.2 mm - 2.0 mm     micro-metastasis              -> contributes pN1mi
    > 2.0 mm            macro-metastasis              -> pN1 (1-3 nodes) / pN2 (4-9)
    none                negative                      -> pN0

So the slide grader here is the clinical rule applied to predicted tumour
geometry, not a black box. Predicted-positive tiles are grouped into connected
components on the tile grid and the largest component's longest axis is measured
in millimetres. That number is directly interpretable and directly auditable,
which is the point of an "interpretable workflow linking image-derived features
to clinical endpoints".

TWO-STAGE VALIDATION, because a rule can be wrong too:

  stage A  rule( ANNOTATED polygons ) vs stages.csv
           Validates the rule and the geometry with no model in the loop. If this
           does not reproduce the published grades, the rule or the mm maths is
           broken and nothing downstream means anything.
  stage B  rule( PREDICTED tiles )    vs stages.csv
           The actual measurement. The tile classifier is trained
           CENTRE-DISJOINT, so grades are produced for hospitals never seen.

Metric: quadratic-weighted Cohen's kappa, the official CAMELYON17 metric. The
classes are ordered, so confusing negative with macro must cost far more than
confusing micro with macro, which plain accuracy would not capture.
"""
from __future__ import annotations

import os
import argparse
import glob
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

# CAMELYON17 slides, tiles and caches (about 180 GB). Set C17_ROOT to keep them on another disk.
SRC = Path(os.environ.get("C17_ROOT", Path(__file__).resolve().parent.parent / "data" / "camelyon17"))
TILES = SRC / "tiles"
GRADES = ("negative", "itc", "micro", "macro")
TILE_MM = 224 * 0.4862 / 1000.0          # 224 px at level 1 (0.4862 um/px) = 0.1089 mm
MPP0 = 0.2431


# ------------------------------------------------------------------ the rule
def grade_from_mm(longest_mm: float) -> int:
    """AJCC / CAMELYON17 size thresholds. Returns an index into GRADES."""
    if longest_mm <= 0:
        return 0                                    # negative
    if longest_mm < 0.2:
        return 1                                    # itc
    if longest_mm <= 2.0:
        return 2                                    # micro
    return 3                                        # macro


def largest_component_mm(coords: np.ndarray, step0: float) -> float:
    """Longest axis of the biggest 8-connected group of tiles, in mm.

    Tiles sit on a grid of pitch step0 (level-0 px). Two tiles are in the same
    lesion if they are grid neighbours including diagonals.
    """
    if len(coords) == 0:
        return 0.0
    g = np.round(coords / step0).astype(np.int64)
    seen: set[tuple[int, int]] = set()
    pts = {(int(x), int(y)) for x, y in g}
    best = 0.0
    for p in pts:
        if p in seen:
            continue
        stack, comp = [p], []
        seen.add(p)
        while stack:
            cx, cy = stack.pop()
            comp.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (cx + dx, cy + dy)
                    if q in pts and q not in seen:
                        seen.add(q)
                        stack.append(q)
        c = np.array(comp, dtype=np.float64)
        if len(c) == 1:
            span = 0.0
        else:
            d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1).max()
            span = float(d)
        best = max(best, span * TILE_MM + TILE_MM)    # + one tile width for extent
    return best


def component_span_grid(coords: np.ndarray, pitch0: float) -> float:
    """Longest axis of the biggest connected component, in GRID UNITS.

    Kept in grid units rather than mm on purpose. A tile covers 0.109 mm but is
    called positive when a lesion covers only 5% of it, so the positive-tile
    footprint is a DILATION of the lesion by up to one tile width -- which is
    comparable to the 0.2 mm ITC/micro threshold itself. Applying the AJCC mm
    thresholds to a dilated measurement is therefore biased by construction.
    Instead the two decision thresholds are calibrated on training centres, which
    absorbs the dilation without pretending it is not there.
    """
    if len(coords) == 0:
        return -1.0
    g = np.round(coords / pitch0).astype(np.int64)
    pts = {(int(x), int(y)) for x, y in g}
    seen: set[tuple[int, int]] = set()
    best = 0.0
    for p0 in pts:
        if p0 in seen:
            continue
        stack, comp = [p0], []
        seen.add(p0)
        while stack:
            cx, cy = stack.pop()
            comp.append((cx, cy))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    q = (cx + dx, cy + dy)
                    if q in pts and q not in seen:
                        seen.add(q)
                        stack.append(q)
        c = np.array(comp, float)
        span = 0.0 if len(c) == 1 else float(
            np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1).max())
        best = max(best, span)
    return best


def fit_thresholds(span: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """Pick the two ordered cut points on span that maximise quadratic kappa."""
    cand = np.unique(np.concatenate([[-0.5], span, span + 0.5]))
    best, bt = -2.0, (0.0, 1.0)
    for i, t1 in enumerate(cand):
        for t2 in cand[i + 1:]:
            pred = np.where(span < 0, 0, np.where(span <= t1, 1, np.where(span <= t2, 2, 3)))
            k = quadratic_kappa(truth, pred, 4)
            if k > best:
                best, bt = k, (float(t1), float(t2))
    return bt


def apply_thresholds(span: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.where(span < 0, 0, np.where(span <= t1, 1, np.where(span <= t2, 2, 3)))


def polygon_longest_mm(xml_path: Path) -> float:
    """Longest axis across annotated tumour polygons, in mm. Ground-truth geometry."""
    if not xml_path.exists():
        return 0.0
    best = 0.0
    for a in ET.parse(xml_path).getroot().findall(".//Annotation"):
        g = (a.get("PartOfGroup") or "").strip().lower()
        if g not in {"tumor", "tumour", "metastases", "_0", "_1"}:
            continue
        p = np.array([(float(c.get("X")), float(c.get("Y")))
                      for c in a.findall(".//Coordinate")])
        if len(p) < 2:
            continue
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1).max()
        best = max(best, d * MPP0 / 1000.0)
    return best


# ------------------------------------------------------------------- metrics
def quadratic_kappa(y_true, y_pred, n_cls: int) -> float:
    """Cohen's kappa with quadratic weights -- the official CAMELYON17 metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    O = np.zeros((n_cls, n_cls))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    w = np.array([[(i - j) ** 2 for j in range(n_cls)] for i in range(n_cls)], float)
    w /= max(w.max(), 1)
    ht = np.bincount(y_true, minlength=n_cls).astype(float)
    hp = np.bincount(y_pred, minlength=n_cls).astype(float)
    E = np.outer(ht, hp) / max(len(y_true), 1)
    num, den = (w * O).sum(), (w * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def confusion(y_true, y_pred, n_cls=4) -> str:
    M = np.zeros((n_cls, n_cls), int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    head = "true\\pred  " + " ".join(f"{g[:5]:>6s}" for g in GRADES[:n_cls])
    rows = [head]
    for i in range(n_cls):
        rows.append(f"{GRADES[i][:9]:<9s} " + " ".join(f"{M[i, j]:6d}" for j in range(n_cls)))
    return "\n".join("  " + r for r in rows)


# --------------------------------------------------------------- pN staging
def pn_stage(slide_grades: list[int]) -> str:
    """Official CAMELYON17 patient rule from per-node grades."""
    n_macro = sum(1 for g in slide_grades if g == 3)
    n_micro = sum(1 for g in slide_grades if g == 2)
    n_itc = sum(1 for g in slide_grades if g == 1)
    if n_macro >= 4:
        return "pN2"
    if n_macro >= 1:
        return "pN1"
    if n_micro >= 1:
        return "pN1mi"
    if n_itc >= 1:
        return "pN0(i+)"
    return "pN0"


# ------------------------------------------------------------------- driver
def load_slides(subdir="tiles", keep_grades=None):
    out = []
    for f in sorted(glob.glob(str(SRC / subdir / "*.npz"))):
        z = np.load(f, allow_pickle=False)
        if keep_grades and str(z["grade"]) not in keep_grades:
            continue
        out.append({
            "sid": str(z["slide_id"]),
            "centre": int(z["centre"]),
            "grade": str(z["grade"]),
            "coords": z["coords"],
            "labels": z["labels"],
            "patches": z["patches"],
            "step0": float(224 * 2),            # level 1 tile size in level-0 px
            "pitch0": float(z["stride0"]) if "stride0" in z.files else float(224 * 2),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--probe-device", default="cpu")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--keep", default="", help="only these grades, e.g. itc,micro")
    ap.add_argument("--tiles", default="tiles",
                    help="tile directory under the camelyon17 root")
    a = ap.parse_args()

    slides = load_slides(a.tiles, set(a.keep.split(",")) if a.keep else None)
    truth = [GRADES.index(s["grade"]) if s["grade"] in GRADES else 0 for s in slides]
    print(f"[stage] {len(slides)} slides | truth grades "
          f"{dict(Counter(GRADES[t] for t in truth))}\n")

    # ---------------- stage A: does the rule reproduce published grades at all?
    print("=== stage A: rule( annotated polygons ) vs stages.csv  [no model] ===")
    predA, sizes = [], []
    for s in slides:
        mm = polygon_longest_mm(SRC / "annotations" / f"{s['sid']}.xml")
        sizes.append(mm)
        predA.append(grade_from_mm(mm))
    kA = quadratic_kappa(truth, predA, 4)
    accA = float(np.mean(np.array(truth) == np.array(predA)))
    print(confusion(truth, predA))
    print(f"  quadratic kappa {kA:.4f}   accuracy {accA:.3f}")
    for gi, g in enumerate(GRADES):
        v = [sizes[i] for i in range(len(slides)) if truth[i] == gi]
        if v:
            print(f"    annotated longest axis, true {g:8s}: "
                  f"median {np.median(v):7.3f} mm  range {min(v):.3f}-{max(v):.3f}")
    if kA < 0.5:
        print("  WARNING rule/geometry does not reproduce published grades; "
              "stage B is meaningless until this is fixed")

    # ---------------- stage B: rule( predicted tiles ), centre-disjoint
    from audit import logistic_probe, roc_auc
    from c17bench import embed, raw_pixel_features
    print(f"\n=== stage B: rule( predicted tiles ), {a.encoder}, centre-disjoint ===")
    allp = np.concatenate([s["patches"] for s in slides])
    owner = np.concatenate([np.full(len(s["labels"]), i) for i, s in enumerate(slides)])
    ytile = np.concatenate([s["labels"] for s in slides])
    cen = np.concatenate([np.full(len(s["labels"]), s["centre"]) for s in slides])

    cache = SRC / f"emb_{a.encoder}_stage_{len(allp)}.npz"
    if cache.exists():
        emb = np.load(cache)["emb"]
        print(f"[emb] cached {emb.shape}")
    else:
        if a.encoder == "raw":
            emb = raw_pixel_features(allp)
        else:
            from transformers import AutoModel
            m = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
            emb = embed(m, allp, 224, a.device)
        np.savez_compressed(cache, emb=emb)
        print(f"[emb] computed {emb.shape} -> {cache.name}")

    # global tile index of slide si's j-th tile is offsets[si] + j
    offsets = np.cumsum([0] + [len(s["labels"]) for s in slides])

    def fit_probe(Xtr, ytr, Xte):
        """Logistic regression returning P(tumour) on the test rows."""
        import torch
        import torch.nn.functional as F
        Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
        Xte_t = torch.as_tensor(Xte, dtype=torch.float32)
        mu, sd = Xtr_t.mean(0, keepdim=True), Xtr_t.std(0, keepdim=True) + 1e-6
        Xtr_t, Xte_t = (Xtr_t - mu) / sd, (Xte_t - mu) / sd
        W = torch.zeros(Xtr_t.shape[1], 2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.1, weight_decay=1e-4)
        yt = torch.as_tensor(ytr, dtype=torch.long)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(Xtr_t @ W + b, yt).backward()
            opt.step()
        with torch.no_grad():
            return (Xte_t @ W + b).softmax(1)[:, 1].numpy()

    predB = [0] * len(slides)
    spanB = np.full(len(slides), -1.0)
    fold_of = np.full(len(slides), -1)
    for c in sorted(set(cen.tolist())):
        tr, te = np.where(cen != c)[0], np.where(cen == c)[0]
        if len(set(ytile[tr].tolist())) < 2:
            continue
        prob = fit_probe(emb[tr], ytile[tr], emb[te])
        auc = roc_auc(prob, ytile[te])
        for si in sorted(set(owner[te].tolist())):
            m = owner[te] == si
            local = te[m] - offsets[si]            # row index inside this slide
            keep = prob[m] >= a.thr
            coords = slides[si]["coords"][local[keep]]
            pitch = slides[si]["pitch0"]
            spanB[si] = component_span_grid(coords, pitch)
            predB[si] = grade_from_mm(spanB[si] * pitch * MPP0 / 1000.0 + TILE_MM
                                      if spanB[si] >= 0 else 0.0)
            fold_of[si] = c
        print(f"  centre {c}: tile-AUC {auc:.4f}  "
              f"slides graded {len(set(owner[te].tolist()))}")
    truth_a = np.array(truth)
    kB = quadratic_kappa(truth_a, np.array(predB), 4)
    print("  [B1] fixed AJCC mm thresholds on the dilated measurement")
    print(confusion(truth, predB))
    print(f"  quadratic kappa {kB:.4f}   "
          f"accuracy {float(np.mean(truth_a == np.array(predB))):.3f}")

    # B2: thresholds calibrated per fold on the OTHER centres only
    predC = np.zeros(len(slides), int)
    for c in sorted(set(fold_of.tolist())):
        if c < 0:
            continue
        te = np.where(fold_of == c)[0]
        tr = np.where((fold_of >= 0) & (fold_of != c))[0]
        if len(tr) < 6:
            continue
        t1, t2 = fit_thresholds(spanB[tr], truth_a[tr])
        predC[te] = apply_thresholds(spanB[te], t1, t2)
        print(f"  centre {c}: calibrated span cuts t1={t1:.1f} t2={t2:.1f} grid units "
              f"({t1 * 112 * MPP0 / 1000:.3f} / {t2 * 112 * MPP0 / 1000:.3f} mm)")
    kC = quadratic_kappa(truth_a, predC, 4)
    print("  [B2] thresholds calibrated on training centres (centre-disjoint)")
    print(confusion(truth, predC))
    print(f"  quadratic kappa {kC:.4f}   "
          f"accuracy {float(np.mean(truth_a == predC)):.3f}")
    predB = list(predC)

    # ---------------- patient pN, where the nodes we hold allow it
    print("\n=== patient pN stage (official rule) ===")
    by_pat: dict[str, list[int]] = {}
    for s, p in zip(slides, predB):
        by_pat.setdefault("_".join(s["sid"].split("_")[:2]), []).append(p)
    n_nodes = Counter(len(v) for v in by_pat.values())
    print(f"  {len(by_pat)} patients, nodes-per-patient held: {dict(sorted(n_nodes.items()))}")
    print("  NOTE pN requires all 5 nodes of a patient. Only the annotated subset was")
    print("       downloaded, so pN below is computed on PARTIAL node sets and is")
    print("       reported as a pipeline demonstration, not a scored result.")
    got = Counter(pn_stage(v) for v in by_pat.values())
    print(f"  predicted pN distribution (partial nodes): {dict(got)}")


if __name__ == "__main__":
    main()
