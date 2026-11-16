"""WSI tiling: 19-gigapixel slides -> a tile dataset with centre and tumour labels.

The capability BreaKHis could not exercise. BreaKHis ships pre-cropped 700x460
microscope fields; CAMELYON17 ships 96651 x 197226 pyramidal TIFFs where finding
the tissue at all is step one.

Design decisions, and the measurements that forced them:

LEVEL 1, NOT LEVEL 0. Slides are 0.243 um/px (40x). Level 1 is 0.486 um/px (20x),
so a 224 px tile spans 109 um -- inside the 128-256 um band pathology FMs are
pretrained on. Reading level 0 would hand a pretrained FM an out-of-distribution
scale, which is precisely the mistake that produced a spurious null result
earlier in this project.

LABELS COME FROM THE XML, NOT FROM masks/. Measured: the distributed *_mask.tif
is a TISSUE mask, not a tumour mask -- every tissue tile scored >= 0.5 coverage,
mean 0.995, and mask-nonzero area (11.5% of slide) tracks Otsu tissue area
(8.2%). Using it as a tumour label makes 100% of tiles positive. The lesions live
in annotations/*.xml as ASAP polygons.

LESION-GUIDED SAMPLING, NOT A UNIFORM GRID. Annotated lesion area spans five
orders of magnitude: 116.25 mm2 (macro) down to 0.0010 mm2 (ITC). A tile is
0.0119 mm2, so the smallest ITC covers 8% of one tile. Any fixed coverage
threshold near 0.5 gives ITC slides ZERO positive tiles -- 16 of 50 slides
silently contributing nothing. So positives are enumerated FROM the polygons
(bbox-overlapping grid tiles plus one tile centred on each lesion centroid),
guaranteeing every lesion yields at least one positive regardless of grid phase.

AMBIGUOUS TILES ARE DISCARDED, NOT GUESSED. Coverage strictly between the
negative and positive thresholds is thrown away. Boundary tiles are genuinely
ambiguous; forcing them into a class injects label noise that later looks like
model error.

EXCLUSION POLYGONS ARE HONOURED. One slide (patient_010_node_4) carries an
`Exclusion` group; tiles touching it are dropped.

CENTRE COMES FROM stages.csv, not inferred from the patient index. It is the
confound variable for every split and probe downstream.
"""
from __future__ import annotations

import os
import argparse
import concurrent.futures as cf
import csv
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import openslide
from PIL import Image, ImageDraw

# CAMELYON17 slides, tiles and caches (about 180 GB). Set C17_ROOT to keep them on another disk.
SRC = Path(os.environ.get("C17_ROOT", Path(__file__).resolve().parent.parent / "data" / "camelyon17"))
OUT = SRC / "tiles"
TILE = 224
LEVEL = 1                      # 0.486 um/px == 20x == 109 um per 224 px tile
TUMOUR_GROUPS = {"tumor", "tumour", "metastases", "_0", "_1"}
EXCLUDE_GROUPS = {"exclusion", "_2"}


# ------------------------------------------------------------------- metadata
def load_stages(path: Path | None = None) -> dict[str, tuple[str, int]]:
    """slide_id -> (grade, centre). Also yields patient-level pN via *.zip rows."""
    p = path or (SRC / "stages.csv")
    out = {}
    with p.open() as fh:
        for row in csv.DictReader(fh):
            key = row["patient"].strip()
            if key.endswith(".tif"):
                out[key[:-4]] = (row["stage"].strip(), int(row["center"]))
    return out


def parse_annotations(path: Path):
    """ASAP XML -> (tumour polygons, exclusion polygons), level-0 coordinates."""
    tum, exc, unknown = [], [], set()
    root = ET.parse(path).getroot()
    for a in root.findall(".//Annotation"):          # only Annotation, not Group defs
        g = (a.get("PartOfGroup") or "").strip().lower()
        pts = np.array([(float(c.get("X")), float(c.get("Y")))
                        for c in a.findall(".//Coordinate")], dtype=np.float64)
        if len(pts) < 3:
            continue
        if g in TUMOUR_GROUPS:
            tum.append(pts)
        elif g in EXCLUDE_GROUPS:
            exc.append(pts)
        else:
            unknown.add(g)
            exc.append(pts)                         # unknown group: exclude, never guess
    return tum, exc, unknown


# ------------------------------------------------------------- tissue detection
def _otsu(vals: np.ndarray) -> int:
    hist = np.bincount(vals.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.nanargmax(sigma_b))


def tissue_mask(slide: openslide.OpenSlide):
    """Otsu on HSV saturation at a low pyramid level. H&E is coloured, glass is not."""
    lvl = min(slide.level_count - 1, 6)
    w, h = slide.level_dimensions[lvl]
    rgb = np.asarray(slide.read_region((0, 0), lvl, (w, h)).convert("RGB"))
    mx, mn = rgb.max(2).astype(np.int16), rgb.min(2).astype(np.int16)
    sat = (mx - mn).astype(np.uint8)
    thr = _otsu(sat)
    # saturated AND not near-black: pen marks and scanner borders are dark
    return (sat > max(thr, 12)) & (mx > 40), lvl, slide.level_downsamples[lvl]


# ------------------------------------------------------------- tile enumeration
def coverage(polys, x0: int, y0: int, step0: int) -> float:
    """Fraction of the tile covered by the polygons, rasterised locally."""
    if not polys:
        return 0.0
    R = 64                                           # raster resolution for the test
    img = Image.new("1", (R, R), 0)
    d = ImageDraw.Draw(img)
    s = R / step0
    hit = False
    for p in polys:
        if p[:, 0].max() < x0 or p[:, 0].min() > x0 + step0:
            continue
        if p[:, 1].max() < y0 or p[:, 1].min() > y0 + step0:
            continue
        d.polygon([((px - x0) * s, (py - y0) * s) for px, py in p], fill=1)
        hit = True
    return float(np.asarray(img).mean()) if hit else 0.0


def grid_tiles(slide, tmask, tds, step0: int, min_tissue: float, stride0: int = 0):
    W0, H0 = slide.dimensions
    stride0 = stride0 or step0
    tb = max(1, int(round(step0 / tds)))
    Ht, Wt = tmask.shape
    keep = []
    for y0 in range(0, H0 - step0 + 1, stride0):
        ty = int(y0 / tds)
        if ty >= Ht:
            break
        row = tmask[ty:min(ty + tb, Ht)]
        if not row.any():
            continue
        for x0 in range(0, W0 - step0 + 1, stride0):
            tx = int(x0 / tds)
            if tx >= Wt:
                break
            b = row[:, tx:min(tx + tb, Wt)]
            if b.size and b.mean() >= min_tissue:
                keep.append((x0, y0))
    return keep


def lesion_tiles(tum, step0: int, W0: int, H0: int, stride0: int = 0):
    """Every grid tile a lesion bbox touches, plus one centred on each centroid."""
    out = set()
    stride0 = stride0 or step0
    for p in tum:
        x1, y1 = p[:, 0].min() - step0, p[:, 1].min() - step0
        x2, y2 = p[:, 0].max(), p[:, 1].max()
        for gx in range(max(0, int(x1 // stride0)), int(x2 // stride0) + 1):
            for gy in range(max(0, int(y1 // stride0)), int(y2 // stride0) + 1):
                out.add((min(gx * stride0, W0 - step0), min(gy * stride0, H0 - step0)))
        # centroid-centred tile: guarantees a hit for lesions smaller than a tile
        cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
        out.add((int(np.clip(cx - step0 / 2, 0, W0 - step0)),
                 int(np.clip(cy - step0 / 2, 0, H0 - step0))))
    return sorted(out)


# --------------------------------------------------------------------- per slide
def process_slide(sid: str, stages, max_neg: int, min_tissue: float,
                  pos_thr: float, neg_thr: float, seed: int,
                  stride_frac: float = 1.0, out_dir=None):
    img_p = SRC / "images" / f"{sid}.tif"
    ann_p = SRC / "annotations" / f"{sid}.xml"
    if not img_p.exists():
        return sid, None, "missing image"
    # This runs CONCURRENTLY with a 2.5 h download, so a .tif may be half written.
    # Two guards: a recent mtime means the file is still growing, and openslide
    # failing to parse means the trailing TIFF directory is not there yet.
    age = time.time() - img_p.stat().st_mtime
    if age < 90:
        return sid, None, f"in flight (written {age:.0f}s ago)"
    try:
        slide = openslide.OpenSlide(str(img_p))
    except openslide.OpenSlideError:
        return sid, None, "unreadable, likely incomplete"
    step0 = int(TILE * slide.level_downsamples[LEVEL])
    W0, H0 = slide.dimensions

    tum, exc, unknown = parse_annotations(ann_p) if ann_p.exists() else ([], [], set())
    stride0 = max(1, int(step0 * stride_frac))
    tmask, _, tds = tissue_mask(slide)
    grid = grid_tiles(slide, tmask, tds, step0, min_tissue, stride0)
    if not grid:
        return sid, None, "no tissue tiles"

    rng = np.random.default_rng(seed)
    pos_cand = lesion_tiles(tum, step0, W0, H0, stride0)
    # negatives: grid tiles far from any lesion, subsampled
    neg_pool = [c for c in grid if c not in set(pos_cand)]
    if len(neg_pool) > max_neg * 3:
        neg_pool = [neg_pool[i] for i in rng.choice(len(neg_pool), max_neg * 3, replace=False)]

    patches, labels, xy, covs = [], [], [], []
    n_amb = n_exc = 0

    def emit(x0, y0, lab, cov):
        a = np.asarray(slide.read_region((x0, y0), LEVEL, (TILE, TILE)).convert("RGB"))
        patches.append(a); labels.append(lab); xy.append((x0, y0)); covs.append(cov)

    for (x0, y0) in pos_cand:
        if coverage(exc, x0, y0, step0) > 0:
            n_exc += 1
            continue
        c = coverage(tum, x0, y0, step0)
        if c >= pos_thr:
            emit(x0, y0, 1, c)
        elif c > neg_thr:
            n_amb += 1

    n_neg = 0
    for (x0, y0) in neg_pool:
        if n_neg >= max_neg:
            break
        if coverage(exc, x0, y0, step0) > 0:
            n_exc += 1
            continue
        c = coverage(tum, x0, y0, step0)
        if c > neg_thr:
            n_amb += 1
            continue
        emit(x0, y0, 0, 0.0)
        n_neg += 1

    if not patches:
        return sid, None, "no usable tiles"
    grade, centre = stages.get(sid, ("unknown", int(sid.split("_")[1]) // 20))
    od = out_dir or OUT
    od.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        od / f"{sid}.npz",
        patches=np.stack(patches).astype(np.uint8),
        labels=np.array(labels, np.int64),
        coords=np.array(xy, np.int64),
        cover=np.array(covs, np.float32),
        centre=np.int64(centre),
        grade=np.str_(grade),
        slide_id=np.str_(sid),
        n_grid=np.int64(len(grid)),
        n_lesion_poly=np.int64(len(tum)),
        n_ambiguous=np.int64(n_amb),
        n_excluded=np.int64(n_exc),
        mpp_level0=np.float64(float(slide.properties.get("openslide.mpp-x", 0) or 0)),
        stride0=np.int64(stride0),
    )
    return sid, (len(patches), int(np.sum(labels)), n_amb, n_exc, grade, centre,
                 len(tum), sorted(unknown)), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-neg", type=int, default=600, help="negative tiles per slide")
    ap.add_argument("--min-tissue", type=float, default=0.50)
    ap.add_argument("--pos-thr", type=float, default=0.05,
                    help="tile tumour coverage to call positive; 0.05 keeps ITC learnable")
    ap.add_argument("--neg-thr", type=float, default=0.0,
                    help="above this and below pos-thr = discarded as ambiguous")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--stride-frac", type=float, default=1.0,
                    help="tile stride as a fraction of tile size; <1 oversamples")
    ap.add_argument("--out-dir", default="")
    a = ap.parse_args()

    stages = load_stages() if (SRC / "stages.csv").exists() else {}
    ids = sorted(p.stem for p in (SRC / "images").glob("*.tif"))
    if a.only:
        ids = [i for i in ids if i in set(a.only.split(","))]
    od = Path(a.out_dir) if a.out_dir else OUT
    todo = [i for i in ids if not (od / f"{i}.npz").exists()]
    print(f"[wsi] {len(ids)} slides present, {len(todo)} to tile | level {LEVEL} "
          f"({TILE}px = {TILE * 0.486:.0f} um) | pos_thr {a.pos_thr} | max_neg {a.max_neg}",
          flush=True)

    t0, tot, pos, amb, exc = time.time(), 0, 0, 0, 0
    unknown_seen: set[str] = set()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process_slide, i, stages, a.max_neg, a.min_tissue,
                          a.pos_thr, a.neg_thr, abs(hash(i)) % 2**31,
                          a.stride_frac, od): i for i in todo}
        for n, f in enumerate(cf.as_completed(futs), 1):
            sid, st, err = f.result()
            if err:
                print(f"[{n:3d}/{len(todo)}] {sid:22s} SKIP {err}", flush=True)
                continue
            k, p, am, ec, grade, centre, npoly, unk = st
            tot += k; pos += p; amb += am; exc += ec
            unknown_seen |= set(unk)
            print(f"[{n:3d}/{len(todo)}] {sid:22s} c{centre} {grade:8s} "
                  f"{npoly:3d} poly -> {k:5d} tiles ({p:4d} tumour, {am:4d} amb, "
                  f"{ec:3d} excl)  {time.time() - t0:6.0f}s", flush=True)

    print(f"\n[wsi] {tot} tiles, {pos} tumour ({pos / max(tot, 1):.1%}), "
          f"{amb} ambiguous discarded, {exc} excluded, {(time.time() - t0) / 60:.1f} min")
    if unknown_seen:
        print(f"[wsi] WARNING unknown annotation groups treated as exclusion: {unknown_seen}")


if __name__ == "__main__":
    main()
