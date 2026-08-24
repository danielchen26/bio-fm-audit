#!/usr/bin/env python3
"""
Acquire ONE paired (H&E image patch, gene expression vector) spatial-transcriptomics
sample and record the exact on-disk schema.

Step 1: inspect MahmoodLab/hest (HEST-1k) on the HuggingFace Hub WITHOUT downloading
        it, and report its layout. HEST-1k is a *gated* dataset; if file download
        returns 401/403 we report that explicitly and fall back to an openly
        downloadable 10x Genomics Visium sample.

Step 2/3: build the paired sample -> data/hest_sample.npz
Step 4:   document the schema      -> data/hest_schema.md

Run:  .venv/bin/python src/get_hest.py
"""

from __future__ import annotations

import collections
import json
import posixpath
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # full-resolution WSIs exceed PIL's decompression-bomb guard

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw_10x"

HEST_REPO = "MahmoodLab/hest"

# Openly downloadable fallback: 10x Genomics Visium Spatial Gene Expression,
# "Human Lymph Node", Space Ranger 1.1.0, H&E stained fresh-frozen section.
TENX_SAMPLE_ID = "V1_Human_Lymph_Node"
TENX_BASE = f"https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/{TENX_SAMPLE_ID}"
TENX_FILES = [
    f"{TENX_SAMPLE_ID}_filtered_feature_bc_matrix.h5",
    f"{TENX_SAMPLE_ID}_spatial.tar.gz",
    f"{TENX_SAMPLE_ID}_image.tif",
]

PATCH = 96          # output patch edge length in full-resolution pixels
N_GENES = 256       # number of most-variable genes to keep


# --------------------------------------------------------------------------- #
# Step 1 - inspect HEST-1k on the Hub
# --------------------------------------------------------------------------- #
def inspect_hest() -> dict:
    """List HEST-1k repo files and probe whether the data is actually fetchable.

    Returns a dict describing the layout and the access status.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
    )

    api = HfApi()
    report: dict = {"repo": HEST_REPO}

    print("=" * 78)
    print(f"STEP 1: inspecting HuggingFace dataset '{HEST_REPO}' (HEST-1k)")
    print("=" * 78)

    try:
        info = api.dataset_info(HEST_REPO)
        report["gated"] = info.gated
        report["private"] = info.private
        print(f"  dataset_info: gated={info.gated}  private={info.private}")
    except Exception as exc:  # noqa: BLE001 - we want to report anything here
        report["info_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  dataset_info FAILED: {type(exc).__name__}: {exc}")

    try:
        files = api.list_repo_files(HEST_REPO, repo_type="dataset")
    except Exception as exc:  # noqa: BLE001
        report["list_error"] = f"{type(exc).__name__}: {exc}"
        print(f"  list_repo_files FAILED: {type(exc).__name__}: {exc}")
        report["accessible"] = False
        return report

    report["n_files"] = len(files)
    top = collections.Counter(f.split("/")[0] if "/" in f else "<root>" for f in files)
    ext = collections.Counter(posixpath.splitext(f)[1] or "<none>" for f in files)
    report["top_level"] = dict(top)
    report["extensions"] = dict(ext)

    print(f"  list_repo_files OK: {len(files)} files")
    print("\n  top-level directory layout (dir -> file count):")
    for k, v in top.most_common():
        example = next((f for f in files if f.startswith(k + "/")), k)
        print(f"    {k:20s} {v:6d}   e.g. {example}")
    print("\n  file-type counts:")
    for k, v in ext.most_common():
        print(f"    {k:12s} {v:6d}")

    # Probe a single tiny file to see whether the *content* is actually reachable.
    probe = "metadata/INT1.json" if "metadata/INT1.json" in files else files[0]
    try:
        hf_hub_download(
            HEST_REPO, probe, repo_type="dataset", cache_dir=str(DATA / "hf_cache")
        )
        report["accessible"] = True
        print(f"\n  download probe '{probe}': OK -> HEST-1k content is accessible")
    except (GatedRepoError, RepositoryNotFoundError, HfHubHTTPError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        report["accessible"] = False
        report["download_error"] = f"{type(exc).__name__}: {exc}"
        report["download_status"] = status
        print(f"\n  download probe '{probe}': FAILED")
        print(f"    error  : {type(exc).__name__}")
        print(f"    status : HTTP {status}")
        print(f"    detail : {str(exc).splitlines()[0]}")
        print("  -> HEST-1k is GATED. Falling back to an open 10x Genomics Visium sample.")
    return report


# --------------------------------------------------------------------------- #
# Step 2 - obtain the open fallback sample
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest.name}  ({dest.stat().st_size:,} bytes)")
        return
    print(f"  GET     {url}")
    with requests.get(url, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code} for {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
        tmp.rename(dest)
    print(f"  saved   {dest.name}  ({dest.stat().st_size:,} bytes)")


def fetch_tenx() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print(f"STEP 2: downloading open 10x Visium sample '{TENX_SAMPLE_ID}'")
    print("=" * 78)
    for name in TENX_FILES:
        download(f"{TENX_BASE}/{name}", RAW / name)
    spatial_dir = RAW / "spatial"
    if not (spatial_dir / "scalefactors_json.json").exists():
        with tarfile.open(RAW / f"{TENX_SAMPLE_ID}_spatial.tar.gz") as tf:
            tf.extractall(RAW)
        print(f"  extracted {spatial_dir}")


def read_tissue_positions(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Parse a Space Ranger tissue_positions file.

    Space Ranger <= 1.3 writes `tissue_positions_list.csv` with NO header row;
    >= 2.0 writes `tissue_positions.csv` WITH a header. Column order is identical:

        barcode, in_tissue, array_row, array_col,
        pxl_row_in_fullres (= y), pxl_col_in_fullres (= x)

    Returns (barcodes, in_tissue, xy) where xy[:, 0] = x = pxl_col_in_fullres
    and xy[:, 1] = y = pxl_row_in_fullres, in FULL-RESOLUTION image pixels.
    """
    lines = path.read_text().strip().splitlines()
    if lines[0].lower().startswith("barcode"):
        lines = lines[1:]
    barcodes, in_tissue, xy = [], [], []
    for line in lines:
        bc, it, _row, _col, pxl_row, pxl_col = line.split(",")
        barcodes.append(bc)
        in_tissue.append(int(it))
        xy.append((float(pxl_col), float(pxl_row)))  # (x, y)
    return barcodes, np.asarray(in_tissue, dtype=np.int8), np.asarray(xy, dtype=np.float64)


def read_cellranger_h5(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read a CellRanger filtered_feature_bc_matrix.h5.

    On disk the matrix is CSC with shape (n_genes, n_barcodes): `matrix/indptr`
    has length n_barcodes + 1, so each "column" is one spot. We densify into a
    spots x genes array (the orientation everything downstream expects).
    """
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        n_genes, n_barcodes = (int(v) for v in m["shape"][:])
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        barcodes = [b.decode() for b in m["barcodes"][:]]
        gene_names = np.array([g.decode() for g in m["features/name"][:]])
        assert len(indptr) == n_barcodes + 1, "expected CSC (columns = barcodes)"

    counts = np.zeros((n_barcodes, n_genes), dtype=np.float32)
    for j in range(n_barcodes):
        lo, hi = indptr[j], indptr[j + 1]
        counts[j, indices[lo:hi]] = data[lo:hi]
    return counts, barcodes, gene_names


def build_sample() -> dict:
    print("=" * 78)
    print("STEP 3: building paired (patch, expression) sample")
    print("=" * 78)

    counts, mat_barcodes, gene_names = read_cellranger_h5(
        RAW / f"{TENX_SAMPLE_ID}_filtered_feature_bc_matrix.h5"
    )
    n_spots_raw, n_genes_raw = counts.shape
    print(f"  expression matrix : {n_spots_raw} spots x {n_genes_raw} genes (densified, float32)")

    scalefactors = json.loads((RAW / "spatial" / "scalefactors_json.json").read_text())
    spot_diameter_fullres = float(scalefactors["spot_diameter_fullres"])
    print(f"  scalefactors      : {scalefactors}")

    pos_barcodes, in_tissue, xy_all = read_tissue_positions(
        RAW / "spatial" / "tissue_positions_list.csv"
    )
    n_positions = len(pos_barcodes)
    n_in_tissue = int((in_tissue == 1).sum())
    print(f"  tissue positions  : {n_positions} rows total, {n_in_tissue} with in_tissue=1")

    # The filtered matrix contains exactly the in-tissue barcodes; align by barcode.
    pos_index = {bc: i for i, bc in enumerate(pos_barcodes)}
    missing = [bc for bc in mat_barcodes if bc not in pos_index]
    if missing:
        raise RuntimeError(f"{len(missing)} matrix barcodes absent from tissue positions")
    order = np.array([pos_index[bc] for bc in mat_barcodes], dtype=np.int64)
    coords_all = xy_all[order]  # (x, y) full-res pixels, matrix-barcode order
    assert (in_tissue[order] == 1).all(), "filtered matrix should be in-tissue only"

    img = np.asarray(Image.open(RAW / f"{TENX_SAMPLE_ID}_image.tif").convert("RGB"))
    H, W = img.shape[:2]
    print(f"  H&E full-res image: {W} x {H} px (W x H), dtype={img.dtype}")

    # Keep only spots whose full 96x96 crop lies inside the image.
    half = PATCH // 2
    cx = np.rint(coords_all[:, 0]).astype(np.int64)
    cy = np.rint(coords_all[:, 1]).astype(np.int64)
    x0, y0 = cx - half, cy - half
    keep = (x0 >= 0) & (y0 >= 0) & (x0 + PATCH <= W) & (y0 + PATCH <= H)
    n_dropped = int((~keep).sum())
    print(f"  spots dropped for out-of-bounds {PATCH}x{PATCH} crop: {n_dropped}")

    counts = counts[keep]
    coords = coords_all[keep]
    x0, y0 = x0[keep], y0[keep]
    n_spots = counts.shape[0]

    patches = np.empty((n_spots, PATCH, PATCH, 3), dtype=np.uint8)
    for i in range(n_spots):
        patches[i] = img[y0[i]: y0[i] + PATCH, x0[i]: x0[i] + PATCH]
    del img

    # --- most-variable gene selection -------------------------------------- #
    # 1. library-size normalise each spot to 10,000 counts (CP10K)
    # 2. log1p transform
    # 3. rank genes by sample variance (ddof=1) of the log1p-CP10K values across spots
    lib = counts.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    norm = np.log1p(counts / lib * 1e4).astype(np.float32)
    variance = norm.var(axis=0, ddof=1)
    top_idx = np.argsort(variance)[::-1][:N_GENES]
    top_idx = np.sort(top_idx)  # keep genomic/file order for reproducibility

    expr = np.ascontiguousarray(norm[:, top_idx], dtype=np.float32)
    top_genes = np.asarray(gene_names[top_idx], dtype=np.str_)

    out = {
        "patches": patches,
        "expr": expr,
        "gene_names": top_genes,
        "coords": np.ascontiguousarray(coords, dtype=np.float32),
        "sample_id": np.array(TENX_SAMPLE_ID, dtype=np.str_),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    npz_path = DATA / "hest_sample.npz"
    np.savez_compressed(npz_path, **out)
    print(f"  wrote {npz_path} ({npz_path.stat().st_size:,} bytes)")

    return {
        "n_spots": n_spots,
        "n_spots_raw": n_spots_raw,
        "n_genes_raw": n_genes_raw,
        "n_positions": n_positions,
        "n_in_tissue": n_in_tissue,
        "n_dropped": n_dropped,
        "image_w": W,
        "image_h": H,
        "spot_diameter_fullres": spot_diameter_fullres,
        "scalefactors": scalefactors,
        "var_min_kept": float(variance[top_idx].min()),
        "var_max_kept": float(variance[top_idx].max()),
        "npz_bytes": npz_path.stat().st_size,
        "npz_path": npz_path,
    }


# --------------------------------------------------------------------------- #
# Step 4 - schema documentation
# --------------------------------------------------------------------------- #
def write_schema(hest: dict, s: dict) -> Path:
    sf = s["scalefactors"]
    hires = sf["tissue_hires_scalef"]
    lowres = sf["tissue_lowres_scalef"]

    if hest.get("accessible"):
        hest_status = "HEST-1k content was downloadable."
    else:
        hest_status = (
            f"HEST-1k downloads FAILED with HTTP {hest.get('download_status')} "
            f"({hest.get('download_error', '').split(':')[0]}); the repo is gated "
            f"(`gated={hest.get('gated')}`). File *listing* succeeded anonymously "
            f"({hest.get('n_files')} files), but fetching any blob is refused unless "
            "the account is on the authorized list."
        )

    layout = "\n".join(
        f"| `{k}/` | {v} | {d} |"
        for k, v, d in [
            ("wsis", hest["top_level"].get("wsis", 0), "`<ID>.tif` full-resolution pyramidal H&E WSI"),
            ("st", hest["top_level"].get("st", 0), "`<ID>.h5ad` AnnData: expression + spot coords + downscaled image"),
            ("patches", hest["top_level"].get("patches", 0), "`<ID>.h5` pre-extracted 224x224 patches at 0.5 um/px"),
            ("metadata", hest["top_level"].get("metadata", 0), "`<ID>.json` metrics incl. `pixel_size_um_estimated`"),
            ("thumbnails", hest["top_level"].get("thumbnails", 0), "`<ID>_downscaled_fullres.jpeg`"),
            ("tissue_seg", hest["top_level"].get("tissue_seg", 0), "`<ID>_contours.geojson` + `<ID>_vis.jpg`"),
            ("cellvit_seg", hest["top_level"].get("cellvit_seg", 0), "`<ID>_cellvit_seg.parquet` / `.geojson.zip`"),
            ("xenium_seg", hest["top_level"].get("xenium_seg", 0), "Xenium cell/nucleus segmentation"),
            ("transcripts", hest["top_level"].get("transcripts", 0), "`<ID>_transcripts.parquet` aligned Xenium transcripts"),
            ("spatial_plots", hest["top_level"].get("spatial_plots", 0), "QC plot `<ID>_spatial_plots.png`"),
            ("pixel_size_vis", hest["top_level"].get("pixel_size_vis", 0), "QC plot `<ID>_pixel_size_vis.png`"),
            ("patches_vis", hest["top_level"].get("patches_vis", 0), "QC plot `<ID>_patch_vis.jpg`"),
        ]
    )
    exts = "\n".join(f"| `{k}` | {v} |" for k, v in sorted(hest["extensions"].items(), key=lambda kv: -kv[1]))

    md = f"""# Spatial transcriptomics + H&E paired-sample schema

## 0. Which source actually worked

**Intended source:** HEST-1k — <https://huggingface.co/datasets/MahmoodLab/hest>
(HF dataset `MahmoodLab/hest`, 1,276 paired ST + H&E WSI profiles, CC-BY-NC-SA-4.0).

**Result:** {hest_status}

**Source actually used for `hest_sample.npz`:**
10x Genomics *Visium Spatial Gene Expression, Human Lymph Node* (Space Ranger 1.1.0),
sample id `{TENX_SAMPLE_ID}`, an openly downloadable H&E + Visium pair.

- Landing page: <https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0>
- Files pulled from `{TENX_BASE}/`:

| File | Bytes | Content |
|---|---|---|
| `{TENX_SAMPLE_ID}_filtered_feature_bc_matrix.h5` | {(RAW / TENX_FILES[0]).stat().st_size:,} | CellRanger HDF5 count matrix (in-tissue spots only) |
| `{TENX_SAMPLE_ID}_spatial.tar.gz` | {(RAW / TENX_FILES[1]).stat().st_size:,} | `spatial/` folder (positions, scale factors, downscaled images) |
| `{TENX_SAMPLE_ID}_image.tif` | {(RAW / TENX_FILES[2]).stat().st_size:,} | full-resolution H&E, uncompressed RGB TIFF |

Extracted `spatial/` contents: `tissue_positions_list.csv`, `scalefactors_json.json`,
`tissue_hires_image.png`, `tissue_lowres_image.png`, `aligned_fiducials.jpg`,
`detected_tissue_image.jpg`.

## 1. HEST-1k on-disk schema (documented from the Hub listing + HEST-Library source)

`HfApi().list_repo_files('MahmoodLab/hest', repo_type='dataset')` returns
**{hest.get('n_files')} files**. Every sample is keyed by a short id (`INT1`, `TENX95`,
`NCBI783`, …) and the same id appears in each directory:

| Directory | Files | Per-sample content |
|---|---|---|
{layout}

Root files: `HEST_v1_1_0.csv` … `HEST_v1_3_0.csv` (per-release sample metadata tables),
`README.md`, `hest.py`, `human_gene_db.parquet`, `fig1a.{{jpg,png}}`, `fig1_preview.jpg`.

File-type counts across the repo:

| Extension | Count |
|---|---|
{exts}

A HEST sample is loaded as `HESTData(adata, img, pixel_size, meta)` where
(`src/hest/HESTData.py`, `HESTData.from_paths`):

- `st/<ID>.h5ad` → `scanpy.AnnData`. `adata.X` is spots x genes; spot pixel
  coordinates on the **full-resolution** WSI live in `adata.obsm['spatial']`
  (`(x, y)` centres); a downscaled image is required at
  `adata.uns['spatial']['ST']['images']['downscaled_fullres']` with its factor at
  `adata.uns['spatial']['ST']['scalefactors']['tissue_downscaled_fullres_scalef']`.
- `wsis/<ID>.tif` → the aligned full-resolution H&E.
- `metadata/<ID>.json` → `metrics['pixel_size_um_estimated']` (um/px) is the
  authoritative scale; also `fullres_px_width` / `fullres_px_height` / `adata_nb_col`.
- `patches/<ID>.h5` → written by `HESTData.dump_patches()` with datasets
  `img` (uint8, `[n_spots, 224, 224, 3]`), `coords` (int32, `[n_spots, 2]`, **top-left**
  corner in level-0 px) and `barcodes` (fixed-width bytes). `coords` carries attributes
  `patch_size`, `patch_size_level0`, `level0_magnification`, `target_magnification`,
  `level0_width`, `level0_height`. Patches are cropped at a target of **0.5 um/px**, so
  the level-0 crop side is `224 * (0.5 / pixel_size_um_estimated)` and the top-left is
  `adata.obsm['spatial'] - patch_size_src // 2`.

The `.npz` produced here mirrors that idea (spot-centred crops + expression matrix),
just at 96x96 px and with the open 10x sample.

## 2. Files and formats actually parsed

### `{TENX_SAMPLE_ID}_filtered_feature_bc_matrix.h5`
HDF5, `software_version = spaceranger-1.1.0`, `chemistry = Spatial 3' v1`, genome `GRCh38`.

| Dataset | Shape | dtype |
|---|---|---|
| `matrix/shape` | `(2,)` = `[{s['n_genes_raw']}, {s['n_spots_raw']}]` | int32 |
| `matrix/data` | `(23717106,)` | int32 |
| `matrix/indices` | `(23717106,)` | int64 |
| `matrix/indptr` | `({s['n_spots_raw'] + 1},)` | int64 |
| `matrix/barcodes` | `({s['n_spots_raw']},)` | `\\|S18` |
| `matrix/features/name` | `({s['n_genes_raw']},)` | `\\|S17` (HGNC symbol) |
| `matrix/features/id` | `({s['n_genes_raw']},)` | `\\|S15` (Ensembl id) |
| `matrix/features/feature_type` | `({s['n_genes_raw']},)` | all `Gene Expression` |

**Orientation:** the stored matrix is **CSC with shape (genes, barcodes)** —
`len(indptr) == n_barcodes + 1`, so every CSC column is one spot and `indices` holds
gene rows. Raw values are **integer UMI counts** (`int32`). This script densifies to a
**spots x genes** `float32` array before any filtering.

### `spatial/scalefactors_json.json`
```json
{json.dumps(sf, indent=2)}
```

### `spatial/tissue_positions_list.csv`
Headerless CSV (Space Ranger <= 1.3; >= 2.0 renames it `tissue_positions.csv` and adds a
header, same column order), {s['n_positions']} rows = every spot on the Visium slide:

| # | Column | Meaning |
|---|---|---|
| 0 | `barcode` | e.g. `ACGCCTGACACGCGCT-1` |
| 1 | `in_tissue` | 1 if under tissue, else 0 |
| 2 | `array_row` | Visium array row (0-77) |
| 3 | `array_col` | Visium array column (0-127) |
| 4 | `pxl_row_in_fullres` | **y** pixel on the full-res image |
| 5 | `pxl_col_in_fullres` | **x** pixel on the full-res image |

{s['n_in_tissue']} rows have `in_tissue == 1`, which matches the {s['n_spots_raw']}
barcodes in the filtered matrix exactly (aligned here by barcode string, not by row order).

### `{TENX_SAMPLE_ID}_image.tif`
Uncompressed (`raw`) single-page RGB TIFF, **{s['image_w']} x {s['image_h']} px (W x H)**.

## 3. Coordinate -> pixel mapping (re-derivable)

Columns 4/5 of `tissue_positions_list.csv` are **already in full-resolution image
pixels**; no scale factor is needed for the full-res TIFF. The only trap is the order:
the file stores **row (y) before column (x)**, while `coords` in the `.npz` is `(x, y)`.

```
x_fullres = float(row[5])   # pxl_col_in_fullres
y_fullres = float(row[4])   # pxl_row_in_fullres
img[y_fullres, x_fullres]   # numpy indexing is [row, col] = [y, x]
```

To index the downscaled images shipped in `spatial/`, multiply by the factor from
`scalefactors_json.json` (that is where the factors come from, written by Space Ranger):

```
x_hires  = x_fullres * tissue_hires_scalef   # {hires}
y_hires  = y_fullres * tissue_hires_scalef
x_lowres = x_fullres * tissue_lowres_scalef  # {lowres}
y_lowres = y_fullres * tissue_lowres_scalef
```

**Patch extraction** (identical convention to HEST's `dump_patches`, which uses
`coords_topleft = coords_center - patch_size // 2`):

```
half = 96 // 2                                 # 48
x0   = round(x_fullres) - half
y0   = round(y_fullres) - half
patch = img[y0 : y0 + 96, x0 : x0 + 96]        # (96, 96, 3) uint8
```

**Spot diameter:** `spot_diameter_fullres = {s['spot_diameter_fullres']}` px
(full-resolution pixels, from `scalefactors_json.json`). A Visium spot is 55 um across,
so the implied resolution is 55 / {s['spot_diameter_fullres']:.4f} =
**{55.0 / s['spot_diameter_fullres']:.6f} um/px**, and the 96x96 px patch covers
96 * {55.0 / s['spot_diameter_fullres']:.6f} = **{96 * 55.0 / s['spot_diameter_fullres']:.2f} um** per side —
i.e. the patch is slightly wider than one spot ({96 / s['spot_diameter_fullres']:.3f} spot diameters).
Spot centre-to-centre pitch is 100 um.

## 4. Counts before and after filtering

| Quantity | Value |
|---|---|
| Spots on slide (`tissue_positions_list.csv` rows) | {s['n_positions']} |
| Spots with `in_tissue == 1` | {s['n_in_tissue']} |
| Spots in filtered matrix (raw sample) | {s['n_spots_raw']} |
| Spots dropped (96x96 crop would leave the image) | {s['n_dropped']} |
| **Spots kept in `hest_sample.npz`** | **{s['n_spots']}** |
| Genes in raw matrix (before filtering) | {s['n_genes_raw']} |
| **Genes kept** | **{N_GENES}** |
| Full-res image size (W x H) | {s['image_w']} x {s['image_h']} |
| `spot_diameter_fullres` | {s['spot_diameter_fullres']} px |

### How "most variable" was computed
1. Densify the CSC matrix to `spots x genes` `float32` **raw UMI counts**.
2. Library-size normalise each spot to 10,000 counts: `counts / counts.sum(axis=1, keepdims=True) * 1e4`
   (spots with zero total counts are left divided by 1 — none occur here).
3. `log1p` transform.
4. Per-gene **sample variance across spots** (`norm.var(axis=0, ddof=1)`).
5. Take the {N_GENES} genes with the largest variance, then re-sort those indices into
   original file (genomic) order so the output is deterministic.

Kept-gene variance range: **{s['var_min_kept']:.6f}** (lowest kept) to
**{s['var_max_kept']:.6f}** (highest).

## 5. `data/hest_sample.npz` contents

Written with `numpy.savez_compressed` ({s['npz_bytes']:,} bytes).

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `patches` | `uint8` | `({s['n_spots']}, 96, 96, 3)` | RGB H&E crop centred on each spot, full-res scale |
| `expr` | `float32` | `({s['n_spots']}, {N_GENES})` | log1p(CP10K) expression, spot-major, column j = `gene_names[j]` |
| `gene_names` | `<U...` (unicode) | `({N_GENES},)` | HGNC symbols from `matrix/features/name` |
| `coords` | `float32` | `({s['n_spots']}, 2)` | `(x, y)` spot centre in full-res image pixels |
| `sample_id` | `<U{len(TENX_SAMPLE_ID)}` (unicode) | `()` scalar | `"{TENX_SAMPLE_ID}"` |

Row *i* of `patches`, `expr` and `coords` all refer to the same spot, in the barcode
order of `matrix/barcodes` (minus dropped out-of-bounds spots).

```python
import numpy as np
d = np.load("data/hest_sample.npz")
d["patches"].shape, d["expr"].shape, str(d["sample_id"])
```

## 6. Reproducing
```
uv venv --python 3.12
uv pip install numpy pillow pandas requests huggingface_hub h5py
.venv/bin/python src/get_hest.py
```
"""
    path = DATA / "hest_schema.md"
    path.write_text(md)
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")
    return path


# --------------------------------------------------------------------------- #
def verify(s: dict) -> None:
    print("=" * 78)
    print("VERIFICATION: reloading data/hest_sample.npz")
    print("=" * 78)
    d = np.load(DATA / "hest_sample.npz", allow_pickle=False)
    expected = {
        "patches": (np.uint8, (s["n_spots"], PATCH, PATCH, 3)),
        "expr": (np.float32, (s["n_spots"], N_GENES)),
        "gene_names": (np.str_, (N_GENES,)),
        "coords": (np.float32, (s["n_spots"], 2)),
        "sample_id": (np.str_, ()),
    }
    assert sorted(d.files) == sorted(expected), f"array names mismatch: {d.files}"
    for name, (dt, shape) in expected.items():
        a = d[name]
        ok_dtype = np.issubdtype(a.dtype, dt)
        ok_shape = a.shape == shape
        print(f"  {name:11s} dtype={str(a.dtype):8s} shape={str(a.shape):22s} "
              f"dtype_ok={ok_dtype} shape_ok={ok_shape}")
        assert ok_dtype, f"{name}: dtype {a.dtype} not a {dt}"
        assert ok_shape, f"{name}: shape {a.shape} != {shape}"
    print(f"  sample_id value   : {str(d['sample_id'])!r}")
    print(f"  gene_names[:6]    : {list(d['gene_names'][:6])}")
    print(f"  patches value rng : {int(d['patches'].min())}..{int(d['patches'].max())}")
    print(f"  expr value rng    : {float(d['expr'].min()):.4f}..{float(d['expr'].max()):.4f}")
    print(f"  coords x rng      : {float(d['coords'][:,0].min()):.1f}..{float(d['coords'][:,0].max()):.1f}")
    print(f"  coords y rng      : {float(d['coords'][:,1].min()):.1f}..{float(d['coords'][:,1].max()):.1f}")


def main() -> int:
    hest = inspect_hest()
    if hest.get("accessible"):
        print("\nNOTE: HEST-1k is reachable, but this script's paired-sample builder "
              "targets the open 10x Visium layout; continuing with the 10x sample.\n")
    fetch_tenx()
    s = build_sample()
    write_schema(hest, s)
    verify(s)

    print()
    print("=" * 78)
    print("FINAL SUMMARY (exact observed counts)")
    print("=" * 78)
    print(f"  HEST-1k ({HEST_REPO}) listing        : {hest.get('n_files')} files, "
          f"gated={hest.get('gated')}")
    dl_status = "OK" if hest.get("accessible") else f"FAILED HTTP {hest.get('download_status')}"
    print(f"  HEST-1k blob download                : {dl_status}")
    print(f"  Source used                          : 10x Visium {TENX_SAMPLE_ID}")
    print(f"  H&E full-res image (W x H)           : {s['image_w']} x {s['image_h']} px")
    print(f"  Spot positions on slide              : {s['n_positions']}")
    print(f"  Spots in_tissue=1                    : {s['n_in_tissue']}")
    print(f"  n_spots RAW (filtered matrix)        : {s['n_spots_raw']}")
    print(f"  n_spots dropped (crop out of bounds) : {s['n_dropped']}")
    print(f"  n_spots FINAL in npz                 : {s['n_spots']}")
    print(f"  Genes RAW before filtering           : {s['n_genes_raw']}")
    print(f"  Genes kept (most variable)           : {N_GENES}")
    print(f"  spot_diameter_fullres                : {s['spot_diameter_fullres']} px")
    print(f"  implied resolution (55 um / diameter): {55.0 / s['spot_diameter_fullres']:.6f} um/px")
    print(f"  patch size                           : {PATCH} x {PATCH} px "
          f"= {96 * 55.0 / s['spot_diameter_fullres']:.2f} x {96 * 55.0 / s['spot_diameter_fullres']:.2f} um")
    print(f"  data/hest_sample.npz                 : {s['npz_bytes']:,} bytes")
    print(f"  arrays: patches uint8 ({s['n_spots']}, 96, 96, 3) | "
          f"expr float32 ({s['n_spots']}, {N_GENES}) | gene_names <U ({N_GENES},) | "
          f"coords float32 ({s['n_spots']}, 2) | sample_id <U scalar")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
