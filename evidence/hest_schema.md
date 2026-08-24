# Spatial transcriptomics + H&E paired-sample schema

## 0. Which source actually worked

**Intended source:** HEST-1k — <https://huggingface.co/datasets/MahmoodLab/hest>
(HF dataset `MahmoodLab/hest`, 1,276 paired ST + H&E WSI profiles, CC-BY-NC-SA-4.0).

**Result:** HEST-1k downloads FAILED with HTTP 403 (GatedRepoError); the repo is gated (`gated=auto`). File *listing* succeeded anonymously (15608 files), but fetching any blob is refused unless the account is on the authorized list.

**Source actually used for `hest_sample.npz`:**
10x Genomics *Visium Spatial Gene Expression, Human Lymph Node* (Space Ranger 1.1.0),
sample id `V1_Human_Lymph_Node`, an openly downloadable H&E + Visium pair.

- Landing page: <https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0>
- Files pulled from `https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Human_Lymph_Node/`:

| File | Bytes | Content |
|---|---|---|
| `V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5` | 30,736,856 | CellRanger HDF5 count matrix (in-tissue spots only) |
| `V1_Human_Lymph_Node_spatial.tar.gz` | 8,246,238 | `spatial/` folder (positions, scale factors, downscaled images) |
| `V1_Human_Lymph_Node_image.tif` | 398,416,526 | full-resolution H&E, uncompressed RGB TIFF |

Extracted `spatial/` contents: `tissue_positions_list.csv`, `scalefactors_json.json`,
`tissue_hires_image.png`, `tissue_lowres_image.png`, `aligned_fiducials.jpg`,
`detected_tissue_image.jpg`.

## 1. HEST-1k on-disk schema (documented from the Hub listing + HEST-Library source)

`HfApi().list_repo_files('MahmoodLab/hest', repo_type='dataset')` returns
**15608 files**. Every sample is keyed by a short id (`INT1`, `TENX95`,
`NCBI783`, …) and the same id appears in each directory:

| Directory | Files | Per-sample content |
|---|---|---|
| `wsis/` | 1276 | `<ID>.tif` full-resolution pyramidal H&E WSI |
| `st/` | 1276 | `<ID>.h5ad` AnnData: expression + spot coords + downscaled image |
| `patches/` | 1276 | `<ID>.h5` pre-extracted 224x224 patches at 0.5 um/px |
| `metadata/` | 1276 | `<ID>.json` metrics incl. `pixel_size_um_estimated` |
| `thumbnails/` | 1276 | `<ID>_downscaled_fullres.jpeg` |
| `tissue_seg/` | 2405 | `<ID>_contours.geojson` + `<ID>_vis.jpg` |
| `cellvit_seg/` | 2552 | `<ID>_cellvit_seg.parquet` / `.geojson.zip` |
| `xenium_seg/` | 344 | Xenium cell/nucleus segmentation |
| `transcripts/` | 86 | `<ID>_transcripts.parquet` aligned Xenium transcripts |
| `spatial_plots/` | 1276 | QC plot `<ID>_spatial_plots.png` |
| `pixel_size_vis/` | 1276 | QC plot `<ID>_pixel_size_vis.png` |
| `patches_vis/` | 1277 | QC plot `<ID>_patch_vis.jpg` |

Root files: `HEST_v1_1_0.csv` … `HEST_v1_3_0.csv` (per-release sample metadata tables),
`README.md`, `hest.py`, `human_gene_db.parquet`, `fig1a.{jpg,png}`, `fig1_preview.jpg`.

File-type counts across the repo:

| Extension | Count |
|---|---|
| `.png` | 2722 |
| `.jpg` | 2239 |
| `.parquet` | 1535 |
| `.zip` | 1448 |
| `.json` | 1276 |
| `.h5` | 1276 |
| `.h5ad` | 1276 |
| `.jpeg` | 1276 |
| `.geojson` | 1276 |
| `.tif` | 1276 |
| `.csv` | 5 |
| `<none>` | 1 |
| `.md` | 1 |
| `.py` | 1 |

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

### `V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5`
HDF5, `software_version = spaceranger-1.1.0`, `chemistry = Spatial 3' v1`, genome `GRCh38`.

| Dataset | Shape | dtype |
|---|---|---|
| `matrix/shape` | `(2,)` = `[36601, 4035]` | int32 |
| `matrix/data` | `(23717106,)` | int32 |
| `matrix/indices` | `(23717106,)` | int64 |
| `matrix/indptr` | `(4036,)` | int64 |
| `matrix/barcodes` | `(4035,)` | `\|S18` |
| `matrix/features/name` | `(36601,)` | `\|S17` (HGNC symbol) |
| `matrix/features/id` | `(36601,)` | `\|S15` (Ensembl id) |
| `matrix/features/feature_type` | `(36601,)` | all `Gene Expression` |

**Orientation:** the stored matrix is **CSC with shape (genes, barcodes)** —
`len(indptr) == n_barcodes + 1`, so every CSC column is one spot and `indices` holds
gene rows. Raw values are **integer UMI counts** (`int32`). This script densifies to a
**spots x genes** `float32` array before any filtering.

### `spatial/scalefactors_json.json`
```json
{
  "spot_diameter_fullres": 89.48996643177776,
  "tissue_hires_scalef": 0.17011142,
  "fiducial_diameter_fullres": 144.56071500517945,
  "tissue_lowres_scalef": 0.051033426
}
```

### `spatial/tissue_positions_list.csv`
Headerless CSV (Space Ranger <= 1.3; >= 2.0 renames it `tissue_positions.csv` and adds a
header, same column order), 4992 rows = every spot on the Visium slide:

| # | Column | Meaning |
|---|---|---|
| 0 | `barcode` | e.g. `ACGCCTGACACGCGCT-1` |
| 1 | `in_tissue` | 1 if under tissue, else 0 |
| 2 | `array_row` | Visium array row (0-77) |
| 3 | `array_col` | Visium array column (0-127) |
| 4 | `pxl_row_in_fullres` | **y** pixel on the full-res image |
| 5 | `pxl_col_in_fullres` | **x** pixel on the full-res image |

4035 rows have `in_tissue == 1`, which matches the 4035
barcodes in the filtered matrix exactly (aligned here by barcode string, not by row order).

### `V1_Human_Lymph_Node_image.tif`
Uncompressed (`raw`) single-page RGB TIFF, **11291 x 11757 px (W x H)**.

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
x_hires  = x_fullres * tissue_hires_scalef   # 0.17011142
y_hires  = y_fullres * tissue_hires_scalef
x_lowres = x_fullres * tissue_lowres_scalef  # 0.051033426
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

**Spot diameter:** `spot_diameter_fullres = 89.48996643177776` px
(full-resolution pixels, from `scalefactors_json.json`). A Visium spot is 55 um across,
so the implied resolution is 55 / 89.4900 =
**0.614594 um/px**, and the 96x96 px patch covers
96 * 0.614594 = **59.00 um** per side —
i.e. the patch is slightly wider than one spot (1.073 spot diameters).
Spot centre-to-centre pitch is 100 um.

## 4. Counts before and after filtering

| Quantity | Value |
|---|---|
| Spots on slide (`tissue_positions_list.csv` rows) | 4992 |
| Spots with `in_tissue == 1` | 4035 |
| Spots in filtered matrix (raw sample) | 4035 |
| Spots dropped (96x96 crop would leave the image) | 0 |
| **Spots kept in `hest_sample.npz`** | **4035** |
| Genes in raw matrix (before filtering) | 36601 |
| **Genes kept** | **256** |
| Full-res image size (W x H) | 11291 x 11757 |
| `spot_diameter_fullres` | 89.48996643177776 px |

### How "most variable" was computed
1. Densify the CSC matrix to `spots x genes` `float32` **raw UMI counts**.
2. Library-size normalise each spot to 10,000 counts: `counts / counts.sum(axis=1, keepdims=True) * 1e4`
   (spots with zero total counts are left divided by 1 — none occur here).
3. `log1p` transform.
4. Per-gene **sample variance across spots** (`norm.var(axis=0, ddof=1)`).
5. Take the 256 genes with the largest variance, then re-sort those indices into
   original file (genomic) order so the output is deterministic.

Kept-gene variance range: **0.171551** (lowest kept) to
**0.617955** (highest).

## 5. `data/hest_sample.npz` contents

Written with `numpy.savez_compressed` (65,559,626 bytes).

| Array | dtype | Shape | Meaning |
|---|---|---|---|
| `patches` | `uint8` | `(4035, 96, 96, 3)` | RGB H&E crop centred on each spot, full-res scale |
| `expr` | `float32` | `(4035, 256)` | log1p(CP10K) expression, spot-major, column j = `gene_names[j]` |
| `gene_names` | `<U...` (unicode) | `(256,)` | HGNC symbols from `matrix/features/name` |
| `coords` | `float32` | `(4035, 2)` | `(x, y)` spot centre in full-res image pixels |
| `sample_id` | `<U19` (unicode) | `()` scalar | `"V1_Human_Lymph_Node"` |

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
