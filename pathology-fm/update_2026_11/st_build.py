"""Build full-slide paired datasets (H&E patch, expression) for three public 10x Visium slides with ONE procedure.
Patch = the same physical field on every slide (the field of the 224-px crop on the V1 lymph node), scale from the
nominal 100 um array pitch.  Genes = top 256 by variance of log1p(CP10K).  Raw counts are kept for the noise ceiling."""
import sys, os, json, tarfile
sys.dont_write_bytecode = True
import numpy as np, h5py
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
D = os.environ.get("PFM_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))
T, S = f"{D}/raw_10x", f"{D}/raw_10x"                      # every 10x download in one folder
SLIDES = {"V1LN": (f"{D}/raw_10x/V1_Human_Lymph_Node_filtered_feature_bc_matrix.h5", f"{D}/raw_10x/V1_Human_Lymph_Node_image.tif",
                   f"{D}/raw_10x/V1_Human_Lymph_Node_spatial.tar.gz"),
          "BRCA": (f"{T}/V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5", f"{T}/V1_Breast_Cancer_Block_A_Section_1_image.tif",
                   f"{T}/V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz"),
          "V2LN": (f"{T}/Visium_V2_Human_Lymph_Node_filtered_feature_bc_matrix.h5", f"{S}/Visium_V2_Human_Lymph_Node_tissue_image.tif",
                   f"{S}/Visium_V2_Human_Lymph_Node_spatial.tar.gz")}
P, NG = 224, 256


def read_h5(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]; ng, nb = (int(v) for v in m["shape"][:])
        data, idx, ptr = m["data"][:], m["indices"][:], m["indptr"][:]
        bcs = [b.decode() for b in m["barcodes"][:]]; genes = np.array([g.decode() for g in m["features/name"][:]])
    c = np.zeros((nb, ng), np.int32)
    for j in range(nb):
        c[j, idx[ptr[j]:ptr[j + 1]]] = data[ptr[j]:ptr[j + 1]]
    return c, bcs, genes


def read_positions(tar_path):
    with tarfile.open(tar_path) as t:
        m = [x for x in t.getmembers() if os.path.basename(x.name) in ("tissue_positions_list.csv", "tissue_positions.csv") and x.isfile()][0]
        lines = t.extractfile(m).read().decode().strip().splitlines()
    if lines[0].lower().startswith("barcode"):
        lines = lines[1:]
    return {l.split(",")[0]: (float(l.split(",")[5]), float(l.split(",")[4])) for l in lines}      # barcode -> (x, y) full-res px

meta, field_um = {}, None
for sid, (h5, tif, tar) in SLIDES.items():
    out = f"st_slide_{sid}.npz"
    counts, bcs, genes = read_h5(h5); pos = read_positions(tar)
    xy = np.array([pos[b] for b in bcs], np.float64)
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1); np.fill_diagonal(d2, np.inf)
    pitch_px = float(np.median(np.sqrt(d2.min(1)))); mpp = 100.0 / pitch_px; del d2
    if field_um is None:
        field_um = P * mpp                                   # V1 lymph node defines the physical field
    s = int(round(field_um / mpp))
    img = Image.open(tif); W, H = img.size; mode = img.mode
    arr = np.asarray(img.convert("RGB")); del img
    x0 = np.rint(xy[:, 0]).astype(np.int64) - s // 2 if s == P else np.rint(xy[:, 0] - s / 2).astype(np.int64)
    y0 = np.rint(xy[:, 1]).astype(np.int64) - s // 2 if s == P else np.rint(xy[:, 1] - s / 2).astype(np.int64)
    keep = (x0 >= 0) & (y0 >= 0) & (x0 + s <= W) & (y0 + s <= H)
    patches = np.empty((int(keep.sum()), P, P, 3), np.uint8)
    for i, j in enumerate(np.where(keep)[0]):
        crop = arr[y0[j]:y0[j] + s, x0[j]:x0[j] + s]
        patches[i] = crop if s == P else np.asarray(Image.fromarray(crop).resize((P, P), Image.Resampling.BILINEAR))
    del arr
    counts, xy, bcs = counts[keep], xy[keep], [b for b, k in zip(bcs, keep) if k]
    lib = counts.sum(1, keepdims=True).astype(np.float64); lib[lib == 0] = 1.0
    norm = np.log1p(counts / lib * 1e4).astype(np.float32)
    top = np.sort(np.argsort(norm.var(0, ddof=1))[::-1][:NG])
    expr = np.ascontiguousarray(norm[:, top]); c256 = counts[:, top]
    meta[sid] = {"spots": int(len(xy)), "dropped": int((~keep).sum()), "genes_total": int(len(genes)), "image": [W, H], "image_mode": mode,
                 "pitch_px": round(pitch_px, 3), "um_per_px": round(mpp, 4), "crop_px": s, "field_um": round(s * mpp, 1),
                 "median_umi_per_spot": float(np.median(lib)), "blankish_patches": int((patches.reshape(len(patches), -1).mean(1) > 235).sum())}
    if sid == "V1LN" and os.path.exists(f"{D}/hest_sample224.npz"):                                         # gate: when src/get_hest.py + src/recrop.py output is present it must be reproduced exactly
        with np.load(f"{D}/hest_sample224.npz", allow_pickle=False) as z:
            assert (z["gene_names"].astype(str) == genes[top]).all() and np.allclose(z["expr"], expr, atol=1e-5)
            assert np.allclose(z["coords"], xy.astype(np.float32)) and (z["patches"] == patches).all()
        meta[sid]["gate"] = "patches, coordinates, genes and expression identical to the project's hest_sample224.npz"
    if sid == "V2LN" and os.path.exists(f"{D}/v2ln_reference_patches.npz"):                                         # gate: optional cross-check against independently prepared crops of the same slide
        with np.load(f"{D}/v2ln_reference_patches.npz", allow_pickle=False) as z:
            ref, ids = z["patches"], z["spot_ids"].astype(str)
        ix = {b: i for i, b in enumerate(bcs)}; hit = [(i, ix[b]) for i, b in enumerate(ids) if b in ix]
        a = np.stack([ref[i] for i, _ in hit]).astype(np.float32); b = np.stack([patches[j] for _, j in hit]).astype(np.float32)
        cc = [float(np.corrcoef(a[k].ravel(), b[k].ravel())[0, 1]) for k in range(len(hit))]
        meta[sid]["gate"] = {"matched_spots": len(hit), "median_pixel_corr_vs_project_patches": round(float(np.median(cc)), 4),
                             "min_pixel_corr": round(float(np.min(cc)), 4), "mean_abs_diff": round(float(np.abs(a - b).mean()), 2)}
    np.savez_compressed(out, patches=patches, coords=xy.astype(np.float32), expr=expr, gene_names=genes[top].astype(str), counts=c256,
                        rest=(counts.sum(1) - c256.sum(1)).astype(np.int64), barcodes=np.array(bcs), um_per_px=np.float64(mpp))
    print(sid, meta[sid], flush=True)
    json.dump(meta, open("st_build.json", "w"), indent=1)
print("done", flush=True)
