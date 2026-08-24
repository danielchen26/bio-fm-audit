"""Re-crop the paired sample at a LARGER field of view.

Why: the null result for Phikon may be an artefact of my own preprocessing, not a
property of the model. Our patches were 96 px at 0.6146 um/px = 59 um, then
upsampled 96 -> 224 to feed a ViT/16. Phikon was pretrained on tiles covering
roughly 128-256 um at native ~20x. So it was being asked to read an image at a
scale and blur level it never saw.

That confound is testable: crop 224 px NATIVELY (138 um, no upsampling) around the
same spot centres and re-run. If Phikon still matches random features, the null
result is about the model. If it improves, the null result was mine.

Everything else (spot centres, expression, gene set) is reused verbatim from the
verified 96 px sample, so the only variable is field of view.
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image

from pathdata import DATA

Image.MAX_IMAGE_PIXELS = None
RAW = DATA / "raw_10x"
TIF = RAW / "V1_Human_Lymph_Node_image.tif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--src", default="hest_sample.npz")
    a = ap.parse_args()

    z = np.load(DATA / a.src, allow_pickle=False)
    coords, expr, genes = z["coords"], z["expr"], z["gene_names"]
    n, half = len(coords), a.size // 2

    print(f"[recrop] opening {TIF.name} ({TIF.stat().st_size / 1e6:.0f} MB)")
    img = np.asarray(Image.open(TIF).convert("RGB"))
    H, W, _ = img.shape
    print(f"[recrop] image {W}x{H}, {n} spots, target patch {a.size}px "
          f"= {a.size * 0.6146:.0f} um field of view")

    keep, patches = [], []
    for i, (x, y) in enumerate(coords):
        x0, y0 = int(round(float(x))) - half, int(round(float(y))) - half
        if x0 < 0 or y0 < 0 or x0 + a.size > W or y0 + a.size > H:
            continue
        patches.append(img[y0:y0 + a.size, x0:x0 + a.size])
        keep.append(i)
    keep = np.array(keep)
    patches = np.stack(patches).astype(np.uint8)
    print(f"[recrop] kept {len(keep)}/{n} spots ({n - len(keep)} fell outside the image)")

    out = DATA / f"hest_sample{a.size}.npz"
    np.savez_compressed(out, patches=patches, expr=expr[keep], coords=coords[keep],
                        gene_names=genes, sample_id=z["sample_id"])
    print(f"[recrop] wrote {out.name}  patches {patches.shape} {patches.nbytes / 1e6:.0f} MB raw")
    print(f"[recrop] mean intensity {patches.mean():.1f}, "
          f"blank-ish patches (>235) {int((patches.reshape(len(patches), -1).mean(1) > 235).sum())}")


if __name__ == "__main__":
    main()
