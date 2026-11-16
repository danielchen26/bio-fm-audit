"""The benchmark table. Every encoder, every split regime, every baseline.

Encoders compared:
  raw-pixel   -- channel means/stds + per-channel colour histogram. The trivial
                 baseline that MUST be in the table. If a foundation model cannot
                 beat a colour histogram, it is not doing anything.
  imagenet    -- ViT-S/16 pretrained on ImageNet. The "did we need pathology
                 data at all" control.
  phikon      -- Owkin Phikon, ViT-B/16 iBOT-pretrained on TCGA. A real published
                 pathology foundation model.
  ours-dino   -- our DINO encoder, pretrained on TRAIN PATIENTS ONLY.

Reported per encoder: biology AUC under each split regime, the leakage delta
between the leaky and honest split, the confound decodability, and patient-level
MIL AUC with error bars.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from audit import audit_embedding, format_audit, leak_report
from mil import format_mil, train_mil
from pathdata import DATA, ROOT, load_manifest, make_split

DINO_CKPT = ["dino.pt"]
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def checkpoint_norm(model):
    """Input mean/std a timm checkpoint was trained with.

    Not every ImageNet model uses the ImageNet statistics: the augreg ViT-S used
    here expects 0.5/0.5. Until 2026-11 it was fed the ImageNet statistics, which
    understated the ImageNet baseline in every table that has one.
    """
    import timm
    cfg = timm.data.resolve_model_data_config(model)
    return tuple(cfg["mean"]), tuple(cfg["std"])


def _mean_std(normalize, dev):
    """'imagenet', 'half' (0.5 / 0.25, our DINO), or the (mean, std) of the checkpoint itself."""
    if normalize == "imagenet":
        return IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    if normalize == "half":
        return 0.5, 0.25
    mean, std = normalize
    return (torch.tensor(mean).view(1, 3, 1, 1).to(dev), torch.tensor(std).view(1, 3, 1, 1).to(dev))


def raw_pixel_features(patches: np.ndarray, bins: int = 16) -> np.ndarray:
    """Colour statistics only. No learning of any kind."""
    x = patches.astype(np.float32) / 255.0
    feats = [x.mean((1, 2)), x.std((1, 2))]
    for c in range(3):
        h = np.stack([np.histogram(im[..., c], bins=bins, range=(0, 1))[0] for im in x])
        feats.append(h / h.sum(1, keepdims=True))
    return np.concatenate(feats, axis=1).astype(np.float32)


@torch.no_grad()
def embed_torch(model, patches, size, device, batch=64, normalize="imagenet", cls_only=True):
    dev = torch.device(device)
    model = model.to(dev).eval()
    out = []
    mean, std = _mean_std(normalize, dev)
    for i in range(0, len(patches), batch):
        u8 = torch.from_numpy(patches[i:i + batch]).to(dev)
        x = u8.permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - mean) / std
        z = model(x)
        if not torch.is_tensor(z):
            z = z.last_hidden_state[:, 0] if cls_only else z.pooler_output
        out.append(z.float().cpu().numpy())
        if (i // batch) % 25 == 0:
            print(f"    embed {i}/{len(patches)}", flush=True)
    return np.concatenate(out)


def get_encoders(which, patches, device):
    """Yield (name, embedding) pairs, computing each lazily."""
    if "raw" in which:
        print("[bench] raw-pixel features")
        yield "raw-pixel", raw_pixel_features(patches)

    if "imagenet" in which:
        import timm
        print("[bench] ImageNet ViT-S/16")
        m = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k",
                              pretrained=True, num_classes=0)
        yield "imagenet-vit-s", embed_torch(m, patches, 224, device, normalize=checkpoint_norm(m))

    if "phikon" in which:
        from transformers import AutoModel
        print("[bench] Phikon (owkin/phikon, ViT-B iBOT on TCGA)")
        m = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        yield "phikon-vit-b", embed_torch(m, patches, 224, device)

    if "ours" in which:
        from backbone import ViT
        ck = torch.load(ROOT / "ckpt" / DINO_CKPT[0], map_location="cpu")
        cfg = ck["cfg"]
        print(f"[bench] ours-dino {cfg}")
        m = ViT(**cfg)
        m.load_state_dict(ck["backbone"])
        yield "ours-dino", embed_torch(m, patches, cfg["img"], device, normalize="half")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="patches96.npz")
    ap.add_argument("--encoders", default="raw,imagenet,phikon,ours")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--probe-device", default="cpu")
    ap.add_argument("--skip-mil", action="store_true")
    ap.add_argument("--dino", default="dino.pt")
    a = ap.parse_args()
    DINO_CKPT[0] = a.dino

    df = load_manifest()
    patches = np.load(DATA / a.cache)["patches"]
    assert len(patches) == len(df), (len(patches), len(df))
    print(f"[bench] {len(df)} patches, {df['patient_id'].nunique()} patients, "
          f"malignant fraction {df['y'].mean():.3f}\n")

    splits = {r: make_split(df, r) for r in ("random", "patient", "magnification")}
    for r, sp in splits.items():
        print(f"[split:{r}] train/val/test = {sp.sizes()}")
    print()

    results = {}
    for name, emb in get_encoders(a.encoders.split(","), patches, a.device):
        print(f"\n=== {name}  (dim {emb.shape[1]}) ===")
        res = {}
        for r, sp in splits.items():
            res[r] = audit_embedding(emb, df, sp, device=a.probe_device)
            print(f"  [{r}]")
            print(format_audit(name, res[r]))
        print("  [leakage]")
        print(leak_report(res["random"], res["patient"]))
        if not a.skip_mil:
            print("  [patient-level MIL, honest split]")
            m = train_mil(df, emb, "patient", device=a.probe_device)
            print(format_mil(name, m))
            res["mil"] = m
        results[name] = res

    print("\n" + "=" * 100)
    print("SUMMARY  biology AUC by split regime (higher is better; the gap is the story)")
    print("=" * 100)
    print(f"{'encoder':<18} {'random(leaky)':>14} {'patient(honest)':>16} "
          f"{'magnif(shift)':>14} {'leak':>8} {'patientID-lift':>15} {'MIL-AUC':>16}")
    for name, res in results.items():
        mil = f"{res['mil']['auc_mean']:.3f}+-{res['mil']['auc_std']:.3f}" if "mil" in res else "-"
        print(f"{name:<18} {res['random']['biology']['auc']:>14.3f} "
              f"{res['patient']['biology']['auc']:>16.3f} "
              f"{res['magnification']['biology']['auc']:>14.3f} "
              f"{res['random']['biology']['auc'] - res['patient']['biology']['auc']:>+8.3f} "
              f"{res['patient']['patient']['lift']:>14.1f}x {mil:>16}")

    np.savez_compressed(DATA / "bench_results.npz", results=np.array([results], dtype=object))
    print(f"\n[bench] wrote {DATA / 'bench_results.npz'}")


if __name__ == "__main__":
    main()
