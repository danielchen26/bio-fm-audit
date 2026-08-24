"""The CAMELYON17 benchmark: what does site shift actually cost?

The quantity this exists to measure:

    delta_site = AUC(patient-disjoint) - AUC(leave-one-centre-out)

Patient-disjoint is the honest split within a cohort -- the number the BreaKHis
study produced. Leave-one-centre-out trains on four hospitals and tests on a
fifth, so the difference isolates site/scanner/stain shift from ordinary
biological variation. A single-cohort dataset cannot produce this number at all,
which is why the earlier stain-augmentation sweep came out underpowered.

Leave-one-centre-out is run for ALL FIVE folds and reported as mean +- std, since
one fold is one hospital and hospitals differ a lot.

Confound probes on the same frozen embeddings:
    centre  -- 5-class. If a patch's hospital is linearly decodable, the encoder
               has learned scanner and staining, which is exactly what fails to
               transfer.
    slide   -- slide identity, the CAMELYON analogue of the patient-ID probe.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from audit import logistic_probe, roc_auc
from c17data import GRADES, audit_split, load, make_split

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def raw_pixel_features(patches: np.ndarray, bins: int = 16, chunk: int = 2048) -> np.ndarray:
    """Colour statistics only. The trivial baseline that must be in the table."""
    out = []
    for i in range(0, len(patches), chunk):
        x = patches[i:i + chunk].astype(np.float32) / 255.0
        f = [x.mean((1, 2)), x.std((1, 2))]
        for c in range(3):
            h = np.stack([np.histogram(im[..., c], bins=bins, range=(0, 1))[0] for im in x])
            f.append(h / np.maximum(h.sum(1, keepdims=True), 1))
        out.append(np.concatenate(f, 1).astype(np.float32))
    return np.concatenate(out)


@torch.no_grad()
def embed(model, patches, size, device, batch=64, norm="imagenet"):
    dev = torch.device(device)
    model = model.to(dev).eval()
    mean, std = IMAGENET_MEAN.to(dev), IMAGENET_STD.to(dev)
    out = []
    for i in range(0, len(patches), batch):
        x = torch.from_numpy(patches[i:i + batch]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        if x.shape[-1] != size:
            x = F.interpolate(x, (size, size), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - mean) / std if norm == "imagenet" else (x - 0.5) / 0.25
        z = model(x)
        out.append((z if torch.is_tensor(z) else z.last_hidden_state[:, 0]).float().cpu().numpy())
        if (i // batch) % 60 == 0:
            print(f"    embed {i}/{len(patches)}", flush=True)
    return np.concatenate(out)


def encoders(which, patches, device, dino_ckpt):
    if "raw" in which:
        print("[enc] raw-pixel colour statistics")
        yield "raw-pixel", raw_pixel_features(patches)
    if "imagenet" in which:
        import timm
        print("[enc] ImageNet ViT-S/16")
        m = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k",
                              pretrained=True, num_classes=0)
        yield "imagenet-vit-s", embed(m, patches, 224, device)
    if "phikon" in which:
        from transformers import AutoModel
        print("[enc] Phikon ViT-B (iBOT on TCGA)")
        m = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        yield "phikon-vit-b", embed(m, patches, 224, device)
    if "ours" in which:
        from pathdata import ROOT
        from backbone import ViT
        ck = torch.load(ROOT / "ckpt" / dino_ckpt, map_location="cpu")
        print(f"[enc] ours-dino {ck['cfg']}")
        m = ViT(**ck["cfg"])
        m.load_state_dict(ck["backbone"])
        yield "ours-dino", embed(m, patches, ck["cfg"]["img"], device, norm="half")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="raw,imagenet,phikon")
    ap.add_argument("--dino", default="dino_p256.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--probe-device", default="cpu")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--grades", default="",
                    help="restrict to these slide grades, e.g. micro,macro")
    ap.add_argument("--max-tiles-per-slide", type=int, default=400)
    a = ap.parse_args()

    d = load(max_tiles_per_slide=a.max_tiles_per_slide)
    if a.grades:
        import dataclasses
        keep_g = {GRADES.index(g) for g in a.grades.split(",")}
        m = np.isin(d.grade, list(keep_g))
        d = dataclasses.replace(
            d, patches=d.patches[m], y=d.y[m], centre=d.centre[m], slide=d.slide[m],
            patient=d.patient[m], grade=d.grade[m], cover=d.cover[m])
        print(f"[filter] grades {a.grades} -> {len(d)} tiles "
              f"({d.y.mean():.1%} tumour)\n")
    print(d.summary(), "\n")
    for r in ("random", "slide", "patient"):
        sp = make_split(d, r)
        s = audit_split(d, sp)
        print(f"[{r:8s}] {sp.sizes()}  shared slides {s['shared_slides']} "
              f"patients {s['shared_patients']} centres {s['shared_centres']}")
    centres = sorted(set(d.centre.tolist()))
    print(f"[centre  ] leave-one-out over {centres}\n")

    results = {}
    for name, emb in encoders(a.encoders.split(","), d.patches, a.device, a.dino):
        print(f"\n=== {name} (dim {emb.shape[1]}) ===")
        row = {}
        for r in ("random", "slide", "patient"):
            aucs = []
            for seed in range(a.seeds):
                sp = make_split(d, r, seed=seed)
                if len(set(d.y[sp.test].tolist())) < 2:
                    continue
                aucs.append(logistic_probe(emb[sp.train], d.y[sp.train], emb[sp.test],
                                           d.y[sp.test], 2, device=a.probe_device)["auc"])
            row[r], row[r + "_std"] = float(np.mean(aucs)), float(np.std(aucs))
            print(f"  tumour AUC [{r:8s}] {row[r]:.4f} +-{row[r + '_std']:.4f} "
                  f"({len(aucs)} seeds)")

        folds = []
        for hc in centres:
            sp = make_split(d, "centre", held_centre=hc)
            if len(set(d.y[sp.train].tolist())) < 2 or len(set(d.y[sp.test].tolist())) < 2:
                continue
            p = logistic_probe(emb[sp.train], d.y[sp.train], emb[sp.test], d.y[sp.test],
                               2, device=a.probe_device)
            folds.append(p["auc"])
            print(f"  tumour AUC [centre {hc}] {p['auc']:.4f}  "
                  f"(test n={len(sp.test)}, pos {d.y[sp.test].mean():.2f})")
        row["centre_mean"] = float(np.mean(folds))
        row["centre_std"] = float(np.std(folds))
        row["n_folds"] = len(folds)

        # confounds, probed within the patient-disjoint TRAIN partition
        sp = make_split(d, "patient")
        tr = sp.train
        rs = np.random.default_rng(0).permutation(len(tr))
        cut = int(0.7 * len(rs))
        A, B = tr[rs[:cut]], tr[rs[cut:]]
        cen = logistic_probe(emb[A], d.centre[A], emb[B], d.centre[B],
                             len(centres), device=a.probe_device)
        sl_codes = {s: i for i, s in enumerate(sorted(set(d.slide[tr].tolist())))}
        # A and B are both subsets of tr, so every slide here IS in sl_codes
        slA = np.array([sl_codes[s] for s in d.slide[A]])
        slB = np.array([sl_codes[s] for s in d.slide[B]])
        sld = logistic_probe(emb[A], slA, emb[B], slB, len(sl_codes),
                            device=a.probe_device)
        row["centre_probe"] = cen["acc"]
        row["centre_chance"] = cen["chance"]
        row["slide_probe"] = sld["acc"]
        row["slide_lift"] = sld["lift"]
        print(f"  confound: centre-ID {cen['acc']:.4f} (chance {cen['chance']:.3f}, "
              f"lift {cen['lift']:.2f}x) | slide-ID {sld['acc']:.4f} "
              f"(chance {sld['chance']:.4f}, lift {sld['lift']:.1f}x)")
        results[name] = row

    print("\n" + "=" * 104)
    print("CAMELYON17 tumour-tile AUC by split regime.  delta_site = patient - centre")
    print("=" * 104)
    print(f"{'encoder':<16} {'random':>8} {'slide':>8} {'patient':>8} "
          f"{'centre (LOCO)':>18} {'delta_site':>11} {'centreID':>10} {'slideID':>9}")
    for n, r in results.items():
        print(f"{n:<16} {r['random']:>8.4f} {r['slide']:>8.4f} {r['patient']:>8.4f} "
              f"{r['centre_mean']:>10.4f}+-{r['centre_std']:<6.4f} "
              f"{r['patient'] - r['centre_mean']:>+11.4f} "
              f"{r['centre_probe']:>10.4f} {r['slide_lift']:>8.1f}x")
    np.savez_compressed("/Volumes/Lexar/camelyon17/c17_bench.npz",
                        results=np.array([results], dtype=object))
    print("\n[bench] wrote /Volumes/Lexar/camelyon17/c17_bench.npz")


if __name__ == "__main__":
    main()
