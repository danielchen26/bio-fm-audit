"""Innovation: the stain-augmentation invariance/utility frontier.

The JD asks for stain QC standards and models that transfer. The usual practice
is to pick an augmentation strength by feel. This measures the actual trade-off.

Sweep HED stain-jitter strength during SSL pretraining. For each strength, on the
SAME frozen embeddings, measure two quantities that pull in opposite directions:

  utility    benign/malignant AUC on PATIENT-DISJOINT test patients
  confound   how well patient identity is linearly decodable from the embedding

Strong stain jitter should destroy slide-specific appearance (confound down). The
question nobody reports is what it costs in biology (utility down?). The useful
output is the frontier, not a single number: it tells you the strongest
augmentation you can afford before the biological signal starts to go.

This is the pathology analogue of the protein-side token information probe: never
accept a representation on the strength of its pretraining loss.
"""
from __future__ import annotations

import argparse
from argparse import Namespace

import numpy as np
import torch

import dino
from audit import audit_embedding
from bench import embed_torch
from backbone import ViT
from pathdata import DATA, ROOT, load_manifest, make_split


def train_one(sigma: float, base: Namespace) -> str:
    out = f"dino_hed{int(sigma * 100):03d}.pt"
    if (ROOT / "ckpt" / out).exists():
        print(f"[frontier] reuse ckpt/{out}")
        return out
    args = Namespace(**{**vars(base), "hed": sigma, "out": out})
    print(f"\n########## SSL pretraining with hed_sigma = {sigma} ##########")
    dino.run(args)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.0,0.05,0.10,0.20,0.35")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--cache", default="patches96.npz")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--probe-device", default="cpu")
    a = ap.parse_args()

    base = Namespace(steps=a.steps, batch=a.batch, lr=5e-4, img=96, patch=8, d=192,
                     depth=6, heads=3, protos=256, n_local=4, s_temp=0.1,
                     momentum=0.996, cache=a.cache, device=a.device, hed=0.10,
                     out="dino.pt", ckpt_every=400)

    df = load_manifest()
    patches = np.load(DATA / a.cache)["patches"]
    # multiple split seeds: with 14 test patients a single split resolves only
    # ~0.03 AUC, so a one-seed frontier cannot separate signal from noise
    seeds = [0, 1, 2, 3, 4]
    splits = [make_split(df, "patient", seed=s) for s in seeds]
    rows = []

    for sigma in [float(s) for s in a.sigmas.split(",")]:
        ck_name = train_one(sigma, base)
        ck = torch.load(ROOT / "ckpt" / ck_name, map_location="cpu")
        m = ViT(**ck["cfg"])
        m.load_state_dict(ck["backbone"])
        emb = embed_torch(m, patches, ck["cfg"]["img"], a.device, normalize="half")
        per = [audit_embedding(emb, df, sp, device=a.probe_device) for sp in splits]
        agg = lambda k, sub: (float(np.mean([r[sub][k] for r in per])),
                              float(np.std([r[sub][k] for r in per])))
        auc_m, auc_s = agg("auc", "biology")
        pid_m, pid_s = agg("acc", "patient")
        mag_m, _ = agg("acc", "magnif")
        lift_m, _ = agg("lift", "patient")
        rows.append({"sigma": sigma, "bio_auc": auc_m, "bio_auc_sd": auc_s,
                     "pid_acc": pid_m, "pid_acc_sd": pid_s,
                     "pid_lift": lift_m, "mag_acc": mag_m, "n_seeds": len(splits)})
        r = rows[-1]
        print(f"\n[frontier] sigma {sigma:.2f}  bio-AUC {auc_m:.4f}+-{auc_s:.4f}  "
              f"patientID {pid_m:.4f}+-{pid_s:.4f} (lift {lift_m:.1f}x)  magnif {mag_m:.4f}")

    print("\n" + "=" * 84)
    print("STAIN-AUGMENTATION INVARIANCE / UTILITY FRONTIER (patient-disjoint split)")
    print("=" * 84)
    print(f"{'hed_sigma':>10} {'bio-AUC (5 seeds)':>22} {'patientID-acc':>22} "
          f"{'pid-lift':>9} {'magnif':>8}")
    for r in rows:
        print(f"{r['sigma']:>10.2f} {r['bio_auc']:>13.4f} +-{r['bio_auc_sd']:<7.4f} "
              f"{r['pid_acc']:>13.4f} +-{r['pid_acc_sd']:<7.4f} "
              f"{r['pid_lift']:>8.1f}x {r['mag_acc']:>8.4f}")
    sd = float(np.mean([r["bio_auc_sd"] for r in rows]))
    spread = max(r["bio_auc"] for r in rows) - min(r["bio_auc"] for r in rows)
    print(f"\n[power] mean across-seed SD {sd:.4f} vs across-sigma spread {spread:.4f} "
          f"-> {'RESOLVABLE' if spread > 2 * sd else 'NOT RESOLVABLE at this n'}")

    best = max(rows, key=lambda r: r["bio_auc"])
    ref = rows[0]
    print(f"\n[frontier] no-jitter reference: bio-AUC {ref['bio_auc']:.4f}, "
          f"patientID {ref['pid_acc']:.4f}")
    print(f"[frontier] best utility at sigma {best['sigma']:.2f}: "
          f"bio-AUC {best['bio_auc']:.4f} ({best['bio_auc'] - ref['bio_auc']:+.4f} vs no jitter), "
          f"patientID {best['pid_acc']:.4f} ({best['pid_acc'] - ref['pid_acc']:+.4f})")
    # the operating point worth reporting: strongest jitter that keeps utility
    tol = 0.01
    ok = [r for r in rows if r["bio_auc"] >= best["bio_auc"] - tol]
    pick = max(ok, key=lambda r: r["sigma"])
    print(f"[frontier] strongest jitter within {tol} AUC of best: sigma {pick['sigma']:.2f} "
          f"-> bio-AUC {pick['bio_auc']:.4f}, patientID {pick['pid_acc']:.4f} "
          f"({(pick['pid_acc'] - ref['pid_acc']) / max(ref['pid_acc'], 1e-9) * 100:+.1f}% confound)")
    np.savez_compressed(DATA / "frontier.npz", rows=np.array([rows], dtype=object))
    print(f"\n[frontier] wrote {DATA / 'frontier.npz'}")


if __name__ == "__main__":
    main()
