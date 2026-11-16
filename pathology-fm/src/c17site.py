"""Paired leave-one-centre-out: measure site shift with everything else held equal.

Two earlier attempts at delta_site failed, both for the same reason: I compared
two DIFFERENT split schemes. Leave-one-centre-out tests on a whole hospital
(n = 1300-3000, prevalence 0.19-0.70); a patient-disjoint split tests on 25% of
patients scattered across all hospitals. Test-set size, tumour prevalence and
lesion-grade composition all differ, and those differences are larger than the
site effect. The result came out NEGATIVE -- "unseen hospitals are easier" --
which is not a finding, it is a broken design.

The fix is a paired design. For each held-out centre c:

  train        tiles from the other 4 centres, minus a held-out patient group
  test_unseen  all tiles from centre c                    <- site never seen
  test_seen    tiles from the held-out patients of the other 4 centres,
               STRATIFIED-MATCHED to test_unseen on the joint
               (grade x label) distribution and on total size

ONE model is scored on BOTH test sets. They differ in exactly one property:
whether the test slides' hospital appeared in training. Everything the previous
attempts confounded -- size, prevalence, grade mix -- is matched by construction,
and the achieved match is printed so it can be checked rather than trusted.

    delta_site_paired = AUC(test_seen) - AUC(test_unseen)

Positive means unseen hospitals genuinely cost accuracy.
"""
from __future__ import annotations

import os
import argparse
from collections import Counter

from pathlib import Path

import numpy as np
import torch

from audit import logistic_probe
from c17bench import checkpoint_norm, embed, raw_pixel_features
from c17data import GRADES, load

# CAMELYON17 slides, tiles and caches (about 180 GB). Set C17_ROOT to keep them on another disk.
CACHE = str(Path(os.environ.get("C17_ROOT", Path(__file__).resolve().parent.parent / "data" / "camelyon17")))


def get_embedding(d, kind: str, device: str) -> np.ndarray:
    import os
    # "_cfgnorm": ImageNet features made with the checkpoint's own input statistics (older caches used the wrong ones)
    path = f"{CACHE}/emb_{kind}{'_cfgnorm' if kind == 'imagenet' else ''}_{len(d)}.npz"
    if os.path.exists(path):
        e = np.load(path)["emb"]
        print(f"[emb] cached {kind} {e.shape}")
        return e
    if kind == "raw":
        e = raw_pixel_features(d.patches)
    elif kind == "imagenet":
        import timm
        m = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k",
                              pretrained=True, num_classes=0)
        e = embed(m, d.patches, 224, device, norm=checkpoint_norm(m))
    elif kind == "phikon":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        e = embed(m, d.patches, 224, device)
    else:
        raise ValueError(kind)
    np.savez_compressed(path, emb=e)
    print(f"[emb] computed {kind} {e.shape} -> {path}")
    return e


def stratified_subsample(d, idx, n, rng):
    """Take n tiles from idx preserving its own (grade,label) composition."""
    if len(idx) <= n:
        return idx
    out = []
    keys = Counter(zip(d.grade[idx].tolist(), d.y[idx].tolist()))
    for (g, y), cnt in keys.items():
        m = idx[(d.grade[idx] == g) & (d.y[idx] == y)]
        want = int(round(n * cnt / len(idx)))
        if want and len(m):
            out.append(m[rng.choice(len(m), min(want, len(m)), replace=False)])
    return np.concatenate(out) if out else idx[:0]


def strata(d, idx):
    """Joint (grade, label) counts -- the thing that must match."""
    return Counter(zip(d.grade[idx].tolist(), d.y[idx].tolist()))


def matched_sample(d, pool_idx, target: Counter, rng) -> np.ndarray:
    """Draw from pool_idx to reproduce `target`'s (grade,label) counts as closely
    as the pool allows. Scale all strata by the tightest feasible ratio so the
    RELATIVE composition is preserved even when absolute counts fall short."""
    by = {}
    for k in target:
        g, y = k
        m = np.where((d.grade[pool_idx] == g) & (d.y[pool_idx] == y))[0]
        by[k] = pool_idx[m]
    ratio = min((len(by[k]) / target[k]) for k in target if target[k] > 0)
    ratio = min(ratio, 1.0)
    out = []
    for k, n in target.items():
        want = int(round(n * ratio))
        avail = by[k]
        if want and len(avail):
            out.append(avail[rng.choice(len(avail), min(want, len(avail)), replace=False)])
    return np.concatenate(out) if out else np.array([], int), ratio


def describe(d, idx) -> str:
    if not len(idx):
        return "empty"
    g = Counter(d.grade[idx].tolist())
    frac = " ".join(f"{GRADES[k]}:{v / len(idx):.2f}" for k, v in sorted(g.items()))
    return f"n={len(idx):5d} pos={d.y[idx].mean():.3f} [{frac}]"


def run(a):
    d = load(max_tiles_per_slide=a.max_tiles_per_slide)
    if a.grades:
        import dataclasses
        keep = {GRADES.index(g) for g in a.grades.split(",")}
        m = np.isin(d.grade, list(keep))
        d = dataclasses.replace(d, patches=d.patches[m], y=d.y[m], centre=d.centre[m],
                                slide=d.slide[m], patient=d.patient[m],
                                grade=d.grade[m], cover=d.cover[m])
        print(f"[filter] grades {a.grades} -> {len(d)} tiles ({d.y.mean():.1%} tumour)")
    print(d.summary(), "\n")

    centres = sorted(set(d.centre.tolist()))
    summary = {}
    for kind in a.encoders.split(","):
        emb = get_embedding(d, kind, a.device)
        print(f"\n=== {kind} (dim {emb.shape[1]}) : paired leave-one-centre-out ===")
        deltas, seens, unseens = [], [], []
        for c in centres:
            rng0 = np.random.default_rng(1000 * c)
            # Fixed test size for BOTH sets: centre tile counts range 3.8k-11.5k, and
            # matching the biggest forced holding out most patients, which starved
            # training to 3.7k tiles on one fold and 21k on another. Equal, modest
            # test sets keep the training pool large and comparable.
            unseen = stratified_subsample(d, np.where(d.centre == c)[0], a.n_test, rng0)
            pool = np.where(d.centre != c)[0]
            target = strata(d, unseen)
            best = None
            for seed in range(a.seeds):
                rng = np.random.default_rng(1000 * c + seed)
                pats = np.unique(d.patient[pool])
                pick = rng.permutation(pats)
                for k in range(2, len(pats) - 1):     # SMALLEST group that matches
                    held = set(pick[:k].tolist())
                    hm = np.isin(d.patient[pool], list(held))
                    seen_pool, train = pool[hm], pool[~hm]
                    if len(set(d.y[train].tolist())) < 2:
                        continue
                    sample, ratio = matched_sample(d, seen_pool, target, rng)
                    if len(sample) < 200 or len(set(d.y[sample].tolist())) < 2:
                        continue
                    if best is None or ratio > best[0]:
                        best = (ratio, train, sample)
                    if ratio >= 0.98:
                        break
                if best and best[0] >= 0.98:
                    break
            if best is None:
                print(f"  centre {c}: could not build a matched seen-set, skipped")
                continue
            ratio, train, seen = best
            # equalize training-set size across folds: otherwise fold-to-fold
            # differences in train size masquerade as site effects
            train = stratified_subsample(d, train, a.n_train, rng0)
            p_un = logistic_probe(emb[train], d.y[train], emb[unseen], d.y[unseen],
                                  2, device=a.probe_device)
            p_se = logistic_probe(emb[train], d.y[train], emb[seen], d.y[seen],
                                  2, device=a.probe_device)
            dl = p_se["auc"] - p_un["auc"]
            deltas.append(dl); seens.append(p_se["auc"]); unseens.append(p_un["auc"])
            print(f"  centre {c}: match {ratio:.2f}  train n={len(train)} "
                  f"(pos {d.y[train].mean():.3f})")
            print(f"      unseen  {describe(d, unseen)}  AUC {p_un['auc']:.4f}")
            print(f"      seen    {describe(d, seen)}  AUC {p_se['auc']:.4f}")
            print(f"      delta_site = {dl:+.4f}")
        if deltas:
            m, s = float(np.mean(deltas)), float(np.std(deltas))
            summary[kind] = (m, s, float(np.mean(seens)), float(np.mean(unseens)), len(deltas))
            print(f"  --> delta_site_paired {m:+.4f} +- {s:.4f} over {len(deltas)} centres "
                  f"(seen {np.mean(seens):.4f} vs unseen {np.mean(unseens):.4f})")

    print("\n" + "=" * 92)
    print("PAIRED delta_site: same model, two matched test sets, one differs only by unseen site")
    print("=" * 92)
    print(f"{'encoder':<16} {'AUC seen':>10} {'AUC unseen':>12} {'delta_site':>20} {'folds':>6} {'verdict':>12}")
    for k, (m, s, se, un, n) in summary.items():
        res = "resolvable" if abs(m) > 2 * s / max(np.sqrt(n), 1) else "within noise"
        print(f"{k:<16} {se:>10.4f} {un:>12.4f} {m:>+13.4f} +-{s:<5.4f} {n:>6d} {res:>12}")
    print("\nPositive delta_site = an unseen hospital genuinely costs accuracy.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="phikon,raw")
    ap.add_argument("--grades", default="micro,macro")
    ap.add_argument("--max-tiles-per-slide", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--probe-device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
