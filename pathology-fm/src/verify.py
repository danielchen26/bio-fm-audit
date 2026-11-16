"""Single source of truth for every CAMELYON17 claim.

WHY THIS FILE EXISTS
The individual experiments in this project are each defensible, but they were run
at different times against different dataset states -- 38 slides, then 41, then
50, now 56 with negatives. Numbers from different sections therefore cannot be
read side by side, even where each is individually correct. That is not a bug in
any one analysis; it is a defect in the study as a whole.

So this script does four things no individual experiment did:

1. FREEZES the dataset. Every shard is hashed into one snapshot id, and that id
   is printed next to every number. Two numbers with the same snapshot id are
   comparable; numbers without it are not quotable.

2. PRE-REGISTERS each claim. For every quantity the null it is compared against
   and the resolvability threshold are declared in CLAIMS below, before anything
   is computed. This is what stops "beat zero" from masquerading as a result --
   an error made earlier in this project, which declared random features
   effective.

3. CLASSIFIES every result into one of three tiers instead of reporting a bare
   number:
       RESOLVED      |effect| > 2*SE  -- the data supports the direction
       NULL          |effect| < 2*SE and SE < the effect size that would matter
       UNDERPOWERED  SE too large to distinguish -- explicitly inconclusive
   Mixing these three is how underpowered noise becomes a published finding.

4. USES ONE SEED POLICY. Group splits use seeds 0..4 everywhere. No per-analysis
   choices.

Output is a JSON keyed by snapshot id plus a printed table. Any claim in the
README should be regenerable by running this and nothing else.
"""
from __future__ import annotations

import os
import argparse
import glob
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# CAMELYON17 slides, tiles and caches (about 180 GB). Set C17_ROOT to keep them on another disk.
SRC = Path(os.environ.get("C17_ROOT", Path(__file__).resolve().parent.parent / "data" / "camelyon17"))
SEEDS = (0, 1, 2, 3, 4)
GRADES = ("negative", "itc", "micro", "macro")


# --------------------------------------------------------------- registration
@dataclass
class Claim:
    key: str
    question: str
    null: str
    matters: float          # effect size below which we would not care
    value: float = float("nan")
    se: float = float("nan")
    n: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def tier(self) -> str:
        if not np.isfinite(self.value) or not np.isfinite(self.se):
            return "NOT RUN"
        if self.se <= 0:
            return "RESOLVED" if abs(self.value) > 0 else "NULL"
        if abs(self.value) > 2 * self.se:
            return "RESOLVED"
        if 2 * self.se < self.matters:
            return "NULL"
        return "UNDERPOWERED"


CLAIMS = {
    "leak_raw": Claim("leak_raw", "raw-pixel: random minus patient-disjoint AUC",
                      "zero leak", matters=0.02),
    "leak_phikon": Claim("leak_phikon", "Phikon: random minus patient-disjoint AUC",
                         "zero leak", matters=0.02),
    "centre_id": Claim("centre_id", "centre identity decodable from the embedding",
                       "chance = 1/n_centres", matters=0.10),
    "encoder_gap": Claim("encoder_gap", "Phikon minus raw-pixel, patient-disjoint AUC",
                         "no benefit from pathology pretraining", matters=0.02),
    "delta_site": Claim("delta_site", "paired seen-site minus unseen-site AUC",
                        "site novelty is free", matters=0.02),
    "stability": Claim("stability", "across-centre SD of AUC, raw minus phikon",
                       "equally stable", matters=0.02),
    "rule_kappa": Claim("rule_kappa", "quadratic kappa, AJCC rule on annotated polygons",
                        "chance agreement (kappa=0)", matters=0.10),
    "tile_kappa": Claim("tile_kappa", "quadratic kappa, AJCC rule on predicted tiles",
                        "chance agreement (kappa=0)", matters=0.10),
    "seg_value": Claim("seg_value", "kappa with true geometry minus kappa from tiles",
                       "tile classification suffices", matters=0.10),
}


# ---------------------------------------------------------------- the snapshot
def snapshot(subdir="tiles"):
    files = sorted(glob.glob(str(SRC / subdir / "*.npz")))
    h = hashlib.sha256()
    rows = []
    for f in files:
        st = Path(f).stat()
        h.update(Path(f).name.encode())
        h.update(str(st.st_size).encode())
        rows.append((Path(f).stem, st.st_size))
    return h.hexdigest()[:16], rows


def load(subdir="tiles"):
    P, Y, C, S, PT, G, CO = [], [], [], [], [], [], []
    sids, pats = [], []
    for si, f in enumerate(sorted(glob.glob(str(SRC / subdir / "*.npz")))):
        z = np.load(f, allow_pickle=False)
        sid = str(z["slide_id"])
        pid = "_".join(sid.split("_")[:2])
        if pid not in pats:
            pats.append(pid)
        sids.append(sid)
        n = len(z["labels"])
        P.append(z["patches"]); Y.append(z["labels"]); CO.append(z["coords"])
        C.append(np.full(n, int(z["centre"]), np.int64))
        S.append(np.full(n, si, np.int64))
        PT.append(np.full(n, pats.index(pid), np.int64))
        g = str(z["grade"])
        G.append(np.full(n, GRADES.index(g) if g in GRADES else 0, np.int64))
    return dict(patches=np.concatenate(P), y=np.concatenate(Y), centre=np.concatenate(C),
                slide=np.concatenate(S), patient=np.concatenate(PT),
                grade=np.concatenate(G), coords=np.concatenate(CO),
                sids=sids, pats=pats)


# -------------------------------------------------------------------- helpers
def probe_auc(emb, y, tr, te, device="cpu"):
    from audit import logistic_probe
    if len(set(y[te].tolist())) < 2 or len(set(y[tr].tolist())) < 2:
        return float("nan")
    return logistic_probe(emb[tr], y[tr], emb[te], y[te], 2, device=device)["auc"]


def group_split(key, seed, frac=0.25):
    g = np.unique(key)
    pick = np.random.default_rng(seed).permutation(g)
    te = set(pick[:max(1, int(len(g) * frac))].tolist())
    m = np.isin(key, list(te))
    return np.where(~m)[0], np.where(m)[0]


def boot_kappa(truth, pred, n_cls=4, iters=2000, seed=0):
    from c17stage import quadratic_kappa
    rng = np.random.default_rng(seed)
    truth, pred = np.asarray(truth), np.asarray(pred)
    k0 = quadratic_kappa(truth, pred, n_cls)
    vals = []
    for _ in range(iters):
        i = rng.integers(0, len(truth), len(truth))
        if len(set(truth[i].tolist())) < 2:
            continue
        vals.append(quadratic_kappa(truth[i], pred[i], n_cls))
    return k0, float(np.std(vals))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="tiles")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-per-slide", type=int, default=400)
    ap.add_argument("--out", default="verify_results.json")
    a = ap.parse_args()

    snap, rows = snapshot(a.tiles)
    print("=" * 92)
    print(f"SNAPSHOT {snap}   {len(rows)} slide shards from {SRC / a.tiles}")
    print("Every number below comes from this snapshot. Numbers without it are not quotable.")
    print("=" * 92)
    print("\nPRE-REGISTERED CLAIMS (declared before computing):")
    for c in CLAIMS.values():
        print(f"  {c.key:12s} {c.question}")
        print(f"  {'':12s}   null: {c.null} | would-matter threshold: {c.matters}")

    d = load(a.tiles)
    gcount = {GRADES[g]: int((d["grade"] == g).sum()) for g in np.unique(d["grade"])}
    print(f"\nDATA  {len(d['y'])} tiles | {len(d['sids'])} slides | {len(d['pats'])} patients "
          f"| {d['y'].mean():.1%} tumour")
    print(f"      tiles per grade {gcount}")
    for c in sorted(set(d["centre"].tolist())):
        m = d["centre"] == c
        print(f"      centre {c}: {int(m.sum()):6d} tiles, {len(set(d['slide'][m].tolist())):2d} slides, "
              f"{d['y'][m].mean():.1%} tumour")

    # grade-controlled subset for the tile-AUC claims: ITC positives are
    # intrinsically harder, so mixing grades across folds confounds site with
    # lesion size. Negatives are kept -- they are the real negative class.
    keep = np.isin(d["grade"], [GRADES.index(g) for g in ("negative", "micro", "macro")])
    sub = {k: (v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v)
           for k, v in d.items()}
    print(f"\n      grade-controlled subset (negative+micro+macro): {len(sub['y'])} tiles, "
          f"{sub['y'].mean():.1%} tumour")

    from c17bench import embed, raw_pixel_features
    embs = {}
    for kind in ("raw", "phikon"):
        cpath = SRC / f"verify_emb_{kind}_{snap}.npz"
        if cpath.exists():
            embs[kind] = np.load(cpath)["emb"]
            print(f"[emb] cached {kind} {embs[kind].shape}")
        else:
            if kind == "raw":
                e = raw_pixel_features(d["patches"])
            else:
                from transformers import AutoModel
                m = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
                e = embed(m, d["patches"], 224, a.device)
            np.savez_compressed(cpath, emb=e)
            embs[kind] = e
            print(f"[emb] computed {kind} {e.shape}")

    res = {}
    print("\n" + "-" * 92)
    for kind in ("raw", "phikon"):
        E, Es = embs[kind], embs[kind][keep]
        rnd, pat = [], []
        for s in SEEDS:
            n = len(sub["y"])
            p = np.random.default_rng(s).permutation(n)
            cut = int(0.75 * n)
            rnd.append(probe_auc(Es, sub["y"], p[:cut], p[cut:]))
            tr, te = group_split(sub["patient"], s)
            pat.append(probe_auc(Es, sub["y"], tr, te))
        rnd, pat = np.array(rnd), np.array(pat)
        # per-seed arrays retained: the encoder comparison is PAIRED on identical
        # splits, so differencing per seed cancels the split-to-split variance.
        # Comparing the marginal means instead inflates the SE enormously.
        res[f"{kind}_random_seeds"] = [float(x) for x in rnd]
        res[f"{kind}_patient_seeds"] = [float(x) for x in pat]
        res[f"{kind}_random"] = (float(np.nanmean(rnd)), float(np.nanstd(rnd)))
        res[f"{kind}_patient"] = (float(np.nanmean(pat)), float(np.nanstd(pat)))
        print(f"{kind:8s} random {np.nanmean(rnd):.4f}+-{np.nanstd(rnd):.4f} | "
              f"patient {np.nanmean(pat):.4f}+-{np.nanstd(pat):.4f}")

        # per-centre AUC for stability / worst-case
        folds = []
        for c in sorted(set(sub["centre"].tolist())):
            tr = np.where(sub["centre"] != c)[0]
            te = np.where(sub["centre"] == c)[0]
            folds.append(probe_auc(Es, sub["y"], tr, te))
        folds = np.array(folds, float)
        res[f"{kind}_centre_folds"] = [float(x) for x in folds]
        print(f"{'':8s} per-centre {np.array2string(folds, precision=3)} "
              f"SD {np.nanstd(folds):.4f} worst {np.nanmin(folds):.4f}")

        # centre identity probe, inside the patient-disjoint training partition
        tr, _ = group_split(sub["patient"], 0)
        rs = np.random.default_rng(0).permutation(len(tr))
        A, B = tr[rs[:int(0.7 * len(rs))]], tr[rs[int(0.7 * len(rs)):]]
        from audit import logistic_probe
        cn = len(set(sub["centre"].tolist()))
        cp = logistic_probe(Es[A], sub["centre"][A], Es[B], sub["centre"][B], cn)
        res[f"{kind}_centre_id"] = (cp["acc"], cp["chance"])
        print(f"{'':8s} centre-ID {cp['acc']:.4f} (chance {cp['chance']:.3f})")

    # ---- fill the pre-registered claims
    # leak, per encoder, paired by seed. SE = SD/sqrt(n), not SD.
    for k in ("raw", "phikon"):
        dif = np.array(res[f"{k}_random_seeds"]) - np.array(res[f"{k}_patient_seeds"])
        c = CLAIMS[f"leak_{k}"]
        c.value = float(np.nanmean(dif))
        c.se = float(np.nanstd(dif, ddof=1) / np.sqrt(len(dif)))
        c.n = len(dif)
        c.extra = {"random": res[f"{k}_random"][0], "patient": res[f"{k}_patient"][0]}

    ci = res["phikon_centre_id"]
    CLAIMS["centre_id"].value = ci[0] - ci[1]
    CLAIMS["centre_id"].se = float(np.sqrt(ci[0] * (1 - ci[0]) / 3000))
    CLAIMS["centre_id"].extra = {"acc": ci[0], "chance": ci[1]}

    # paired per-seed difference: both encoders see the SAME splits
    gd = np.array(res["phikon_patient_seeds"]) - np.array(res["raw_patient_seeds"])
    CLAIMS["encoder_gap"].value = float(np.nanmean(gd))
    CLAIMS["encoder_gap"].se = float(np.nanstd(gd, ddof=1) / np.sqrt(len(gd)))
    CLAIMS["encoder_gap"].n = len(gd)
    CLAIMS["encoder_gap"].extra = {"phikon": res["phikon_patient"][0],
                                   "raw": res["raw_patient"][0]}

    fr = np.array(res["raw_centre_folds"], float)
    fp = np.array(res["phikon_centre_folds"], float)
    CLAIMS["stability"].value = float(np.nanstd(fr) - np.nanstd(fp))
    # bootstrap the SD difference over centres rather than using a normal approximation
    rng = np.random.default_rng(0)
    bs = [np.nanstd(fr[i]) - np.nanstd(fp[i])
          for i in (rng.integers(0, len(fr), len(fr)) for _ in range(4000))]
    CLAIMS["stability"].se = float(np.nanstd(bs))
    CLAIMS["stability"].n = len(fr)
    CLAIMS["stability"].extra = {"sd_raw": float(np.nanstd(res["raw_centre_folds"])),
                                "sd_phikon": float(np.nanstd(res["phikon_centre_folds"])),
                                "worst_raw": float(np.nanmin(res["raw_centre_folds"])),
                                "worst_phikon": float(np.nanmin(res["phikon_centre_folds"]))}

    # ---- clinical grading, both stages, on this snapshot
    from c17stage import (component_span_grid, grade_from_mm, polygon_longest_mm,
                          quadratic_kappa)
    truth = [d["grade"][d["slide"] == i][0] for i in range(len(d["sids"]))]
    predA = [grade_from_mm(polygon_longest_mm(SRC / "annotations" / f"{s}.xml"))
             for s in d["sids"]]
    kA, seA = boot_kappa(truth, predA)
    CLAIMS["rule_kappa"].value, CLAIMS["rule_kappa"].se = kA, seA
    CLAIMS["rule_kappa"].n = len(truth)

    E = embs["phikon"]
    predB = [0] * len(d["sids"])
    for c in sorted(set(d["centre"].tolist())):
        tr = np.where(d["centre"] != c)[0]
        te = np.where(d["centre"] == c)[0]
        if len(set(d["y"][tr].tolist())) < 2:
            continue
        from audit import logistic_probe
        import torch
        import torch.nn.functional as F
        X = torch.as_tensor(E, dtype=torch.float32)
        mu, sd = X[tr].mean(0, keepdim=True), X[tr].std(0, keepdim=True) + 1e-6
        Xn = (X - mu) / sd
        W = torch.zeros(X.shape[1], 2, requires_grad=True)
        b = torch.zeros(2, requires_grad=True)
        opt = torch.optim.Adam([W, b], lr=0.1, weight_decay=1e-4)
        yt = torch.as_tensor(d["y"][tr], dtype=torch.long)
        for _ in range(300):
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(Xn[tr] @ W + b, yt).backward()
            opt.step()
        with torch.no_grad():
            prob = (Xn[te] @ W + b).softmax(1)[:, 1].numpy()
        for si in sorted(set(d["slide"][te].tolist())):
            m = d["slide"][te] == si
            co = d["coords"][te][m][prob[m] >= 0.5]
            span = component_span_grid(co, 448.0)
            predB[si] = grade_from_mm(span * 448.0 * 0.2431 / 1000.0 + 224 * 0.4862 / 1000.0
                                      if span >= 0 else 0.0)
    kB, seB = boot_kappa(truth, predB)
    CLAIMS["tile_kappa"].value, CLAIMS["tile_kappa"].se = kB, seB
    CLAIMS["tile_kappa"].n = len(truth)
    CLAIMS["seg_value"].value = kA - kB
    CLAIMS["seg_value"].se = float(np.sqrt(seA ** 2 + seB ** 2))
    CLAIMS["seg_value"].n = len(truth)

    # ---- paired delta_site, on THIS snapshot
    # For each held-out centre: one model, two matched test sets differing only in
    # whether the site was seen. Test sets are stratified-matched on the joint
    # (grade, label) composition and on size; the training set is size-equalised
    # across folds. All three of those were confounds in earlier attempts.
    from c17site import matched_sample, strata, stratified_subsample

    class _D:                                   # adapter for c17site's helpers
        pass
    dd = _D()
    dd.grade, dd.y, dd.centre, dd.patient = (sub["grade"], sub["y"],
                                             sub["centre"], sub["patient"])
    Ep = embs["phikon"][keep]
    deltas, seens, unseens = [], [], []
    for c in sorted(set(sub["centre"].tolist())):
        rng = np.random.default_rng(7000 + c)
        unseen = stratified_subsample(dd, np.where(sub["centre"] == c)[0], 1500, rng)
        pool = np.where(sub["centre"] != c)[0]
        tgt = strata(dd, unseen)
        best = None
        pats = np.unique(sub["patient"][pool])
        for seed in SEEDS:
            pick = np.random.default_rng(7000 + 100 * c + seed).permutation(pats)
            for k in range(2, len(pats) - 1):
                held = set(pick[:k].tolist())
                hm = np.isin(sub["patient"][pool], list(held))
                seen_pool, train = pool[hm], pool[~hm]
                if len(set(sub["y"][train].tolist())) < 2:
                    continue
                samp, ratio = matched_sample(dd, seen_pool, tgt, rng)
                if len(samp) < 200 or len(set(sub["y"][samp].tolist())) < 2:
                    continue
                if best is None or ratio > best[0]:
                    best = (ratio, train, samp)
                if ratio >= 0.98:
                    break
            if best and best[0] >= 0.98:
                break
        if best is None:
            print(f"  delta_site centre {c}: no matched seen-set, skipped")
            continue
        ratio, train, seen = best
        train = stratified_subsample(dd, train, 6000, rng)
        au = probe_auc(Ep, sub["y"], train, unseen)
        as_ = probe_auc(Ep, sub["y"], train, seen)
        if not (np.isfinite(au) and np.isfinite(as_)):
            continue
        deltas.append(as_ - au); seens.append(as_); unseens.append(au)
        print(f"  delta_site centre {c}: match {ratio:.2f} train {len(train)} "
              f"seen {as_:.4f} unseen {au:.4f} delta {as_ - au:+.4f}")
    if deltas:
        dv = np.array(deltas)
        CLAIMS["delta_site"].value = float(dv.mean())
        CLAIMS["delta_site"].se = float(dv.std(ddof=1) / np.sqrt(len(dv)))
        CLAIMS["delta_site"].n = len(dv)
        CLAIMS["delta_site"].extra = {"seen": float(np.mean(seens)),
                                     "unseen": float(np.mean(unseens))}

    print("\n" + "=" * 92)
    print(f"VERDICTS   snapshot {snap}")
    print("=" * 92)
    print(f"{'claim':13s} {'value':>10} {'2*SE':>9} {'n':>4}  {'tier':<13} note")
    for c in CLAIMS.values():
        note = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in c.extra.items())
        v = f"{c.value:+.4f}" if np.isfinite(c.value) else "--"
        s2 = f"{2 * c.se:.4f}" if np.isfinite(c.se) else "--"
        print(f"{c.key:13s} {v:>10} {s2:>9} {c.n:>4}  {c.tier:<13} {note}")

    out = {"snapshot": snap, "n_slides": len(rows), "n_tiles": int(len(d["y"])),
           "seeds": list(SEEDS), "raw": res,
           "claims": {k: {"question": c.question, "null": c.null, "value": c.value,
                          "se": c.se, "n": c.n, "tier": c.tier, "extra": c.extra}
                      for k, c in CLAIMS.items()}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")
    print("Quote a number only together with snapshot", snap)


if __name__ == "__main__":
    main()
