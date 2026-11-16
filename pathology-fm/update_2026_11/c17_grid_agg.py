"""Aggregate the pre-registered CAMELYON17 grid: per-encoder table, paired contrasts with patient-level bootstrap, claim rules."""
import sys, json, os
from c17_common import *
M = load_meta(); y, patient, centre = M["y"], M["patient"], M["centre"]
ORDER = ["raw", "in_sup_vits", "in_dino_vits", "path_dino_vits_lunit", "in_sup_vitb", "in_dino_vitb", "path_dino_vitb_kaiko", "phikon",
         "path_dinov2_vitl_phikon2"]
enc = [e for e in ORDER if os.path.exists(f"{HERE}/c17_grid_{e}.json")]
J = {e: json.load(open(f"{HERE}/c17_grid_{e}.json")) for e in enc}
S = {e: dict(np.load(f"{HERE}/c17_grid_scores_{e}.npz")) for e in enc}
by_pat = [np.where(patient == p)[0] for p in np.unique(patient)]
rng = np.random.default_rng(0); B = 2000
boot = {k: np.zeros((B, len(enc))) for k in ("oof", "loco", "sel")}
for b in range(B):
    idx = np.concatenate([by_pat[i] for i in rng.integers(0, len(by_pat), len(by_pat))]); yy = y[idx]
    for j, e in enumerate(enc):
        boot["oof"][b, j] = fast_auc(S[e]["oof"][0][idx], yy); boot["loco"][b, j] = fast_auc(S[e]["loco_M0"][idx], yy)
        boot["sel"][b, j] = fast_auc(S[e]["sel"][idx], yy)
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
out = {"encoders": enc, "table": {}, "contrasts": {}, "E4": {}, "bootstrap": "patient-cluster, B=2000, 34 patients"}
for j, e in enumerate(enc):
    le = J[e]["E1_label_eff"]; L = J[e]["E3_E4_loco"]
    out["table"][e] = {"dim": J[e]["dim"], "E1_mean_by_budget": {n: round(float(np.mean(v)), 4) for n, v in le.items()},
                       "E1_min_split_by_budget": {n: round(float(np.min(v)), 4) for n, v in le.items()},
                       "E2_pooled_oof_auc": round(float(J[e]["E2_pooled_oof_auc_per_repeat"][0]), 4), "E2_ci95": ci(boot["oof"][:, j]),
                       "E2_pooled_mean_over_repeats": round(float(np.mean(J[e]["E2_pooled_oof_auc_per_repeat"])), 4),
                       "E2_fold_mean_over_repeats": round(float(np.mean(J[e]["E2_fold_mean_auc_per_repeat"])), 4),
                       "E3_per_hospital": L["M0"]["per_hospital_auc"], "E3_mean": L["M0"]["mean_auc"], "E3_worst": L["M0"]["worst_auc"],
                       "E3_pooled": L["M0"]["pooled_auc"], "E3_pooled_ci95": ci(boot["loco"][:, j]),
                       "E3_sens_spec_at_0.5": [L["M0"]["sens_at_0.5"], L["M0"]["spec_at_0.5"]],
                       "E3_per_hospital_spec": L["M0"]["per_hospital_spec"], "E3_per_hospital_sens": L["M0"]["per_hospital_sens"]}
    out["E4"][e] = {m: {k: L[m][k] for k in ("per_hospital_auc", "mean_auc", "worst_auc", "pooled_auc", "sens_at_0.5", "spec_at_0.5")}
                    for m in ("M0", "M1", "M2", "M3", "selected")}
    out["E4"][e]["selected_per_hospital"] = [c["selected"] for c in J[e]["E4_selection"]]
    out["E4"][e]["success"] = bool(L["selected"]["worst_auc"] > L["M0"]["worst_auc"] and L["selected"]["mean_auc"] >= L["M0"]["mean_auc"] - 0.005)
CON = {"C1 lunit - in_dino_vits": ("path_dino_vits_lunit", "in_dino_vits"), "C2 kaiko - in_dino_vitb": ("path_dino_vitb_kaiko", "in_dino_vitb"),
       "C3 phikon - in_dino_vitb": ("phikon", "in_dino_vitb"), "C4 phikon - in_sup_vitb": ("phikon", "in_sup_vitb"),
       "S1 in_dino_vitb - in_dino_vits": ("in_dino_vitb", "in_dino_vits"), "S2 phikon2 - phikon": ("path_dinov2_vitl_phikon2", "phikon"),
       "X1 phikon - in_sup_vits": ("phikon", "in_sup_vits"), "X2 kaiko - lunit": ("path_dino_vitb_kaiko", "path_dino_vits_lunit"),
       "X3 lunit - in_sup_vits": ("path_dino_vits_lunit", "in_sup_vits"), "X4 phikon - kaiko": ("phikon", "path_dino_vitb_kaiko")}
for name, (a, b) in CON.items():
    if a not in enc or b not in enc:
        continue
    ia, ib = enc.index(a), enc.index(b); la, lb = J[a]["E1_label_eff"], J[b]["E1_label_eff"]
    d = {"E2_diff": round(out["table"][a]["E2_pooled_oof_auc"] - out["table"][b]["E2_pooled_oof_auc"], 4), "E2_ci95": ci(boot["oof"][:, ia] - boot["oof"][:, ib]),
         "E2_diff_per_repeat": [round(x - z, 4) for x, z in zip(J[a]["E2_pooled_oof_auc_per_repeat"], J[b]["E2_pooled_oof_auc_per_repeat"])],
         "E2_fold_mean_diff_per_repeat": [round(x - z, 4) for x, z in zip(J[a]["E2_fold_mean_auc_per_repeat"], J[b]["E2_fold_mean_auc_per_repeat"])],
         "E1_diff_mean_by_budget": {n: round(float(np.mean(np.array(la[n]) - np.array(lb[n]))), 4) for n in la},
         "E1_splits_positive_by_budget": {n: int((np.array(la[n]) > np.array(lb[n])).sum()) for n in la},
         "E3_per_hospital_diff": [round(x - z, 4) for x, z in zip(out["table"][a]["E3_per_hospital"], out["table"][b]["E3_per_hospital"])],
         "E3_pooled_diff": round(out["table"][a]["E3_pooled"] - out["table"][b]["E3_pooled"], 4), "E3_pooled_ci95": ci(boot["loco"][:, ia] - boot["loco"][:, ib])}
    d["rule_E2_ci_excludes_0_positive"] = bool(d["E2_ci95"][0] > 0)
    d["rule_E1_budgets_2_and_16_at_least_4_of_5"] = bool(d["E1_splits_positive_by_budget"]["2"] >= 4 and d["E1_splits_positive_by_budget"]["16"] >= 4)
    out["contrasts"][name] = d
c1, c2 = out["contrasts"].get("C1 lunit - in_dino_vits"), out["contrasts"].get("C2 kaiko - in_dino_vitb")
if c1 and c2:
    out["claim_pathology_pretraining_helps_at_matched_size"] = bool(all(c[k] for c in (c1, c2) for k in
                                                                        ("rule_E2_ci_excludes_0_positive", "rule_E1_budgets_2_and_16_at_least_4_of_5")))
sl = sorted(J[enc[0]]["E5_per_slide"])
out["E5_per_slide"] = {s: {"hospital": J[enc[0]]["E5_per_slide"][s]["hospital"], "grade": J[enc[0]]["E5_per_slide"][s]["grade"],
                           "tumour": J[enc[0]]["E5_per_slide"][s]["tumour"], "tiles": J[enc[0]]["E5_per_slide"][s]["tiles"],
                           "auc": {e: J[e]["E5_per_slide"][s]["auc"] for e in enc}} for s in sl}
json.dump(out, open(f"{HERE}/c17_grid_summary.json", "w"), indent=1)
print("encoder                      E1@1   @2    @16   all  | E2 pooled [ci]            | LOCO per hospital                      mean  worst pooled  sens  spec")
for e in enc:
    t = out["table"][e]; m = t["E1_mean_by_budget"]
    print(f"{e:27s} {m['1']:.3f} {m['2']:.3f} {m['16']:.3f} {m['all']:.3f} | {t['E2_pooled_oof_auc']:.3f} {t['E2_ci95']} | {t['E3_per_hospital']} {t['E3_mean']:.3f} {t['E3_worst']:.3f} {t['E3_pooled']:.3f} "
          f"{t['E3_sens_spec_at_0.5'][0]:.3f} {t['E3_sens_spec_at_0.5'][1]:.3f}")
for k, d in out["contrasts"].items():
    print(f"{k:30s} E2 {d['E2_diff']:+.4f} {d['E2_ci95']} | E1 splits+ {d['E1_splits_positive_by_budget']} | LOCO diff {d['E3_per_hospital_diff']} pooled {d['E3_pooled_diff']:+.4f} {d['E3_pooled_ci95']}")
print("claim (pathology pretraining helps at matched size):", out.get("claim_pathology_pretraining_helps_at_matched_size"))
for e in enc:
    r = out["E4"][e]
    print(f"E4 {e:27s} M0 worst {r['M0']['worst_auc']:.3f} mean {r['M0']['mean_auc']:.3f} | M1 {r['M1']['worst_auc']:.3f}/{r['M1']['mean_auc']:.3f} M2 {r['M2']['worst_auc']:.3f}/{r['M2']['mean_auc']:.3f} "
          f"M3 {r['M3']['worst_auc']:.3f}/{r['M3']['mean_auc']:.3f} | selected {r['selected_per_hospital']} -> {r['selected']['worst_auc']:.3f}/{r['selected']['mean_auc']:.3f} success {r['success']}")
bad = {s: v for s, v in out["E5_per_slide"].items() if any(a is not None and a < 0.8 for e, a in v["auc"].items() if e != "raw")}
print("slides with AUC < 0.8 for at least one encoder:")
for s, v in bad.items():
    print("  ", s, "hospital", v["hospital"], v["grade"], f"tumour {v['tumour']}/{v['tiles']}", {e[:14]: round(v["auc"][e], 2) for e in enc if e != "raw"})
