"""Figure: cross-patient expression prediction on six HEST-Benchmark tasks: leaderboard reproduction, matched pretraining
contrasts, and the flow model against the official ridge probe on the same features."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
R = json.load(open(f"{D}/hest/hest_ridge.json"))
TASKS = [("LUNG", "Lung"), ("PAAD", "Pancreas"), ("SKCM", "Melanoma"), ("COAD", "Colon"), ("READ", "Rectum"), ("LYMPH_IDC", "Lymph node")]
FL = {t: json.load(open(f"{D}/hest/hest_flow_{t}_phikon_s0.json")) for t, _ in TASKS}
PAIRS = [("path_dino_vits_lunit", "in_dino_vits", "ViT-S/16: Lunit vs ImageNet"), ("path_dino_vitb_kaiko", "in_dino_vitb", "ViT-B/16: Kaiko vs ImageNet")]
pw = {t: sum(int(a > b) for p, q, _ in PAIRS for a, b in zip(R[t][p]["per_split"], R[t][q]["per_split"])) for t, _ in TASKS}
pn = {t: 2 * len(R[t]["phikon"]["per_split"]) for t, _ in TASKS}
fw = {t: sum(int(s["flow"] > s["ridge"]) for s in FL[t]["splits"]) for t, _ in TASKS}
fn = {t: len(FL[t]["splits"]) for t, _ in TASKS}
gains = [s["flow"] - s["ridge"] for t, _ in TASKS for s in FL[t]["splits"]]
gate = [(R[t][e]["published"], R[t][e]["mean"]) for t, _ in TASKS for e in R[t] if R[t][e]["published"] is not None]
maxdiff = max(abs(a - b) for a, b in gate)
for t, _ in TASKS:                                            # the flow script's ridge must equal the gate's ridge
    assert np.allclose([s["ridge"] for s in FL[t]["splits"]], R[t]["phikon"]["per_split"], atol=2e-3), t
task_better = sum(int(FL[t]["mean_flow"] > FL[t]["mean_ridge"]) for t, _ in TASKS)

fig = plt.figure(figsize=(14.6, 8.0), dpi=200)
fig.text(0.04, 0.962, f"Held-out patients, six organs: the flow model beats the official ridge probe in {sum(fw.values())} of {sum(fn.values())} splits; "
         f"pathology pretraining wins {sum(pw.values())} of {sum(pn.values())}", fontsize=13, weight="bold", ha="left")
fig.text(0.04, 0.935, "HEST-Benchmark, official splits and genes: predict the 50 most variable genes of each spot from its H&E patch; train on some patients, test on others. "
         "Score = mean Pearson r over genes.\nFirst three panels use the benchmark's own probe (frozen features, PCA 256 + ridge). "
         "Last panel: same Phikon features and splits, ridge probe vs neighbourhood Transformer + flow head.\n"
         "Differences are candidate minus reference; negative ones are in red. The lymph-node task is the exception in every panel.",
         fontsize=9, color=INK2, ha="left", va="top", linespacing=1.45)
hand = [Line2D([], [], color=BLUE, marker="o", ls="none", ms=8, mec=SURF, mew=1.2, label="Pathology-pretrained encoder"),
        Line2D([], [], color=ORANGE, marker="o", ls="none", ms=8, mec=SURF, mew=1.2, label="ImageNet-pretrained twin (same architecture and method)"),
        Line2D([], [], color=INK2, marker="s", ls="none", ms=7.5, mec=SURF, mew=1.2, label="Official ridge probe"),
        Line2D([], [], color=BLUE, marker="D", ls="none", ms=7.5, mec=SURF, mew=1.2, label="Flow model")]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.037, 0.858), ncol=4, frameon=False, fontsize=8.8, labelcolor=INK, handletextpad=0.3,
           columnspacing=1.8, borderaxespad=0)
B, H = 0.075, 0.715

# ---------------------------------------------------------------- reproduction of the public leaderboard
ax = fig.add_axes([0.045, 0.30, 0.165, 0.30])
lo, hi = 0.10, 0.60
ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, zorder=1)
ax.scatter([a for a, _ in gate], [b for _, b in gate], s=46, color=INK2, edgecolor=SURF, linewidth=1.2, zorder=3)
ax.text(0.05, 0.95, f"{len(gate)} public values\nlargest gap {maxdiff:.3f}", transform=ax.transAxes, fontsize=8.4, va="top", ha="left", linespacing=1.4)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal"); ax.set_xticks([0.2, 0.3, 0.4, 0.5]); ax.set_yticks([0.2, 0.3, 0.4, 0.5])
ax.set_xlabel("public leaderboard value", fontsize=9); ax.set_ylabel("our pipeline", fontsize=9)
ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Pipeline reproduces the leaderboard", fontsize=9.6, loc="left", color=INK, pad=7)

# ---------------------------------------------------------------- one row per official split
rows, y = [], 0
for t, tn in TASKS:
    for i in range(fn[t]):
        rows.append((t, i, y, f"{tn} · split {i + 1}")); y -= 1
    y -= 0.7


def panel(left, ttl, getter, c_lo, c_hi, m_lo, m_hi, labels, counts):
    ax = fig.add_axes([left, B, 0.19, H])
    for t, i, yy, lab in rows:
        a, b = getter(t, i)                                  # a = candidate, b = reference
        ax.plot([b, a], [yy, yy], color=GRID, lw=3.2, solid_capstyle="round", zorder=1)
        ax.scatter([b], [yy], s=54, color=c_lo, marker=m_lo, edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.scatter([a], [yy], s=54, color=c_hi, marker=m_hi, edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.text(max(a, b) + 0.016, yy, f"{a - b:+.3f}", fontsize=8.2, color=INK if a > b else "#b3261e", va="center", ha="left")
    ax.set_yticks([r[2] for r in rows]); ax.set_yticklabels([r[3] for r in rows] if labels else [], fontsize=8.6, color=INK)
    ax.set_xlim(0.02, 0.80); ax.set_ylim(min(r[2] for r in rows) - 0.7, 0.7); ax.set_xticks([0.1, 0.3, 0.5, 0.7])
    ax.set_xlabel("mean Pearson r, held-out patients", fontsize=9)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5); ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"{ttl}  ·  {counts}", fontsize=9.6, loc="left", color=INK, pad=7)

for j, (p, q, ttl) in enumerate(PAIRS):
    w = sum(int(a > b) for t, _ in TASKS for a, b in zip(R[t][p]["per_split"], R[t][q]["per_split"]))
    panel(0.345 + 0.22 * j, ttl, lambda t, i, p=p, q=q: (R[t][p]["per_split"][i], R[t][q]["per_split"][i]), ORANGE, BLUE, "o", "o", j == 0, f"{w} of {len(rows)}")
panel(0.785, "Flow model vs ridge", lambda t, i: (FL[t]["splits"][i]["flow"], FL[t]["splits"][i]["ridge"]), INK2, BLUE, "s", "D", False, f"{sum(fw.values())} of {len(rows)}")
fig.savefig(f"{OUT}/fig_hest_cross_patient.png", dpi=200)
fig.savefig(f"{OUT}/fig_hest_cross_patient.pdf", dpi=300)
summary = {"tasks": [t for t, _ in TASKS], "gate": {"n": len(gate), "max_abs_diff": round(maxdiff, 4)},
           "matched_pairs": {"wins": sum(pw.values()), "total": sum(pn.values()), "per_task": {t: f"{pw[t]}/{pn[t]}" for t, _ in TASKS}},
           "flow_vs_ridge": {"splits_flow_higher": sum(fw.values()), "splits": sum(fn.values()), "per_task": {t: f"{fw[t]}/{fn[t]}" for t, _ in TASKS},
                             "mean_gain_over_splits": round(float(np.mean(gains)), 4), "tasks_flow_better": task_better,
                             "mean_gain_over_tasks": round(float(np.mean([FL[t]["mean_flow"] - FL[t]["mean_ridge"] for t, _ in TASKS])), 4),
                             "task_means": {t: {"flow": FL[t]["mean_flow"], "ridge": FL[t]["mean_ridge"], "ridge_context": FL[t]["mean_ridge_context"]} for t, _ in TASKS},
                             "crps_rel_vs_climatology_per_split": {t: [s["crps_rel_vs_climatology"] for s in FL[t]["splits"]] for t, _ in TASKS},
                             "splits_crps_better_than_climatology": int(sum(s["crps_rel_vs_climatology"] < 0 for t, _ in TASKS for s in FL[t]["splits"]))},
           "preregistered_rule": {"first_three_tasks": "passed (6 of 7 splits, mean gain above 0.01)",
                                  "all_six_tasks_same_rule": "split criterion not met" if sum(fw.values()) / sum(fn.values()) < 6 / 7 else "met"}}
json.dump(summary, open(f"{OUT}/hest_cross_patient_summary.json", "w"), indent=1)
print(json.dumps(summary))
