"""Figure (Phase 4): both training budgets chosen on a validation strip of the training slides, never on the test patients.
Our flow model against the public STFlow code and the official ridge probe; identical Phikon features and HEST-Benchmark splits."""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SRC, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#b3261e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
J = json.load(open(SRC)); G = J["_overall"]; M = G["six_organ_mean"]
NAMES = [("LUNG", "Lung"), ("PAAD", "Pancreas"), ("SKCM", "Melanoma"), ("COAD", "Colon"), ("READ", "Rectum"), ("LYMPH_IDC", "Lymph node")]
BUD = [250, 500, 1000, 2000, 4000, 8000]
rows = [(t, tn, s) for t, tn in NAMES for s in J[t]["splits"]]
pa, pb = G["primary_A_ours_vs_stflow"], G["secondary_B_ours_vs_stflow"]
WORD = {"on par": "on par with", "ahead": "ahead of", "behind": "behind", "mixed": "mixed against"}


def by_rep(m, key):                                            # six-organ mean of one replicate
    return [float(np.mean([np.mean([s["reps"][r][m][key] for s in J[t]["splits"]]) for t, _ in NAMES])) for r in range(3)]

RULES = [("Default budget,\nall training spots", "ours_default", "stflow_default", "phase3_steps4000", "phase3_epoch100"),
         ("Chosen on the validation strip,\nfour fifths of the spots · primary", "ours_A", "stflow_A", "A", "A"),
         ("Same budget, refit on\nall training spots · secondary", "ours_B", "stflow_B", "B", "B"),
         ("Best budget picked on the\ntest patients · leaks", "ours_hindsight", "stflow_hindsight", "hindsight", "hindsight")]

fig = plt.figure(figsize=(14.6, 9.6), dpi=200)
W2 = {"on par": "still on par", "ahead": "ahead", "behind": "behind", "mixed": "mixed"}
fig.text(0.035, 0.968, f"Budget chosen without test patients: our model is {WORD[pa['verdict']]} the public STFlow code; "
         f"{W2[pb['verdict']]} once all training spots are used", fontsize=12.4, weight="bold", ha="left")
fig.text(0.035, 0.945, "HEST-Benchmark, six organs, official patient-held-out splits, 50 genes, the same frozen Phikon embeddings for every method; score = mean Pearson r over genes on held-out patients; three replicates.\n"
         "A contiguous strip holding a fifth of every training slide is kept out of training. STFlow stops by its own rule (best epoch, patience 20) applied to that strip; our model picks among six step counts by the\n"
         f"same strip. Six-organ mean at the selected budget: ours {M['ours_A']:.3f}, STFlow {M['stflow_A']:.3f}; re-using that budget with all training spots: ours {M['ours_B']:.3f}, STFlow {M['stflow_B']:.3f}. "
         f"Ridge probe: {M['ridge']:.3f}.", fontsize=8.8, color=INK2, ha="left", va="top", linespacing=1.45)
hand = [Line2D([], [], color=BLUE, marker="D", lw=1.8, ms=7.5, mec=SURF, mew=1.2, label="Our flow model"),
        Line2D([], [], color=ORANGE, marker="o", lw=1.6, ms=8, mec=SURF, mew=1.2, label="STFlow public code"),
        Line2D([], [], color=INK2, marker="s", lw=1.0, ls=(0, (3, 2)), ms=7, mec=SURF, mew=1.2, label="Official ridge probe"),
        Line2D([], [], color=INK2, lw=2.6, label="range of three replicates")]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.032, 0.884), ncol=4, frameon=False, fontsize=8.7, labelcolor=INK, handletextpad=0.4,
           columnspacing=1.7, borderaxespad=0, handlelength=2.2)
TOP = 0.785

# ---------------------------------------------------------------- top left: six-organ mean under four ways of setting the budget
ax = fig.add_axes([0.178, 0.585, 0.175, TOP - 0.585])
ax.axvline(M["ridge"], color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
for k, (lab, ko, ks, ro, rs) in enumerate(RULES):
    yy = -k; a, f = M[ko], M[ks]; ra, rf = by_rep("ours", ro), by_rep("stflow", rs)
    if k == 1:
        ax.axhspan(yy - 0.46, yy + 0.46, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.plot([min(ra), max(ra)], [yy + 0.17, yy + 0.17], color=BLUE, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.plot([min(rf), max(rf)], [yy - 0.17, yy - 0.17], color=ORANGE, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.scatter([a], [yy + 0.17], s=52, color=BLUE, marker="D", edgecolor=SURF, linewidth=1.1, zorder=5)
    ax.scatter([f], [yy - 0.17], s=58, color=ORANGE, marker="o", edgecolor=SURF, linewidth=1.1, zorder=5)
    ax.text(max(max(ra), a) + 0.004, yy + 0.17, f"{a:.3f}", fontsize=8.2, color=INK, va="center", ha="left")
    ax.text(max(max(rf), f) + 0.004, yy - 0.17, f"{f:.3f}", fontsize=8.2, color=INK, va="center", ha="left")
    dd = a - f
    ax.text(0.4835, yy, "0.000" if abs(dd) < 0.0005 else f"{dd:+.3f}", fontsize=8.3, color=INK if dd > -0.0005 else RED, va="center", ha="right")
ax.text(0.4835, 0.62, "ours − STFlow", fontsize=7.6, color=INK2, va="bottom", ha="right")
ax.text(M["ridge"] - 0.002, 0.62, "ridge", fontsize=7.6, color=INK2, va="bottom", ha="right")
ax.set_yticks([-k for k in range(len(RULES))]); ax.set_yticklabels([r[0] for r in RULES], fontsize=8.3, color=INK, linespacing=1.3)
ax.set_xlim(0.355, 0.484); ax.set_ylim(-len(RULES) + 0.45, 0.95); ax.set_xticks([0.36, 0.38, 0.40, 0.42, 0.44])
ax.set_xlabel("mean Pearson r, six organs", fontsize=8.8, labelpad=2)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.3); ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.text(0.035, TOP + 0.042, "Six-organ mean under four ways of setting the budget", fontsize=9.6, color=INK, ha="left", va="bottom")

# ---------------------------------------------------------------- bottom left: one row per official split, primary result
ax = fig.add_axes([0.118, 0.055, 0.235, 0.445])
ys, y, last = [], 0.0, None
for t, tn, s in rows:
    if last is not None and t != last:
        y -= 0.6
    ys.append(y); last = t; y -= 1
for (t, tn, s), yy in zip(rows, ys):
    a, f, g = s["ours_A"], s["stflow_A"], s["ridge"]; ra = [x["ours"]["A"] for x in s["reps"]]; rf = [x["stflow"]["A"] for x in s["reps"]]
    yo, yf = yy + 0.17, yy - 0.17
    ax.plot([min(ra + rf + [g]) - 0.004, max(ra + rf + [g]) + 0.004], [yy, yy], color=GRID, lw=7.5, solid_capstyle="round", zorder=1)
    ax.plot([min(rf), max(rf)], [yf, yf], color=ORANGE, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.plot([min(ra), max(ra)], [yo, yo], color=BLUE, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.scatter([g], [yy], s=30, color=INK2, marker="s", edgecolor=SURF, linewidth=1.0, zorder=4)
    ax.scatter([f], [yf], s=40, color=ORANGE, marker="o", edgecolor=SURF, linewidth=1.0, zorder=6)
    ax.scatter([a], [yo], s=38, color=BLUE, marker="D", edgecolor=SURF, linewidth=1.0, zorder=7)
    ax.text(0.925, yy, f"{a - f:+.3f}", fontsize=8.1, color=INK if a > f else RED, va="center", ha="right")
ax.text(0.925, 0.75, "ours − STFlow", fontsize=7.5, color=INK2, va="bottom", ha="right")
ax.set_yticks(ys); ax.set_yticklabels([f"{tn} · split {s['split'] + 1}" for _, tn, s in rows], fontsize=8.4, color=INK)
ax.set_xlim(-0.01, 0.93); ax.set_ylim(min(ys) - 0.7, 1.9); ax.set_xticks([0.1, 0.3, 0.5, 0.7])
ax.set_xlabel("mean Pearson r, held-out patients", fontsize=8.8, labelpad=2)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.3); ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.text(0.035, 0.512, f"Primary result, each split  ·  ours higher in {pa['splits_ours_higher']} of {len(rows)}", fontsize=9.6, color=INK, ha="left", va="bottom")

# ---------------------------------------------------------------- right: where validation and test put the best budget
L, R, B = 0.432, 0.992, 0.075; NC = 6; gw = 0.012; pw = (R - L - gw * (NC - 1)) / NC; gh = 0.095; ph = (TOP - B - gh) / 2
fig.text(L - 0.022, TOP + 0.042, "Score against training budget, each curve relative to its own best value  ·  solid: test patients, dashed: validation strip", fontsize=9.6, color=INK, ha="left", va="bottom")
YL = (-0.125, 0.012)
for c_, (t, tn) in enumerate(NAMES):
    reps = [x for s in J[t]["splits"] for x in s["reps"]]
    for r_, (m, col, xs, kt, kv) in enumerate((("ours", BLUE, np.array(BUD, float), "test_by_budget", "val_by_budget"),
                                               ("stflow", ORANGE, np.arange(1, 101, dtype=float), "test_curve", "val_curve"))):
        a2 = fig.add_axes([L + c_ * (pw + gw), TOP - (r_ + 1) * ph - r_ * gh, pw, ph])
        te = np.mean([x[m][kt] for x in reps], 0); va = np.mean([x[m][kv] for x in reps], 0)
        a2.axhline(0, color=INK2, lw=0.6, zorder=1)
        for cur, ls, lw in ((va, (0, (3, 2)), 1.5), (te, "-", 2.0)):
            d = cur - cur.max(); a2.plot(xs, d, color=col, lw=lw, ls=ls, marker="o" if m == "ours" else None, ms=3.2, zorder=3)
            a2.scatter([xs[int(cur.argmax())]], [0], s=44, color=col if ls == "-" else SURF, edgecolor=SURF if ls == "-" else col, linewidth=1.2 if ls == "-" else 1.7, zorder=5, clip_on=False)
        if m == "ours":
            a2.set_xscale("log"); a2.set_xlim(200, 10000); a2.set_xticks([250, 1000, 4000]); a2.set_xticklabels(["250", "1,000", "4,000"]); a2.minorticks_off()
            a2.text(0.0, 1.04, tn, transform=a2.transAxes, fontsize=8.8, color=INK, ha="left", va="bottom", weight="bold")
        else:
            a2.set_xlim(0, 102); a2.set_xticks([1, 50, 100])
        a2.set_ylim(*YL); a2.set_yticks([-0.10, -0.05, 0]); a2.tick_params(length=2, width=0.5, labelsize=7.8)
        a2.yaxis.grid(True, color=GRID, lw=0.5); a2.set_axisbelow(True)
        for sp in ("top", "right"):
            a2.spines[sp].set_visible(False)
        if c_ == 0:
            a2.set_ylabel("score − its best", fontsize=8.4)
        else:
            a2.set_yticklabels([])
        a2.text(0.97, 0.06, f"best: test {int(xs[int(te.argmax())]):,}\nvalidation {int(xs[int(va.argmax())]):,}", transform=a2.transAxes, fontsize=7.4, color=INK2, ha="right", va="bottom", linespacing=1.3)
fig.text((L + R) / 2, TOP - ph - 0.043, "Our model: training steps (log scale)", fontsize=8.8, color=INK2, ha="center", va="center")
fig.text((L + R) / 2, B - 0.043, "STFlow public code: training epoch", fontsize=8.8, color=INK2, ha="center", va="center")
fig.savefig(f"{OUT}/fig_hest_budget_selection.png", dpi=200)
fig.savefig(f"{OUT}/fig_hest_budget_selection.pdf", dpi=300)
summary = {"protocol": "Phase 4: validation strip (one fifth of every training slide) chooses the training budget of both models; same Phikon embeddings, official HEST-Benchmark splits, 50 genes; three replicates",
           "per_organ": {t: {k: J[t][k] for k in ("ridge", "ours_default", "stflow_default", "ours_A", "stflow_A", "ours_B", "stflow_B", "ours_hindsight", "stflow_hindsight")} for t, _ in NAMES}}
summary.update(G)
summary["notes"] = ["A = test score of the four-fifths model at its validation-selected budget (primary, pre-registered)",
                    "B = the selected budget re-used with all training spots (secondary)",
                    "hindsight = best budget picked on the test patients; STFlow chooses among 100 epochs, ours among 6 step counts",
                    "the validation strip comes from the training patients; LUNG and SKCM have one training patient per split"]
json.dump(summary, open(f"{OUT}/hest_budget_selection_summary.json", "w"), indent=1)
print("primary:", pa["verdict"], pa["six_organ_mean_diff"], "| secondary:", pb["verdict"], pb["six_organ_mean_diff"])
