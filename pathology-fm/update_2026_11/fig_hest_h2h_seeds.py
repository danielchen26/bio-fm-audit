"""Figure: head-to-head on identical Phikon features and official HEST-Benchmark splits, three seeds per method.
Public STFlow code (test curve of every epoch) against our flow model and the official ridge probe."""
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
J = json.load(open(SRC)); G = J["_overall"]
NAMES = [("LUNG", "Lung"), ("PAAD", "Pancreas"), ("SKCM", "Melanoma"), ("COAD", "Colon"), ("READ", "Rectum"), ("LYMPH_IDC", "Lymph node")]
SEEDS = ("1", "2", "3")
rows = [(t, tn, s) for t, tn in NAMES for s in J[t]["splits"]]
M = G["six_organ_mean"]; OURS, FIX, REL, RIDGE = M["ours"], M["stflow_epoch100"], M["stflow_released"], M["ridge"]
BS = G["six_organ_mean_by_seed"]; sel = G["selection_effect"]["mean_over_organs"]
verdict = {"on par": "on par with", "ahead": "ahead of", "behind": "behind", "mixed": "mixed against"}[G["verdict_preregistered_rule_on_seed_means"]]
w_fix = G["ours_minus_stflow_epoch100"]["splits_ours_higher"]
curve = np.array(G["budget_sensitivity_post_hoc"]["stflow_six_organ_mean_by_epoch"]); pk = int(curve.argmax())
cseed = {k: np.mean([np.mean([s["curves"][k] for s in J[t]["splits"]], 0) for t, _ in NAMES], 0) for k in SEEDS}

fig = plt.figure(figsize=(14.6, 9.8), dpi=200)
fig.text(0.035, 0.968, f"Same features, same held-out patients, three seeds: our flow model is {verdict} the public STFlow code at default budgets",
         fontsize=12.8, weight="bold", ha="left")
fig.text(0.035, 0.946, "HEST-Benchmark, six organs, official patient-held-out splits and 50 genes; every method gets the same frozen Phikon embeddings. Score = mean Pearson r over genes on the held-out patients.\n"
         "STFlow (ICML 2025) is run from its public code with its README settings. That script scores the test patients after every epoch and keeps the best value; we show that value\n"
         f"and the value after its full 100 epochs. Six-organ mean over three seeds: ours {OURS:.3f}, STFlow {FIX:.3f} at epoch 100 and {REL:.3f} with the epoch picked on test patients ({sel:+.3f}).\n"
         f"STFlow peaks at {curve[pk]:.3f} near epoch {pk + 1}, so a shorter training budget would favour it; our model's budget was never varied. Thin lines and bars show the individual seeds.",
         fontsize=8.9, color=INK2, ha="left", va="top", linespacing=1.45)
hand = [Line2D([], [], color=BLUE, marker="D", lw=1.8, ms=7.5, mec=SURF, mew=1.2, label="Our flow model, fixed 4,000 steps"),
        Line2D([], [], color=ORANGE, marker="o", lw=1.6, ms=8, mec=SURF, mew=1.2, label="STFlow public code, epoch 100"),
        Line2D([], [], color=ORANGE, marker="o", ls="none", ms=7.5, mfc=SURF, mec=ORANGE, mew=1.8, label="STFlow public code, best epoch picked on test patients (as released)"),
        Line2D([], [], color=INK2, marker="s", lw=1.0, ls=(0, (3, 2)), ms=7, mec=SURF, mew=1.2, label="Official ridge probe")]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.032, 0.862), ncol=4, frameon=False, fontsize=8.7, labelcolor=INK, handletextpad=0.4,
           columnspacing=1.7, borderaxespad=0, handlelength=2.2)
TOP = 0.782

# ---------------------------------------------------------------- top left: six-organ mean after every epoch
ax = fig.add_axes([0.052, 0.60, 0.262, TOP - 0.60])
ep = np.arange(1, 101)
ax.axhline(RIDGE, color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
ax.axhspan(min(BS["ours"]), max(BS["ours"]), color=BLUE, alpha=0.16, lw=0, zorder=1)
ax.axhline(OURS, color=BLUE, lw=1.8, zorder=2)
ax.plot([1, 100], [REL, REL], color=ORANGE, lw=1.1, ls=(0, (1, 2)), zorder=2)
for k in SEEDS:
    ax.plot(ep, cseed[k], color=ORANGE, lw=0.8, alpha=0.5, zorder=3)
ax.plot(ep, curve, color=ORANGE, lw=1.9, zorder=4)
ax.scatter([100], [curve[-1]], s=52, color=ORANGE, edgecolor=SURF, linewidth=1.2, zorder=6, clip_on=False)
lab = sorted([(REL, "o", dict(color=SURF, edgecolor=ORANGE, linewidth=1.8)), (OURS, "D", dict(color=BLUE, edgecolor=SURF, linewidth=1.2)),
              (RIDGE, "s", dict(color=INK2, edgecolor=SURF, linewidth=1.2))], key=lambda x: -x[0])
for yv, mk, kw in lab:
    ax.scatter([104.5], [yv], s=46, marker=mk, zorder=6, clip_on=False, **kw)
    ax.text(108, yv, f"{yv:.3f}", fontsize=8.6, color=INK, va="center", ha="left")
ax.annotate(f"epoch 100: {curve[-1]:.3f}", xy=(100, curve[-1]), xytext=(97, 0.338), fontsize=8.2, color=INK, va="center", ha="right",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=2, shrinkB=5))
ax.text(pk + 1, max(cseed[k].max() for k in SEEDS) + 0.005, f"peak {curve[pk]:.3f}, epoch {pk + 1}", fontsize=8.2, color=INK, va="bottom", ha="center")
ax.set_xlim(0, 100); ax.set_ylim(0.30, 0.452); ax.set_xticks([1, 25, 50, 75, 100]); ax.set_yticks([0.30, 0.35, 0.40, 0.45])
ax.set_xlabel("STFlow training epoch", fontsize=8.8, labelpad=2); ax.set_ylabel("mean Pearson r, six organs", fontsize=8.8)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.3)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.text(0.035, TOP + 0.022, "Six-organ mean after every STFlow epoch, three seeds", fontsize=9.6, color=INK, ha="left", va="bottom")

# ---------------------------------------------------------------- bottom left: one row per official split
ax = fig.add_axes([0.118, 0.055, 0.235, 0.455])
ys, y, last = [], 0.0, None
for t, tn, s in rows:
    if last is not None and t != last:
        y -= 0.6
    ys.append(y); last = t; y -= 1
for (t, tn, s), yy in zip(rows, ys):
    a, f, r, g = s["ours"], s["stflow_epoch100"], s["stflow_released"], s["ridge"]
    yo, yf = yy + 0.17, yy - 0.17
    ax.plot([min(a, f, r, g) - 0.004, max(a, f, r, g) + 0.004], [yy, yy], color=GRID, lw=7.5, solid_capstyle="round", zorder=1)
    ax.plot([f, r], [yf, yf], color=ORANGE, lw=1.0, zorder=2)
    ax.plot([min(s["stflow_epoch100_by_seed"]), max(s["stflow_epoch100_by_seed"])], [yf, yf], color=ORANGE, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.plot([min(s["ours_by_seed"]), max(s["ours_by_seed"])], [yo, yo], color=BLUE, lw=2.6, solid_capstyle="butt", zorder=3)
    ax.scatter([g], [yy], s=30, color=INK2, marker="s", edgecolor=SURF, linewidth=1.0, zorder=4)
    ax.scatter([r], [yf], s=30, color=SURF, marker="o", edgecolor=ORANGE, linewidth=1.6, zorder=5)
    ax.scatter([f], [yf], s=40, color=ORANGE, marker="o", edgecolor=SURF, linewidth=1.0, zorder=6)
    ax.scatter([a], [yo], s=38, color=BLUE, marker="D", edgecolor=SURF, linewidth=1.0, zorder=7)
    ax.text(0.925, yy, f"{a - f:+.3f}", fontsize=8.1, color=INK if a > f else RED, va="center", ha="right")
ax.text(0.925, 0.75, "ours − STFlow\nat epoch 100", fontsize=7.5, color=INK2, va="bottom", ha="right", linespacing=1.25)
ax.set_yticks(ys); ax.set_yticklabels([f"{tn} · split {s['split'] + 1}" for _, tn, s in rows], fontsize=8.4, color=INK)
ax.set_xlim(-0.01, 0.93); ax.set_ylim(min(ys) - 0.7, 1.9); ax.set_xticks([0.1, 0.3, 0.5, 0.7])
ax.set_xlabel("mean Pearson r, held-out patients", fontsize=8.8, labelpad=2)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.3); ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
fig.text(0.035, 0.52, f"Each split, mean and range of three seeds  ·  ours higher in {w_fix} of {len(rows)}", fontsize=9.6, color=INK, ha="left", va="bottom")

# ---------------------------------------------------------------- right: test curve of every split
NC = 5; NR = int(np.ceil(len(rows) / NC)); L, R, B = 0.432, 0.992, 0.055
gw, gh = 0.034, 0.062; pw = (R - L - gw * (NC - 1)) / NC; ph = (TOP - B - gh * (NR - 1)) / NR
fig.text(L - 0.022, TOP + 0.022, "Test-patient score after every epoch of the public STFlow script, split by split  ·  horizontal lines: our model and the ridge probe",
         fontsize=9.6, color=INK, ha="left", va="bottom")
for k, (t, tn, s) in enumerate(rows):
    r_, c_ = divmod(k, NC)
    a2 = fig.add_axes([L + c_ * (pw + gw), TOP - (r_ + 1) * ph - r_ * gh, pw, ph])
    C = np.array([s["curves"][q] for q in SEEDS]); c = C.mean(0); e2 = np.arange(1, 101)
    a2.axhline(s["ridge"], color=INK2, lw=1.0, ls=(0, (3, 2)), zorder=1)
    a2.axhspan(min(s["ours_by_seed"]), max(s["ours_by_seed"]), color=BLUE, alpha=0.16, lw=0, zorder=1)
    a2.axhline(s["ours"], color=BLUE, lw=1.8, zorder=2)
    for j in range(3):
        a2.plot(e2, C[j], color=ORANGE, lw=0.7, alpha=0.45, zorder=3)
    a2.plot(e2, c, color=ORANGE, lw=1.6, zorder=4)
    for j in range(3):
        b = s["stflow_best_epoch_by_seed"][j]
        a2.scatter([b], [C[j][b - 1]], s=26, color=SURF, edgecolor=ORANGE, linewidth=1.5, zorder=5)
    a2.scatter([100], [c[-1]], s=44, color=ORANGE, edgecolor=SURF, linewidth=1.2, zorder=6, clip_on=False)
    hi = max(C.max(), max(s["ours_by_seed"]), s["ridge"]); lo = min(C[:, 3:].min(), min(s["ours_by_seed"]), s["ridge"])
    pad = max(0.02, 0.16 * (hi - lo)); a2.set_ylim(lo - pad, hi + 2.1 * pad); a2.set_xlim(0, 104)
    a2.set_xticks([1, 50, 100]); a2.tick_params(length=2, width=0.5, labelsize=7.6)
    a2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); a2.yaxis.grid(True, color=GRID, lw=0.5); a2.set_axisbelow(True)
    for sp in ("top", "right"):
        a2.spines[sp].set_visible(False)
    a2.text(0.0, 1.03, f"{tn} · split {s['split'] + 1}", transform=a2.transAxes, fontsize=8.3, color=INK, ha="left", va="bottom")
    a2.text(0.985, 0.975, f"best {s['stflow_released']:.2f} → end {s['stflow_epoch100']:.2f}", transform=a2.transAxes, fontsize=7.4, color=INK2, ha="right", va="top")
    if r_ == NR - 1:
        a2.set_xlabel("epoch", fontsize=8.4, labelpad=2)
    if c_ == 0:
        a2.set_ylabel("test Pearson r", fontsize=8.4)
fig.savefig(f"{OUT}/fig_hest_head_to_head.png", dpi=200)
fig.savefig(f"{OUT}/fig_hest_head_to_head.pdf", dpi=300)
summary = {"protocol": "same frozen Phikon embeddings, official HEST-Benchmark splits and 50 genes, mean Pearson over genes on pooled test spots; three seeds per method",
           "per_organ_seed_mean": {t: {k: J[t][k] for k in ("ridge", "ours", "stflow_epoch100", "stflow_released")} for t, _ in NAMES},
           "per_organ_by_seed": {t: {k: J[t][k + "_by_seed"] for k in ("ours", "stflow_epoch100", "stflow_released")} for t, _ in NAMES}}
summary.update({k: v for k, v in G.items()})
summary["budget_sensitivity_post_hoc"] = {k: v for k, v in G["budget_sensitivity_post_hoc"].items() if k != "stflow_six_organ_mean_by_epoch"}
summary["caveats"] = ["public STFlow code needed a one-line repair to start; its authors describe it as a demo",
                      "STFlow's published numbers use UNI features and are not reproduced here",
                      "our model's training budget was never varied; STFlow's test curve peaks before its default 100 epochs"]
json.dump(summary, open(f"{OUT}/hest_head_to_head_summary.json", "w"), indent=1)
print("verdict:", verdict, "| ours", OURS, "stflow e100", FIX, "released", REL, "ridge", RIDGE)
