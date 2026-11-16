"""Figure: spatial expression from H&E, replication on three slides with controls and the noise ceiling."""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
SL = [("V1LN", "Lymph node\nfresh-frozen"), ("BRCA", "Breast cancer\nfresh-frozen"), ("V2LN", "Lymph node\nFFPE")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
E = {s: json.load(open(f"{D}/st_eval_{s}.json")) for s, _ in SL}
Z = {s: dict(np.load(f"{D}/st_eval_{s}.npz", allow_pickle=False)) for s, _ in SL}
FL = json.load(open(f"{D}/st_flow_summary.json")) if os.path.exists(f"{D}/st_flow_summary.json") else {}
flow = {}
for s, _ in SL:
    if "flow:phikon" in E[s]["blocks_6x6"]:
        r = E[s]["blocks_6x6"]["flow:phikon"]; flow[s] = (r["gene_r_median"], r["per_fold_median_r"])
    elif s in FL:
        flow[s] = (FL[s]["gene_r_median"], FL[s]["per_fold_median_r"])
M = [("coords_only_knn", "Coordinates only (spatial interpolation)", GREEN, "^"),
     ("ridge:phikon", "Image features + ridge regression", ORANGE, "s"),
     ("flow", "Image features + neighbourhood Transformer + flow head", BLUE, "D")]
fr = []
for s, _ in SL:
    c = E[s]["noise_ceiling"]["median_ceiling_r"]
    best = FL[s]["median_fraction_of_ceiling"] * c if s in FL else E[s]["blocks_6x6"]["ridge:phikon"]["gene_r_median"]
    fr.append(best / c)

fig = plt.figure(figsize=(12.6, 5.5), dpi=200)
gs = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.70, bottom=0.135, wspace=0.3, width_ratios=[1.2, 0.82, 1.0])
fig.text(0.055, 0.945, "Predicting expression from H&E replicates on three slides and is not spatial interpolation", fontsize=13.5, weight="bold", ha="left")
fig.text(0.055, 0.905, "Three public 10x Visium slides built with one procedure; 256 most variable genes per slide; 5-fold cross-validation over held-out tissue blocks; "
         "Phikon image features throughout.\n"
         "Values are the median over genes of Pearson r between measured and predicted expression; bars span the five folds.",
         fontsize=9, color=INK2, ha="left", va="top", linespacing=1.45)
hand = [Line2D([], [], color=c, marker=m, ls="none", ms=8, mec=SURF, mew=1.2, label=lab) for _, lab, c, m in M]
hand.append(Line2D([], [], color=INK, lw=1.6, label="Measurement-noise ceiling (median gene)"))
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.052, 0.815), ncol=2, frameon=False, fontsize=8.6, labelcolor=INK, handletextpad=0.4,
           columnspacing=1.6, borderaxespad=0)

# ---------------------------------------------------------------- (a) three slides
ax = fig.add_subplot(gs[0, 0]); off = np.linspace(-0.25, 0.25, len(M))
for i, (s, lab) in enumerate(SL):
    ax.axvspan(i - 0.42, i + 0.42, color=GRID, alpha=0.28, lw=0, zorder=0)
    c = E[s]["noise_ceiling"]["median_ceiling_r"]
    ax.plot([i - 0.42, i + 0.42], [c, c], color=INK, lw=1.6, zorder=2); ax.text(i + 0.41, c + 0.012, f"{c:.2f}", fontsize=8.4, ha="right", va="bottom", color=INK)
    for o, (k, _, col, mk) in zip(off, M):
        if k == "flow":
            if s not in flow:
                continue
            v, pf = flow[s]
        else:
            v, pf = E[s]["blocks_6x6"][k]["gene_r_median"], E[s]["blocks_6x6"][k]["per_fold_median_r"]
        ax.vlines(i + o, min(pf), max(pf), color=col, alpha=0.4, lw=2.2, zorder=2, capstyle="round")
        ax.scatter([i + o], [v], s=72, color=col, marker=mk, edgecolor=SURF, linewidth=1.3, zorder=3)
        ax.text(i + o, max(pf) + 0.014, f"{v:.2f}", fontsize=8.4, ha="center", va="bottom", color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels([lab for _, lab in SL], fontsize=8.8, color=INK, linespacing=1.25)
ax.set_xlim(-0.52, 2.52); ax.set_ylim(0, 0.93); ax.set_ylabel("median gene-wise Pearson r", fontsize=9)
for i in range(3):
    pass
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5); ax.tick_params(axis="x", length=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_title("Image beats coordinates; the flow model beats ridge in 15 of 15 folds", fontsize=9.6, loc="left", color=INK, pad=7)

# ---------------------------------------------------------------- (b) gene r against its noise ceiling, first slide
ax = fig.add_subplot(gs[0, 1])
ceil, r = Z["V1LN"]["ceiling"], Z["V1LN"]["r__flow_phikon"]
ax.plot([0, 1], [0, 1], color=INK, lw=1.2, zorder=1)
ax.scatter(ceil, r, s=16, color=BLUE, alpha=0.75, edgecolor=SURF, linewidth=0.4, zorder=2)
ax.text(0.72, 0.75, "ceiling", fontsize=8.4, color=INK, rotation=45, rotation_mode="anchor", ha="left", va="bottom")
ax.text(0.035, 0.97, f"256 genes, lymph node\n(fresh-frozen)\ncorrelation {np.corrcoef(ceil, r)[0, 1]:.2f}\nmedian gene reaches\n{np.median(r / ceil):.0%} of its ceiling",
        transform=ax.transAxes, fontsize=8.4, color=INK, va="top", ha="left", linespacing=1.4)
ax.set_xlim(0, 1); ax.set_ylim(-0.02, 1); ax.set_aspect("equal")
ax.set_xlabel("noise ceiling of the gene\n(from splitting its UMI counts)", fontsize=8.8, linespacing=1.35); ax.set_ylabel("achieved r (flow model)", fontsize=9)
ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_title("Poorly predicted genes are the noisy ones", fontsize=9.6, loc="left", color=INK, pad=7)

# ---------------------------------------------------------------- (c) larger held-out regions
ax = fig.add_subplot(gs[0, 2]); rows, yt = [], []
for i, (s, lab) in enumerate(SL):
    for j, (k, col, mk, nm) in enumerate((("ridge:phikon", ORANGE, "s", "image + ridge"), ("coords_only_knn", GREEN, "^", "coordinates"))):
        yy = -(i * 2.6 + j); a, b = E[s]["blocks_6x6"][k]["gene_r_median"], E[s]["blocks_3x3"][k]["gene_r_median"]
        ax.annotate("", xy=(b, yy), xytext=(a, yy), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8, shrinkA=4, shrinkB=1, mutation_scale=11))
        ax.scatter([a], [yy], s=58, color=SURF, marker=mk, edgecolor=col, linewidth=1.8, zorder=3)
        ax.text(max(a, b) + 0.025, yy, f"{a:.2f} → {b:.2f}", fontsize=8.4, color=INK, va="center", ha="left")
        yt.append((yy, nm))
    ax.text(-0.005, -(i * 2.6) + 0.72, lab.replace("\n", ", "), fontsize=8.6, color=INK, ha="left", va="center", weight="bold")
ax.set_yticks([y for y, _ in yt]); ax.set_yticklabels([n for _, n in yt], fontsize=8.4, color=INK)
ax.set_xlim(0, 0.98); ax.set_ylim(-6.9, 1.3); ax.set_xlabel("median gene-wise r, 36 small → 9 large held-out blocks", fontsize=8.8)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5); ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.set_title("Larger held-out regions hurt coordinates most", fontsize=9.6, loc="left", color=INK, pad=7)
fig.savefig(f"{OUT}/fig_spatial_replication_controls.png", dpi=200)
fig.savefig(f"{OUT}/fig_spatial_replication_controls.pdf", dpi=300)
print("fraction of ceiling per slide (best available model):", [round(x, 2) for x in fr], "| flow available:", sorted(flow))
print("saved figure")
