"""Figure: H&E -> spatial gene expression, 5-fold spatial-block cross-validation (out-of-fold)."""
import sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

D, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
COL = {"phikon": "#2a78d6", "random": "#eb6834", "knn": "#1baf7a"}
NAME = {"phikon": "Phikon features + flow head", "random": "Random conv features + flow head",
        "knn": "Nearest neighbours in Phikon space"}
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
                                                 "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"])
SHOW = ["IGKC", "FDCSP", "CCL21"]
CELL = {"IGKC": "plasma cells", "FDCSP": "follicular dendritic cells", "CCL21": "T-zone stroma"}                 # fixed before the cross-validation was run
Image.MAX_IMAGE_PIXELS = None
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

folds = [dict(np.load(f, allow_pickle=False)) for f in sorted(glob.glob(f"{D}/st_cv_fold*.npz"))]
assert len(folds) == 5, len(folds)
genes = folds[0]["genes"].astype(str).tolist()
cat = lambda k: np.concatenate([f[k] for f in folds])
order = np.argsort(cat("te_idx"))
assert len(np.unique(cat("te_idx"))) == len(order), "a spot was predicted twice"
fold_id = np.concatenate([np.full(len(f["te_idx"]), i) for i, f in enumerate(folds)])[order]
coords, truth = cat("coords")[order], cat("truth_log")[order]
pred = {k: cat(f"pred_{k}_log")[order] for k in COL}
mean_base = np.concatenate([np.repeat(f["train_mean_log"][None], len(f["te_idx"]), 0) for f in folds])[order]


def pearson(p, t):
    p, t = p - p.mean(0, keepdims=True), t - t.mean(0, keepdims=True)
    return (p * t).sum(0) / (np.sqrt((p ** 2).sum(0) * (t ** 2).sum(0)) + 1e-12)

r = {k: pearson(pred[k], truth) for k in COL}
r_fold = {k: np.stack([pearson(pred[k][fold_id == i], truth[fold_id == i]) for i in range(5)]) for k in COL}
crps = {k: cat(f"crps_{k}")[order] for k in ("phikon", "random", "clim")}
summary = {"n_spots": int(len(order)), "n_genes": len(genes), "spots_per_fold": [int((fold_id == i).sum()) for i in range(5)],
           "crps_mean": {k: float(v.mean()) for k, v in crps.items()},
           "crps_rel_vs_climatology": {k: float(crps[k].mean() / crps["clim"].mean() - 1) for k in ("phikon", "random")},
           "crps_rel_vs_climatology_per_fold": {k: [float(crps[k][fold_id == i].mean() / crps["clim"][fold_id == i].mean() - 1)
                                                    for i in range(5)] for k in ("phikon", "random")},
           "mae_log": {**{k: float(np.abs(pred[k] - truth).mean()) for k in COL},
                       "train_mean": float(np.abs(mean_base - truth).mean())},
           "gene_r": {k: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "n_gt_0.5": int((v > 0.5).sum()),
                          "n_gt_0.3": int((v > 0.3).sum()),
                          "mean_of_per_fold_means": float(np.nanmean(r_fold[k])),
                          "per_fold_mean": [float(np.nanmean(x)) for x in r_fold[k]]} for k, v in r.items()},
           "phikon_minus_random_gene_r": {"mean": float(np.mean(r["phikon"] - r["random"])),
                                          "n_genes_phikon_higher": int((r["phikon"] > r["random"]).sum())},
           "shown_genes": {g: {"r_pooled": {k: float(r[k][genes.index(g)]) for k in COL},
                               "r_per_fold_phikon": [float(x) for x in r_fold["phikon"][:, genes.index(g)]]} for g in SHOW},
           "top20_phikon": [(genes[i], round(float(r["phikon"][i]), 3), round(float(r["random"][i]), 3), round(float(r["knn"][i]), 3),
                             round(float(r_fold["phikon"][:, i].min()), 3)) for i in np.argsort(-r["phikon"])[:20]]}
json.dump(summary, open(f"{OUT}/st_cv_summary.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("spots_per_fold", "crps_rel_vs_climatology", "crps_rel_vs_climatology_per_fold",
                                         "mae_log", "gene_r", "phikon_minus_random_gene_r", "shown_genes")}, indent=1))
print("top20 (gene, r phikon, r random, r knn, min per-fold r phikon):")
for row in summary["top20_phikon"]:
    print("  ", row)

# ------------------------------------------------------------------ figure
sf = json.load(open(f"{D}/st_scalefactors.json"))["tissue_hires_scalef"]
he = np.asarray(Image.open(f"{D}/st_hires_image.png").convert("RGB"))
pad = 0.03 * (coords[:, 0].max() - coords[:, 0].min())
xl = (coords[:, 0].min() - pad, coords[:, 0].max() + pad)
yl = (coords[:, 1].max() + pad, coords[:, 1].min() - pad)
d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(d, np.inf)
pitch = float(np.median(d.min(1))); del d

fig = plt.figure(figsize=(12.4, 7.2), dpi=200)
gs = fig.add_gridspec(2, 4, left=0.035, right=0.985, top=0.845, bottom=0.085, wspace=0.07, hspace=0.16,
                      width_ratios=[1.08, 1, 1, 1])
fig.text(0.035, 0.955, "Predicting spatial gene expression from H&E alone", fontsize=14, weight="bold", ha="left")
fig.text(0.035, 0.935, "Human lymph node, 10x Visium, 4,035 spots. Each spot is predicted by a model trained with no expression from its tissue block "
         "(5-fold spatial CV); the image is the only input.\nTop row: measured. Bottom row: predicted from the image. "
         "Measured and predicted maps of a gene share one colour scale.", fontsize=9, color=INK2, ha="left", va="top", linespacing=1.45)


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)

ax = fig.add_subplot(gs[0, 0])
ax.imshow(he, extent=[0, he.shape[1] / sf, he.shape[0] / sf, 0], interpolation="lanczos")
clean(ax); ax.set_title("H&E image (model input)", fontsize=10, loc="left", color=INK, pad=5)

fig.canvas.draw()
for j, g in enumerate(SHOW):
    gi = genes.index(g)
    t, p = truth[:, gi], pred["phikon"][:, gi]
    vmin, vmax = np.percentile(t, [2, 98])
    for row, (vals, lab) in enumerate(((t, "measured"), (p, "predicted from H&E, held-out"))):
        ax = fig.add_subplot(gs[row, j + 1]); clean(ax)
        fig.canvas.draw()
        pts_per_data = ax.get_window_extent().width * 72 / fig.dpi / (xl[1] - xl[0])
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, s=(0.98 * pitch * pts_per_data) ** 2, cmap=SEQ, vmin=vmin, vmax=vmax,
                        marker="h", linewidths=0, rasterized=True)
        tt = ax.set_title(f"{g}  ·  measured" if row == 0 else f"{g}  ·  predicted from H&E", fontsize=10, loc="left",
                          color=INK, pad=5)
        if row == 0:
            fig.canvas.draw(); tb = tt.get_window_extent()
            ax.annotate(CELL[g], xy=(tb.x1, tb.y0), xycoords="figure pixels", xytext=(7, 1.2), textcoords="offset points",
                        fontsize=8, color=INK2, va="bottom", ha="left")
        if row == 1:
            ax.text(0.985, 0.03, f"r = {r['phikon'][gi]:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                    weight="bold", color=INK, bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=0.9))
            bb = ax.get_position()
            cax = fig.add_axes([bb.x0 + 0.02 * bb.width, bb.y0 - 0.032, 0.5 * bb.width, 0.012])
            cb = fig.colorbar(sc, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
            cb.ax.tick_params(labelsize=7.5, length=2, width=0.5, color=INK2); cb.set_ticks(np.round(np.linspace(vmin, vmax, 3), 1))
            fig.text(bb.x0 + 0.55 * bb.width, bb.y0 - 0.031, "log1p(CP10K)", fontsize=7.5, color=INK2, va="bottom", ha="left")

ax = fig.add_subplot(gs[1, 0])
bb = ax.get_position(); ax.set_position([0.075, bb.y0 + 0.012, bb.x1 - 0.075 - 0.012, bb.height - 0.035])
x = np.arange(1, len(genes) + 1)
ax.axhline(0.5, color=GRID, lw=0.8, zorder=0); ax.axhline(0, color=GRID, lw=0.8, zorder=0)
for k in ("knn", "random", "phikon"):
    ax.plot(x, np.sort(r[k])[::-1], color=COL[k], lw=2, solid_capstyle="round", label=NAME[k])
ax.set_xlim(0, len(genes) + 2); ax.set_ylim(-0.03, 1.2); ax.set_yticks([0, 0.25, 0.5, 0.75])
ax.set_xlabel(f"genes, ranked within each method (n = {len(genes)})", fontsize=8.5)
ax.set_ylabel("Pearson r, measured vs predicted", fontsize=8.5)
ax.tick_params(labelsize=8, length=2.5, width=0.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("All 256 genes, every spot held out", fontsize=10, loc="left", color=INK, pad=5)
SH = {"phikon": "Phikon + flow head", "random": "Random conv + flow head", "knn": "Nearest neighbours"}
lines = [f"{SH[k]}  ·  {int((r[k] > 0.5).sum())} genes r > 0.5" for k in ("phikon", "random", "knn")]
leg = ax.legend(handles=[plt.Line2D([], [], color=COL[k], lw=2) for k in ("phikon", "random", "knn")], labels=lines, loc="upper left",
                fontsize=7.8, frameon=False, handlelength=1.3, borderaxespad=0.0, labelcolor=INK, bbox_to_anchor=(0.0, 0.995))
ax.text(len(genes) + 1, 0.5, "r = 0.5", fontsize=7.5, color=INK2, va="bottom", ha="right")
fig.savefig(f"{OUT}/fig_spatial_expression_from_HE.png", dpi=200)
fig.savefig(f"{OUT}/fig_spatial_expression_from_HE.pdf", dpi=300)
print("saved figure")
