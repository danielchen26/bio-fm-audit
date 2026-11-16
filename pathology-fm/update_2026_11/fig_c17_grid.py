"""Figure: CAMELYON17, matched encoder grid (pathology vs ImageNet pretraining at fixed architecture)."""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

D, OUT = sys.argv[1], sys.argv[2]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"
# colour = pretraining data, marker = model
ENC = {"path_dino_vits_lunit": ("Lunit · pathology slides · ViT-S, DINO", BLUE, "o", "-"),
       "in_dino_vits": ("ImageNet · ViT-S, DINO", ORANGE, "o", "-"),
       "path_dino_vitb_kaiko": ("Kaiko · pathology slides · ViT-B, DINO", BLUE, "s", "-"),
       "in_dino_vitb": ("ImageNet · ViT-B, DINO", ORANGE, "s", "-"),
       "phikon": ("Phikon · pathology slides · ViT-B, iBOT", BLUE, "D", (0, (4, 2)))}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
S = json.load(open(f"{D}/c17_grid_summary.json"))
have = [e for e in ENC if e in S["encoders"] and os.path.exists(f"{D}/c17_grid_{e}.json")]
J = {e: json.load(open(f"{D}/c17_grid_{e}.json")) for e in have}
NS = ["1", "2", "4", "8", "16", "all"]
con = S["contrasts"]
c1, c2 = con.get("C1 lunit - in_dino_vits"), con.get("C2 kaiko - in_dino_vitb")
ok = lambda c: bool(c and c["rule_E2_ci_excludes_0_positive"] and c["rule_E1_budgets_2_and_16_at_least_4_of_5"])
if ok(c1) and ok(c2):
    title = "With architecture and training method held fixed, pathology pretraining needs fewer labelled patients"
elif ok(c1) and c2 and c2["rule_E1_budgets_2_and_16_at_least_4_of_5"] and c2["E2_diff"] > 0:
    title = "At matched architecture, pathology pretraining needs fewer labelled patients: resolved at ViT-S, borderline at ViT-B"
elif ok(c1):
    title = "Pathology pretraining needs fewer labelled patients at ViT-S; at ViT-B the gain is not resolved"
else:
    title = "Pathology vs ImageNet pretraining at matched architecture"

fig = plt.figure(figsize=(12.4, 8.9), dpi=200)
gs = fig.add_gridspec(2, 2, left=0.062, right=0.985, top=0.79, bottom=0.07, wspace=0.17, hspace=0.36)
fig.text(0.062, 0.965, title, fontsize=12.2, weight="bold", ha="left")
fig.text(0.062, 0.94, "CAMELYON17 lymph-node metastasis, tumour vs normal tiles: 49,532 tiles, 34 patients, 5 hospitals. Frozen features + linear probe; "
         "test patients held out.\nColour = pretraining data (blue: pathology slides, orange: ImageNet), marker = model. Each encoder uses its own input normalisation.",
         fontsize=9, color=INK2, ha="left", va="top", linespacing=1.45)
hand = [Line2D([], [], color=ENC[e][1], marker=ENC[e][2], ls=ENC[e][3], lw=2, ms=7, mec=SURF, mew=1.3, label=ENC[e][0]) for e in ENC if e in have]
fig.legend(handles=hand, loc="upper left", bbox_to_anchor=(0.058, 0.893), ncol=3, frameon=False, fontsize=8.6, labelcolor=INK, handlelength=2.4,
           columnspacing=1.6, borderaxespad=0)


def label_eff(ax, encs, ttl, dodge):
    x = np.arange(len(NS)); ends = []
    for e in encs:
        if e not in have:
            continue
        v = np.array([J[e]["E1_label_eff"][n] for n in NS]); m = v.mean(1); xx = x + dodge[e]
        ax.vlines(xx, v.min(1), v.max(1), color=ENC[e][1], alpha=0.35, lw=2.2, zorder=2, capstyle="round")
        ax.plot(xx, m, color=ENC[e][1], lw=2, marker=ENC[e][2], ls=ENC[e][3], ms=7, mec=SURF, mew=1.4, zorder=3)
        ends.append((m[0], m[-1]))
    for side, xpos, ha in ((0, x[0] - 0.22, "right"), (1, x[-1] + 0.22, "left")):      # direct labels, nudged apart
        ys = sorted((e[side] for e in ends), reverse=True); placed = []
        for yv in ys:
            yy = yv if not placed else min(yv, placed[-1] - 0.026)
            placed.append(yy); ax.text(xpos, yy, f"{yv:.2f}", fontsize=8.6, color=INK, va="center", ha=ha)
    ax.set_xticks(x); ax.set_xticklabels(["1", "2", "4", "8", "16", "all\n(26)"], fontsize=8.5)
    ax.set_xlim(-0.75, len(NS) - 0.3); ax.set_ylim(0.5, 1.008)
    ax.set_xlabel("patients whose labelled tiles train the probe", fontsize=9)
    ax.set_ylabel("tumour-tile AUC on held-out patients", fontsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(ttl, fontsize=9.6, loc="left", color=INK, pad=7)

label_eff(fig.add_subplot(gs[0, 0]), ["in_dino_vits", "path_dino_vits_lunit"],
          "ViT-S/16, DINO: same architecture and method  ·  mean and min–max of 5 splits", {"in_dino_vits": 0.05, "path_dino_vits_lunit": -0.05})
label_eff(fig.add_subplot(gs[0, 1]), ["in_dino_vitb", "path_dino_vitb_kaiko", "phikon"], "ViT-B/16  ·  mean and min–max of 5 splits",
          {"in_dino_vitb": 0.08, "path_dino_vitb_kaiko": -0.08, "phikon": 0.0})

# ---------------------------------------------------------------- paired differences, patient bootstrap
ax = fig.add_subplot(gs[1, 0])
bb = ax.get_position(); ax.set_position([bb.x0 + 0.105, bb.y0, bb.width - 0.105, bb.height])
rows = [("C1 lunit - in_dino_vits", "Lunit − ImageNet\nViT-S, DINO vs DINO"), ("C2 kaiko - in_dino_vitb", "Kaiko − ImageNet\nViT-B, DINO vs DINO"),
        ("C3 phikon - in_dino_vitb", "Phikon − ImageNet\nViT-B, iBOT vs DINO"), ("C4 phikon - in_sup_vitb", "Phikon − ImageNet\nViT-B, iBOT vs supervised")]
rows = [(k, lab) for k, lab in rows if k in con]
ax.axvline(0, color=INK2, lw=0.8, zorder=1)
for i, (k, lab) in enumerate(rows):
    c = con[k]; yy = len(rows) - 1 - i; lo, hi = c["E2_ci95"]; solid = lo > 0 or hi < 0
    ax.plot([lo, hi], [yy, yy], color=BLUE, lw=2.4, solid_capstyle="round", zorder=2)
    ax.plot([c["E2_diff"]], [yy], marker="o", ms=9, color=BLUE if solid else SURF, mec=BLUE, mew=2, zorder=3)
    ax.text(hi + 0.004, yy, f"{c['E2_diff']:+.3f}  [{lo:+.3f}, {hi:+.3f}]", fontsize=8.6, color=INK, va="center", ha="left")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([lab for _, lab in rows][::-1], fontsize=8.6, color=INK, linespacing=1.25)
xs = [v for k, _ in rows for v in con[k]["E2_ci95"]]
ax.set_xlim(min(-0.045, min(xs) - 0.012), max(xs) + 0.105); ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel("AUC difference, pathology − ImageNet\nevery patient tested once; 95% interval from resampling the 34 patients", fontsize=8.6, linespacing=1.4)
ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Paired difference with patient-level uncertainty  ·  filled = interval excludes 0", fontsize=9.6, loc="left", color=INK, pad=7, x=-0.3)

# ---------------------------------------------------------------- leave one hospital out
ax = fig.add_subplot(gs[1, 1])
order = [e for e in ("path_dino_vits_lunit", "in_dino_vits", "path_dino_vitb_kaiko", "in_dino_vitb", "phikon") if e in have]
off = np.linspace(-0.26, 0.26, len(order))
for c in range(5):
    ax.axvspan(c - 0.4, c + 0.4, color=GRID, alpha=0.28, lw=0, zorder=0)
for o, e in zip(off, order):
    v = np.array(S["table"][e]["E3_per_hospital"])
    ax.scatter(np.arange(5) + o, v, s=62, color=ENC[e][1], marker=ENC[e][2], edgecolor=SURF, linewidth=1.3, zorder=3)
ph = S["table"]["phikon"]["E3_per_hospital"]; w = int(np.argmin(ph))
ax.annotate(f"Phikon {ph[w]:.2f}:\none slide fails", xy=(w + off[order.index("phikon")], ph[w] - 0.004), xytext=(w + 0.62, ph[w] - 0.035), fontsize=8.2,
            color=INK, ha="left", va="center", linespacing=1.25, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=1, shrinkB=4))
ax.set_xticks(range(5)); ax.set_xlim(-0.55, 4.55); ax.set_ylim(0.84, 1.004)
ax.set_xlabel("held-out hospital (CAMELYON17 centre ID)", fontsize=9); ax.set_ylabel("tumour-tile AUC on the held-out hospital", fontsize=9)
ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True); ax.tick_params(length=2.5, width=0.5, labelsize=8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("Unseen hospital: train on 4, test on the 5th", fontsize=9.6, loc="left", color=INK, pad=7)
fig.savefig(f"{OUT}/fig_camelyon17_matched_pretraining.png", dpi=200)
fig.savefig(f"{OUT}/fig_camelyon17_matched_pretraining.pdf", dpi=300)
print("title:", title); print("saved figure")
