"""E5: tile montage for the slide where Phikon fails under leave-one-hospital-out (patient_052_node_1)."""
import glob, json
from c17_common import *
from PIL import Image, ImageDraw
M = load_meta(); y, slide, sids = M["y"], M["slide"], M["sids"]
SID = "patient_052_node_1"; si = sids.index(SID); m = np.where(slide == si)[0]
sc = {e: np.load(f"{HERE}/c17_grid_scores_{e}.npz")["loco_M0"][m] for e in ("phikon", "path_dino_vits_lunit", "in_dino_vitb")}
with np.load(sorted(glob.glob(f"{ROOT}/tiles/*.npz"))[si], allow_pickle=False) as z:
    P, lab, cover = z["patches"], z["labels"], z["cover"] if "cover" in z.files else None
assert (lab == y[m]).all()
yy = y[m]; ph = sc["phikon"]
rows = [("tumour tiles Phikon scores LOWEST (missed)", np.where(yy == 1)[0][np.argsort(ph[yy == 1])[:14]]),
        ("tumour tiles Phikon scores HIGHEST", np.where(yy == 1)[0][np.argsort(-ph[yy == 1])[:14]]),
        ("normal tiles Phikon scores HIGHEST (false alarms)", np.where(yy == 0)[0][np.argsort(-ph[yy == 0])[:14]]),
        ("normal tiles Phikon scores LOWEST", np.where(yy == 0)[0][np.argsort(ph[yy == 0])[:14]])]
T = 128; W = 14 * T; img = Image.new("RGB", (W, len(rows) * (T + 22)), (252, 252, 251)); d = ImageDraw.Draw(img)
for r, (name, idx) in enumerate(rows):
    d.text((4, r * (T + 22) + 4), f"{name}   Phikon P(tumour): {ph[idx].min():.2f}-{ph[idx].max():.2f}   Lunit: {sc['path_dino_vits_lunit'][idx].mean():.2f} mean", fill=(11, 11, 11))
    for c, i in enumerate(idx):
        img.paste(Image.fromarray(P[i]).resize((T, T), Image.Resampling.LANCZOS), (c * T, r * (T + 22) + 20))
img.save("c17_failure_patient_052_node_1.png")
q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]
out = {"slide": SID, "tiles": int(len(m)), "tumour": int(yy.sum()),
       "P_tumour_on_tumour_tiles_p10_p50_p90": {e: q(s[yy == 1]) for e, s in sc.items()},
       "P_tumour_on_normal_tiles_p10_p50_p90": {e: q(s[yy == 0]) for e, s in sc.items()},
       "tumour_cover_p10_p50_p90": q(cover[yy == 1]) if cover is not None else None,
       "frac_tumour_tiles_scored_below_0.5": {e: round(float((s[yy == 1] < 0.5).mean()), 3) for e, s in sc.items()},
       "frac_normal_tiles_scored_above_0.5": {e: round(float((s[yy == 0] >= 0.5).mean()), 3) for e, s in sc.items()}}
json.dump(out, open("c17_failure_patient_052_node_1.json", "w"), indent=1); print(json.dumps(out))
