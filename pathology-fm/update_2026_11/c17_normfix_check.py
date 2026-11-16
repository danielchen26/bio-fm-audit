"""How much did the wrong input normalisation cost the ImageNet ViT-S baseline? Same splits, same probe."""
import sys, json, glob
sys.dont_write_bytecode = True
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJ + "/src")
import numpy as np
from audit import logistic_probe
ROOT = os.environ.get("C17_TILESET", f"{PROJ}/data/camelyon17_tileset")   # tiles/<slide>.npz plus the two reference embeddings
GRADES = ("negative", "itc", "micro", "macro")
y, patient, grade, centre, pats = [], [], [], [], []
for f in sorted(glob.glob(f"{ROOT}/tiles/*.npz")):
    with np.load(f, allow_pickle=False) as z:
        lab, pid = z["labels"], str(z["patient_id"])
        if pid not in pats:
            pats.append(pid)
        n = len(lab)
        y.append(lab); patient.append(np.full(n, pats.index(pid))); grade.append(np.full(n, GRADES.index(str(z["grade"]))))
        centre.append(np.full(n, int(z["centre"])))
y, patient, grade, centre = map(np.concatenate, (y, patient, grade, centre))
keep = np.isin(grade, [0, 2, 3])
E = {"legacy": np.load("c17_emb_imagenet.npy")[keep], "fixed": np.load("c17_emb_in_sup_vits.npy")[keep]}
y, patient, centre = y[keep], patient[keep], centre[keep]


def group_split(key, seed, frac=0.25):
    g = np.unique(key); pick = np.random.default_rng(seed).permutation(g)
    m = np.isin(key, pick[:max(1, int(len(g) * frac))])
    return np.where(~m)[0], np.where(m)[0]

stored = json.load(open("c17_labeleff3.json"))
out = {}
for k in ("legacy", "fixed"):
    allp = []
    for s in range(5):
        tr, te = group_split(patient, s)
        allp.append(float(logistic_probe(E[k][tr], y[tr], E[k][te], y[te], 2)["auc"]))
    loco = [float(logistic_probe(E[k][centre != c], y[centre != c], E[k][centre == c], y[centre == c], 2)["auc"]) for c in range(5)]
    out[k] = {"all_patients_per_split": [round(a, 3) for a in allp], "all_mean": round(float(np.mean(allp)), 4),
              "loco": [round(a, 3) for a in loco], "loco_mean": round(float(np.mean(loco)), 4)}
    print(k, out[k], flush=True)
    if k == "legacy":
        assert np.allclose(allp, stored["imagenet"]["label_eff"]["all"], atol=1e-6) and np.allclose(loco, stored["imagenet"]["loco"], atol=1e-6)
        print("sanity: legacy numbers reproduce the stored ImageNet row exactly", flush=True)
print("phikon stored:", [round(a, 3) for a in stored["phikon"]["label_eff"]["all"]], [round(a, 3) for a in stored["phikon"]["loco"]])
json.dump(out, open("c17_normfix_check.json", "w"), indent=1)
