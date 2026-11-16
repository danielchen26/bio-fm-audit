"""Phase 4: split every sample's Phikon h5 into a training part (four fifths) and a validation strip (one fifth), per replicate.
Layout: embed_val/{task}/rep{r}/{train|val}/{sample}.h5 with the keys the public STFlow loader reads."""
import os, sys, glob, json, numpy as np, h5py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hest"))
from val_strip import val_strip
info = {}
for task in sys.argv[1:]:
    for f in sorted(glob.glob(f"embed/{task}/phikon/fp32/*.h5")):
        sid = os.path.basename(f)[:-3]
        with h5py.File(f, "r") as h:
            emb, bc, co = h["embeddings"][:], h["barcodes"][:], h["coords"][:]
        for r in (0, 1, 2):
            va = val_strip(co, r); assert 50 <= va.sum() <= 0.3 * len(va), (task, sid, r, int(va.sum()), len(va))
            for part, m in (("train", ~va), ("val", va)):
                out = f"embed_val/{task}/rep{r}/{part}/{sid}.h5"; os.makedirs(os.path.dirname(out), exist_ok=True)
                with h5py.File(out, "w") as h:
                    h.create_dataset("embeddings", data=emb[m]); h.create_dataset("barcodes", data=bc[m]); h.create_dataset("coords", data=co[m])
            info[f"{task}/{sid}/rep{r}"] = {"n": int(len(va)), "n_val": int(va.sum())}
        print(task, sid, len(co), [info[f"{task}/{sid}/rep{r}"]["n_val"] for r in (0, 1, 2)], flush=True)
json.dump(info, open("embed_val/strips.json", "w"), indent=1)
