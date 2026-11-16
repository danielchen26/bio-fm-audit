"""Gate: reproduce public HEST-Benchmark leaderboard values with the official protocol."""
import sys, json, glob, os
from hest_common import *
PUB = {"phikon": {"IDC": 0.5327, "PRAD": 0.3420, "PAAD": 0.4425, "SKCM": 0.5355, "COAD": 0.2623, "READ": 0.1532, "CCRCC": 0.2423, "LUNG": 0.5466, "LYMPH_IDC": 0.2373},
       "path_dino_vitb_kaiko": {"IDC": 0.5352, "PRAD": 0.3275, "PAAD": 0.4524, "SKCM": 0.5502, "COAD": 0.2812, "READ": 0.1525, "CCRCC": 0.2291, "LUNG": 0.5156, "LYMPH_IDC": 0.2365},
       "path_dinov2_vitl_phikon2": {"IDC": 0.5408, "PRAD": 0.3545, "PAAD": 0.4455, "SKCM": 0.5554, "COAD": 0.2500, "READ": 0.1749, "CCRCC": 0.2659, "LUNG": 0.5419, "LYMPH_IDC": 0.2437}}
tasks = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{ROOT}/*/splits"))
encs = sys.argv[1:] or ["phikon", "path_dino_vitb_kaiko", "path_dinov2_vitl_phikon2", "in_dino_vitb", "path_dino_vits_lunit", "in_dino_vits"]
out = {}
for task in tasks:
    gl = genes(task); sp_ = splits(task); S = {s: load_sample(task, s, gl) for tr, te in sp_ for s in tr + te}
    info = {s: len(S[s]["barcodes"]) for s in S}
    for enc in encs:
        if not all(os.path.exists(f"{EMB}/{task}__{s}__{enc}.npy") for s in S):
            continue
        per = []
        for tr, te in sp_:
            Xtr, ytr = np.concatenate([emb(task, s, enc) for s in tr]), np.concatenate([S[s]["y"] for s in tr])
            Xte, yte = np.concatenate([emb(task, s, enc) for s in te]), np.concatenate([S[s]["y"] for s in te])
            per.append(float(np.nanmean(pearson(official_ridge(Xtr, ytr, Xte), yte))))
        pub = PUB.get(enc, {}).get(task)
        out.setdefault(task, {})[enc] = {"mean": round(float(np.mean(per)), 4), "per_split": [round(v, 4) for v in per], "published": pub,
                                         "diff": None if pub is None else round(float(np.mean(per)) - pub, 4)}
        print(f"{task:10s} {enc:26s} ours {np.mean(per):.4f} published {pub} diff {out[task][enc]['diff']} per split {np.round(per, 4).tolist()} | spots {info}", flush=True)
json.dump(out, open("hest_ridge.json", "w"), indent=1)
