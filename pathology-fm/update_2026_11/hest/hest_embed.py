"""Embed every HEST-Benchmark patch file with the encoder grid (each encoder's own input normalisation)."""
import sys, os, glob, json, time
sys.dont_write_bytecode = True
import numpy as np, torch, timm, h5py
DEV = "mps"
GRID = [("phikon", "hf", "owkin/phikon"), ("path_dino_vitb_kaiko", "timm", "hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms"),
        ("in_dino_vitb", "timm", "vit_base_patch16_224.dino"), ("path_dino_vits_lunit", "timm", "hf-hub:1aurent/vit_small_patch16_224.lunit_dino"),
        ("in_dino_vits", "timm", "vit_small_patch16_224.dino"), ("path_dinov2_vitl_phikon2", "hf", "owkin/phikon-v2")]
want = sys.argv[1:]


def load(kind, ident):
    if kind == "timm":
        m = timm.create_model(ident, pretrained=True, num_classes=0).to(DEV).eval()
        cfg = timm.data.resolve_model_data_config(m)
        return (lambda x: m.forward_head(m.forward_features(x), pre_logits=True)), list(cfg["mean"]), list(cfg["std"]), m
    from transformers import AutoModel, AutoImageProcessor
    m = AutoModel.from_pretrained(ident, **({"add_pooling_layer": False} if ident == "owkin/phikon" else {})).to(DEV).eval()
    pr = AutoImageProcessor.from_pretrained(ident)
    return (lambda x: m(pixel_values=x).last_hidden_state[:, 0]), list(pr.image_mean), list(pr.image_std), m

files = sorted(glob.glob("bench_data/*/patches/*.h5"))
os.makedirs("emb", exist_ok=True); meta = {}
for name, kind, ident in GRID:
    if want and name not in want:
        continue
    todo = [f for f in files if not os.path.exists(f"emb/{f.split('/')[1]}__{os.path.basename(f)[:-3]}__{name}.npy")]
    if not todo:
        continue
    feat, mean, std, m = load(kind, ident); t0 = time.time(); n = 0
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(DEV); std_t = torch.tensor(std).view(1, 3, 1, 1).to(DEV)
    for f in todo:
        with h5py.File(f, "r") as h:
            p = h["img"][:]
        assert p.shape[1:] == (224, 224, 3), p.shape
        parts = []
        for j in range(0, len(p), 128):
            x = torch.from_numpy(p[j:j + 128]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            with torch.inference_mode():
                parts.append(feat((x - mean_t) / std_t).float().cpu().numpy())
        out = f"emb/{f.split('/')[1]}__{os.path.basename(f)[:-3]}__{name}.npy"
        np.save(out + ".tmp.npy", np.concatenate(parts).astype(np.float32)); os.replace(out + ".tmp.npy", out); n += len(p)
    print(f"[{name}] mean {mean} std {std}: {len(todo)} files, {n} patches, {n / (time.time() - t0):.0f} patches/s", flush=True); del m
print("done", flush=True)
