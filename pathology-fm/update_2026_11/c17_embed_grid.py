"""CAMELYON17: embed all tiles with a matched grid of encoders (ImageNet vs pathology pretraining, ViT-S vs ViT-B).
Every encoder gets the input normalisation its own model config specifies. Resumable; cheap encoders first."""
import sys, os, glob, time, json
sys.dont_write_bytecode = True
import numpy as np, torch, timm
import os; PROJ = os.environ.get("PFM_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.environ.get("C17_TILESET", f"{PROJ}/data/camelyon17_tileset")   # tiles/<slide>.npz plus the two reference embeddings
files = sorted(glob.glob(f"{ROOT}/tiles/*.npz"))
DEV = "mps"
GRID = [("in_sup_vits", "timm", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
        ("in_dino_vits", "timm", "vit_small_patch16_224.dino"),
        ("path_dino_vits_lunit", "timm", "hf-hub:1aurent/vit_small_patch16_224.lunit_dino"),
        ("in_dino_vitb", "timm", "vit_base_patch16_224.dino"),
        ("in_sup_vitb", "timm", "vit_base_patch16_224.augreg_in21k_ft_in1k"),
        ("path_dino_vitb_kaiko", "timm", "hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms"),
        ("path_dinov2_vitl_phikon2", "hf", "owkin/phikon-v2")]
IMNET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load(kind, ident):
    if kind == "timm":
        m = timm.create_model(ident, pretrained=True, num_classes=0).to(DEV).eval()
        cfg = timm.data.resolve_model_data_config(m)
        return (lambda x: m.forward_head(m.forward_features(x), pre_logits=True)), list(cfg["mean"]), list(cfg["std"]), int(cfg["input_size"][-1]), m
    from transformers import AutoModel, AutoImageProcessor
    m = AutoModel.from_pretrained(ident).to(DEV).eval()
    pr = AutoImageProcessor.from_pretrained(ident)
    return (lambda x: m(pixel_values=x).last_hidden_state[:, 0]), list(pr.image_mean), list(pr.image_std), 224, m


def embed(feat, mean, std, size, tag, limit=None, batch=128):
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(DEV); std_t = torch.tensor(std).view(1, 3, 1, 1).to(DEV)
    parts, n, t0 = [], 0, time.time()
    for i, f in enumerate(files[:limit] if limit else files):
        with np.load(f, allow_pickle=False) as z:
            p = z["patches"]
        for j in range(0, len(p), batch):
            x = torch.from_numpy(p[j:j + batch]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            if x.shape[-1] != size:
                x = torch.nn.functional.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
            with torch.inference_mode():
                parts.append(feat((x - mean_t) / std_t).float().cpu().numpy())
        n += len(p)
        if i % 5 == 0 and not limit:
            print(f"[{tag}] shard {i + 1}/{len(files)}  {n} tiles  {n / (time.time() - t0):.0f} tiles/s", flush=True)
    return np.concatenate(parts).astype(np.float32)


def cos(a, b):
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)).min())

meta = json.load(open("c17_embed_grid.json")) if os.path.exists("c17_embed_grid.json") else {}
# ---- sanity gates: the new embedding path must reproduce the stored embeddings on the first shard
if "sanity" not in meta:
    from transformers import AutoModel
    m = AutoModel.from_pretrained("owkin/phikon", add_pooling_layer=False).to(DEV).eval()
    e = embed(lambda x: m(pixel_values=x).last_hidden_state[:, 0], *IMNET, 224, "phikon-check", limit=1)
    with np.load(f"{ROOT}/ref_emb_phikon.npz", allow_pickle=False) as z:
        ref = z["emb"][:len(e)]
    c1 = cos(e, ref); del m
    feat, mean, std, size, m = load("timm", GRID[0][2])
    e = embed(feat, *IMNET, 224, "imagenet-legacy-check", limit=1)
    c2 = cos(e, np.load("c17_emb_imagenet.npy")[:len(e)])
    meta["sanity"] = {"phikon_min_cosine_vs_stored": c1, "imagenet_legacy_norm_min_cosine_vs_stored": c2,
                      "imagenet_vits_model_config_mean_std": [mean, std], "legacy_run_used_mean_std": list(IMNET)}
    print("sanity", meta["sanity"], flush=True)
    assert c1 > 0.999 and c2 > 0.999, "new embedding path does not reproduce stored embeddings"
    json.dump(meta, open("c17_embed_grid.json", "w"), indent=1); del m
for name, kind, ident in GRID:
    out = f"c17_emb_{name}.npy"
    if os.path.exists(out):
        print(f"[{name}] exists, skipping", flush=True); continue
    t0 = time.time()
    feat, mean, std, size, m = load(kind, ident)
    print(f"[{name}] {ident}  mean {mean} std {std} input {size}  params {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M", flush=True)
    e = embed(feat, mean, std, size, name)
    np.save(out + ".tmp.npy", e); os.replace(out + ".tmp.npy", out)
    meta[name] = {"id": ident, "mean": mean, "std": std, "input": size, "dim": int(e.shape[1]), "n": int(e.shape[0]),
                  "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 1), "seconds": round(time.time() - t0)}
    json.dump(meta, open("c17_embed_grid.json", "w"), indent=1)
    print(f"[{name}] done {e.shape} in {time.time() - t0:.0f}s", flush=True); del m
print("all done", flush=True)
