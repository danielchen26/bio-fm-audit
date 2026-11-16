"""Embed the spot patches of each slide with the same encoder grid as CAMELYON17 (own normalisation per encoder)."""
import sys, os, json, time
sys.dont_write_bytecode = True
import numpy as np, torch, timm
DEV = "mps"
GRID = [("phikon", "hf", "owkin/phikon"), ("in_sup_vits", "timm", "vit_small_patch16_224.augreg_in21k_ft_in1k"),
        ("in_dino_vits", "timm", "vit_small_patch16_224.dino"), ("path_dino_vits_lunit", "timm", "hf-hub:1aurent/vit_small_patch16_224.lunit_dino"),
        ("in_dino_vitb", "timm", "vit_base_patch16_224.dino"), ("in_sup_vitb", "timm", "vit_base_patch16_224.augreg_in21k_ft_in1k"),
        ("path_dino_vitb_kaiko", "timm", "hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms"),
        ("path_dinov2_vitl_phikon2", "hf", "owkin/phikon-v2")]


def load(kind, ident):
    if kind == "timm":
        m = timm.create_model(ident, pretrained=True, num_classes=0).to(DEV).eval()
        cfg = timm.data.resolve_model_data_config(m)
        return (lambda x: m.forward_head(m.forward_features(x), pre_logits=True)), list(cfg["mean"]), list(cfg["std"]), m
    from transformers import AutoModel, AutoImageProcessor
    m = AutoModel.from_pretrained(ident, **({"add_pooling_layer": False} if ident == "owkin/phikon" else {})).to(DEV).eval()
    pr = AutoImageProcessor.from_pretrained(ident)
    return (lambda x: m(pixel_values=x).last_hidden_state[:, 0]), list(pr.image_mean), list(pr.image_std), m

slides = {}
for sid in ("V1LN", "BRCA", "V2LN"):
    while not os.path.exists(f"st_slide_{sid}.npz"):
        time.sleep(5)
    time.sleep(3)
    with np.load(f"st_slide_{sid}.npz", allow_pickle=False) as z:
        slides[sid] = z["patches"]
for name, kind, ident in GRID:
    if all(os.path.exists(f"st_feat_{sid}_{name}.npy") for sid in slides):
        continue
    feat, mean, std, m = load(kind, ident)
    mean_t = torch.tensor(mean).view(1, 3, 1, 1).to(DEV); std_t = torch.tensor(std).view(1, 3, 1, 1).to(DEV)
    for sid, p in slides.items():
        parts = []
        for j in range(0, len(p), 128):
            x = torch.from_numpy(p[j:j + 128]).to(DEV).permute(0, 3, 1, 2).float() / 255.0
            with torch.inference_mode():
                parts.append(feat((x - mean_t) / std_t).float().cpu().numpy())
        e = np.concatenate(parts).astype(np.float32); np.save(f"st_feat_{sid}_{name}.npy", e)
        msg = f"[{name}] {sid} {e.shape}"
        ref_file = os.path.join(os.environ.get("PFM_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")), "morph_phikon_p224.npz")
        if name == "phikon" and sid == "V1LN" and os.path.exists(ref_file):   # optional gate against src/xmodal.py features
            with np.load(ref_file, allow_pickle=False) as z:
                ref = z["feat"]
            c = float(((e * ref).sum(1) / (np.linalg.norm(e, axis=1) * np.linalg.norm(ref, axis=1))).min())
            msg += f" | gate: min cosine vs project Phikon features {c:.6f}"; assert c > 0.999
        print(msg, flush=True)
    del m
print("done", flush=True)
