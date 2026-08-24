"""Step 4: close the loop.

    SS8 prompt -> multimodal model co-generates (sequence, structure tokens)
              -> frozen VQ decoder turns structure tokens back into 3D frames
              -> backbone rebuilt, written as a PDB
              -> our own DSSP re-run on the DECODED coordinates

The last line is the actual verification: does the geometry we generated really
have the secondary structure we asked for? This is the small-scale analogue of
"generate, then fold, then compare" used to validate ESM3-style designs.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from data import AA1, SS8, SS8_IDX, assign_ss8
from dataset import ROOT, load_splits
from multimodal import MultimodalFM, add_struct_tokens, generate
from nn import local_frames
from struct_tokenizer import StructTokenizer

BACKBONE = ("N", "CA", "C", "O")


def ideal_local_offsets(records, limit: int = 400) -> np.ndarray:
    """Mean position of N/CA/C/O inside each residue's own frame, measured from data.

    By construction CA sits at the origin and C lies on +e1, so this doubles as a
    check that local_frames() is correct.
    """
    acc = []
    for r in records[:limit]:
        c = torch.from_numpy(r["coords"])[None]
        R, t = local_frames(c)                                  # [1,L,3,3], [1,L,3]
        rel = c[0] - t[0][:, None, :]                           # [L,4,3]
        acc.append(torch.einsum("lck,lac->lak", R[0], rel).numpy())
    return np.concatenate(acc, 0).mean(0)                        # [4,3]


def rebuild_backbone(R: torch.Tensor, t: torch.Tensor, offsets: np.ndarray) -> np.ndarray:
    """frames + ideal local offsets -> global N/CA/C/O coordinates."""
    off = torch.from_numpy(offsets).to(R.dtype)                  # [4,3]
    # global = R @ local + t   (R columns are the basis)
    return (torch.einsum("lck,ak->lac", R, off) + t[:, None, :]).cpu().numpy()


def write_pdb(path, coords: np.ndarray, seq: np.ndarray):
    from data import AA3
    lines, serial = [], 1
    for i in range(len(seq)):
        resn = AA3[int(seq[i])] if int(seq[i]) < 20 else "GLY"
        for j, atom in enumerate(BACKBONE):
            x, y, z = coords[i, j]
            lines.append(f"ATOM  {serial:5d} {atom:<4s}{resn:>3s} A{i + 1:4d}    "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {atom[0]:>2s}")
            serial += 1
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def helix_bundle_prompt(n_helix=3, helix=16, loop=4) -> np.ndarray:
    """A three-helix bundle expressed purely as an SS8 constraint."""
    p = []
    for k in range(n_helix):
        p += [SS8_IDX["H"]] * helix
        if k < n_helix - 1:
            p += [SS8_IDX["T"]] * loop
    return np.array(p, np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", default="fm.pt")
    ap.add_argument("--vq", default="vqvae.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    dev = torch.device(a.device)

    vq_ck = torch.load(ROOT / "ckpt" / a.vq, map_location=dev)
    vq = StructTokenizer(codebook=vq_ck["codebook"]).to(dev)
    vq.load_state_dict(vq_ck["model"]); vq.eval()

    fm_ck = torch.load(ROOT / "ckpt" / a.fm, map_location=dev)
    fm = MultimodalFM(fm_ck["codebook"], tracks=fm_ck["tracks"], d=fm_ck["d"],
                      layers=fm_ck["layers"], geometric=fm_ck["geo"]).to(dev)
    fm.load_state_dict(fm_ck["model"]); fm.eval()

    splits = load_splits()
    offsets = ideal_local_offsets(splits["train"])
    print("[decode] measured ideal local offsets (CA at origin, C on +e1):")
    for atom, v in zip(BACKBONE, offsets):
        print(f"   {atom:<2s} {v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f}")

    prompt_ss8 = helix_bundle_prompt()
    L = len(prompt_ss8)
    print(f"\n[decode] prompt: 3-helix bundle, L={L}")
    print(f"   target ss8  {''.join(SS8[i] for i in prompt_ss8)}")

    outdir = ROOT / "out"
    outdir.mkdir(exist_ok=True)
    scores = []
    for s in range(a.samples):
        gen = generate(fm, L, dev, prompt={"ss8": prompt_ss8}, steps=a.steps,
                       temp=a.temp, seed=s)
        seq, stok = gen["seq"].numpy(), gen["struct"].numpy()

        with torch.no_grad():
            z = vq.vq.codebook[torch.from_numpy(stok).to(dev)][None]
            R, t, _, _ = vq.dec(z, torch.ones(1, L, dtype=torch.bool, device=dev))
        coords = rebuild_backbone(R[0].float(), t[0].float(), offsets)

        realised = assign_ss8(coords.astype(np.float32))
        agree = float((realised == prompt_ss8).mean())
        helix_frac = float((realised == SS8_IDX["H"]).mean())
        scores.append((agree, helix_frac))

        aa = "".join(AA1[i] if i < 20 else "X" for i in seq)
        pdb = outdir / f"design_{s}.pdb"
        write_pdb(pdb, coords, seq)
        print(f"\n[sample {s}]  ss8-agreement {agree * 100:5.1f}%   helix {helix_frac * 100:5.1f}%")
        print(f"   sequence    {aa}")
        print(f"   realised    {''.join(SS8[i] for i in realised)}")
        print(f"   wrote       {pdb}")

    ag = [x for x, _ in scores]
    print(f"\n[decode] mean ss8-agreement {np.mean(ag) * 100:.1f}%  "
          f"(random-3-state baseline ~{100 / 8:.1f}%, all-H baseline "
          f"{(prompt_ss8 == SS8_IDX['H']).mean() * 100:.1f}%)")


if __name__ == "__main__":
    main()
