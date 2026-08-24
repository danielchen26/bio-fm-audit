"""Diagnostic: is the structure information missing from the TOKENS, or only from
the DECODER HEAD?

CA-RMSD from predicted frames was 11 A, but the distance-map error was only
~2.4 A. Those two are consistent with "local geometry learned, global assembly
not" -- a weak decoder head, not uninformative tokens.

Test: rebuild coordinates from the predicted DISTOGRAM via classical MDS, which
uses no frame prediction at all. If MDS-RMSD << frame-RMSD, the information is
in the tokens and the frame head is the bottleneck. That decides whether to fix
the tokenizer or move on.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from dataset import ROOT, batches, load_splits
from nn import kabsch_rmsd
from struct_tokenizer import N_DISTO, StructTokenizer


def mds_from_distances(D: np.ndarray) -> np.ndarray:
    """Classical multidimensional scaling: distance matrix -> 3D coordinates."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J                       # double centering
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:3]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")        # GPU is busy training
    ap.add_argument("--limit", type=int, default=120)
    a = ap.parse_args()
    dev = torch.device(a.device)

    ck = torch.load(ROOT / "ckpt" / "vqvae.pt", map_location=dev)
    model = StructTokenizer(codebook=ck["codebook"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    centers = torch.linspace(2.0, 22.0, N_DISTO - 1)
    centers = torch.cat([centers[:1] - 0.5, (centers[:-1] + centers[1:]) / 2, centers[-1:] + 0.5])

    splits = load_splits()
    test = splits["test"][:a.limit]
    frame_rmsd, mds_rmsd, dmap_err, contact_f1 = [], [], [], []

    with torch.no_grad():
        for b in batches(test, 2000, dev, shuffle=False):
            R, t, disto, _, tok, _, _, _ = model(b["coords"], b["mask"])
            p = disto.softmax(-1)
            Dexp = (p * centers.to(dev)).sum(-1)                # expected distance
            for i in range(len(t)):
                m = b["mask"][i]
                n = int(m.sum())
                true_ca = b["coords"][i][m][:, 1].float()
                Dtrue = torch.cdist(true_ca, true_ca)

                frame_rmsd.append(kabsch_rmsd(t[i][m].float(), true_ca).item())

                D = Dexp[i][:n][:, :n].float()
                D = (D + D.T) / 2
                D.fill_diagonal_(0)
                xyz = torch.from_numpy(mds_from_distances(D.numpy())).float()
                mds_rmsd.append(kabsch_rmsd(xyz, true_ca).item())

                dmap_err.append((D - Dtrue).abs().mean().item())
                # contact prediction (< 8 A, |i-j| >= 6) -- the standard proxy metric
                sep = (torch.arange(n)[:, None] - torch.arange(n)[None]).abs() >= 6
                pt, pp = (Dtrue < 8) & sep, (D < 8) & sep
                tp = (pt & pp).sum().item()
                prec = tp / max(pp.sum().item(), 1)
                rec = tp / max(pt.sum().item(), 1)
                contact_f1.append(2 * prec * rec / max(prec + rec, 1e-9))

    f = lambda v: f"{np.mean(v):6.2f}"
    print(f"[diag] n={len(frame_rmsd)} test proteins")
    print(f"  CA-RMSD from predicted FRAMES     {f(frame_rmsd)} A")
    print(f"  CA-RMSD from predicted DISTOGRAM  {f(mds_rmsd)} A   <- MDS, no frame head")
    print(f"  mean |d_pred - d_true|            {f(dmap_err)} A")
    print(f"  contact F1 (<8A, |i-j|>=6)        {np.mean(contact_f1):6.3f}")
    verdict = "DECODER HEAD is the bottleneck" if np.mean(mds_rmsd) < 0.7 * np.mean(frame_rmsd) \
        else "TOKENS themselves lack structure information"
    print(f"\n[diag] verdict: {verdict}")


if __name__ == "__main__":
    main()
