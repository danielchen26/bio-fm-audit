"""Diagnostic 2: do structure tokens encode LOCAL backbone geometry?

Global 3D reconstruction from tokens is not what a structure token is for --
ESM3's tokens are local by construction and its decoder is a separate large
model. The property the multimodal trunk actually needs is: does the token id
pin down local backbone conformation?

Two parameter-free tests on the held-out split:
  1. token -> SS8 via a plain lookup table built on train (no network)
  2. mutual information I(token; SS8) and I(token; phi/psi bin)
Both compared against the marginal baseline. A lookup table cannot overfit in
any interesting way, so this isolates information content from model capacity.
"""
from __future__ import annotations

import collections

import numpy as np
import torch

from data import SS8
from dataset import ROOT, batches, load_splits
from struct_tokenizer import StructTokenizer


def dihedrals(coords: np.ndarray) -> np.ndarray:
    """phi/psi per residue, binned into a 6x6 Ramachandran grid."""
    def torsion(p0, p1, p2, p3):
        b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
        b1 /= np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8
        v = b0 - (b0 * b1).sum(-1, keepdims=True) * b1
        w = b2 - (b2 * b1).sum(-1, keepdims=True) * b1
        x = (v * w).sum(-1)
        y = (np.cross(b1, v) * w).sum(-1)
        return np.degrees(np.arctan2(y, x))

    N, CA, C = coords[:, 0], coords[:, 1], coords[:, 2]
    L = len(coords)
    phi = np.full(L, np.nan); psi = np.full(L, np.nan)
    phi[1:] = torsion(C[:-1], N[1:], CA[1:], C[1:])
    psi[:-1] = torsion(N[:-1], CA[:-1], C[:-1], N[1:])
    b = lambda x: np.clip(((x + 180) / 60).astype(int), 0, 5)
    out = np.where(np.isnan(phi) | np.isnan(psi), -1, b(np.nan_to_num(phi)) * 6 + b(np.nan_to_num(psi)))
    return out.astype(int)


def mutual_information(x, y) -> float:
    """I(X;Y) in bits."""
    joint = collections.Counter(zip(x, y))
    n = sum(joint.values())
    px = collections.Counter(x); py = collections.Counter(y)
    return sum(c / n * np.log2((c / n) / (px[a] / n * py[b] / n))
               for (a, b), c in joint.items() if c > 0)


def main():
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "vqvae.pt"
    dev = torch.device("cpu")
    print(f"[diag2] tokenizer = {name}")
    ck = torch.load(ROOT / "ckpt" / name, map_location=dev)
    model = StructTokenizer(codebook=ck["codebook"]).to(dev)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    # v1 predates the local-reconstruction head; the probe only touches the
    # encoder and codebook, so an uninitialised decoder head is irrelevant here
    if missing or unexpected:
        print(f'[diag2] load strict=False | missing {len(missing)} unexpected {len(unexpected)}')
    model.eval()

    splits = load_splits()
    data = {}
    with torch.no_grad():
        for name in ("train", "test"):
            recs = splits[name] if name == "test" else splits[name][:1500]
            toks, ss8s, rama = [], [], []
            order = sorted(range(len(recs)), key=lambda i: len(recs[i]["seq"]))
            ptr = 0
            for b in batches([recs[i] for i in order], 3000, dev, shuffle=False):
                tk = model.encode(b["coords"], b["mask"])
                for i in range(len(tk)):
                    n = int(b["mask"][i].sum())
                    r = recs[order[ptr]]; ptr += 1
                    toks += tk[i, :n].tolist()
                    ss8s += r["ss8"].tolist()
                    rama += dihedrals(r["coords"]).tolist()
            data[name] = (toks, ss8s, rama)

    tr_tok, tr_ss, tr_ra = data["train"]
    te_tok, te_ss, te_ra = data["test"]

    print(f"[diag2] train residues {len(tr_tok)}  test residues {len(te_tok)}")
    print(f"[diag2] distinct tokens used: train {len(set(tr_tok))}  test {len(set(te_tok))}")

    for label, tr_y, te_y, n_cls in (("SS8", tr_ss, te_ss, 8), ("Ramachandran-36", tr_ra, te_ra, 36)):
        tab = collections.defaultdict(collections.Counter)
        for t, y in zip(tr_tok, tr_y):
            tab[t][y] += 1
        pred = {t: c.most_common(1)[0][0] for t, c in tab.items()}
        marg = collections.Counter(tr_y).most_common(1)[0][0]
        acc = np.mean([pred.get(t, marg) == y for t, y in zip(te_tok, te_y)])
        base = np.mean([marg == y for y in te_y])
        mi = mutual_information(te_tok, te_y)
        h = -sum((c / len(te_y)) * np.log2(c / len(te_y)) for c in collections.Counter(te_y).values())
        print(f"\n  token -> {label}")
        print(f"    lookup-table accuracy   {acc * 100:5.1f}%")
        print(f"    marginal baseline       {base * 100:5.1f}%")
        print(f"    lift                    {acc / base:5.2f}x")
        print(f"    I(token;{label}) = {mi:.2f} bits of H({label}) = {h:.2f} bits  "
              f"-> {mi / h * 100:.0f}% of the entropy explained")


if __name__ == "__main__":
    main()
