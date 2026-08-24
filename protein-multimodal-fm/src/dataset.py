"""Batching. Length-bucketed to keep padding waste low."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
PAD = -100  # ignore_index sentinel for label tensors


def load_splits():
    with (ROOT / "data" / "dataset.pkl").open("rb") as fh:
        return pickle.load(fh)


def collate(records: list[dict], device) -> dict:
    L = max(len(r["seq"]) for r in records)
    B = len(records)
    out = {
        "seq": torch.full((B, L), 0, dtype=torch.long),
        "ss8": torch.full((B, L), 0, dtype=torch.long),
        "sasa": torch.full((B, L), 0, dtype=torch.long),
        "coords": torch.zeros(B, L, 4, 3),
        "mask": torch.zeros(B, L, dtype=torch.bool),
    }
    has_struct = "struct" in records[0]
    if has_struct:
        out["struct"] = torch.zeros(B, L, dtype=torch.long)
    for i, r in enumerate(records):
        n = len(r["seq"])
        out["seq"][i, :n] = torch.from_numpy(r["seq"].astype(np.int64))
        out["ss8"][i, :n] = torch.from_numpy(r["ss8"].astype(np.int64))
        out["sasa"][i, :n] = torch.from_numpy(r["sasa"].astype(np.int64))
        out["coords"][i, :n] = torch.from_numpy(r["coords"])
        out["mask"][i, :n] = True
        if has_struct:
            out["struct"][i, :n] = torch.from_numpy(r["struct"].astype(np.int64))
    return {k: v.to(device) for k, v in out.items()}


def batches(records: list[dict], batch_tokens: int, device, shuffle=True, seed=0):
    """Token-budget batching: sort by length, pack until budget, then shuffle order."""
    idx = sorted(range(len(records)), key=lambda i: len(records[i]["seq"]))
    groups, cur = [], []
    for i in idx:
        cur.append(i)
        if len(cur) * len(records[cur[-1]]["seq"]) >= batch_tokens:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    if shuffle:
        np.random.default_rng(seed).shuffle(groups)
    for g in groups:
        yield collate([records[i] for i in g], device)
