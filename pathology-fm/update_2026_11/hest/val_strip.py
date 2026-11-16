"""Validation strip shared by both models (Phase 4): one of five equal-count strips along the slide's longer coordinate axis."""
import numpy as np
STRIP_OF_REP = {0: 4, 1: 2, 2: 0}          # replicate -> strip index (one end, the middle, the other end)


def val_strip(coords, rep, n_strips=5):
    c = np.asarray(coords, dtype=np.float64)
    ax = int(np.argmax(c.max(0) - c.min(0)))
    q = np.quantile(c[:, ax], np.linspace(0, 1, n_strips + 1)); k = STRIP_OF_REP[rep]
    return (c[:, ax] >= q[k]) & ((c[:, ax] < q[k + 1]) if k < n_strips - 1 else (c[:, ax] <= q[k + 1]))
