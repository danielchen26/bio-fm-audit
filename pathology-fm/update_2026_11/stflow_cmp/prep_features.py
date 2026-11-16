"""Write our Phikon embeddings in the folder layout the public STFlow code reads (one h5 per sample)."""
import os, sys, glob, numpy as np, h5py
HEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hest")
for task in sys.argv[1:]:
    n = 0
    for f in sorted(glob.glob(f"{HEST}/bench_data/{task}/patches/*.h5")):
        sid = os.path.basename(f)[:-3]; out = f"embed/{task}/phikon/fp32/{sid}.h5"; os.makedirs(os.path.dirname(out), exist_ok=True)
        with h5py.File(f, "r") as h:
            bc = np.array([b.decode() if isinstance(b, bytes) else str(b) for b in h["barcode"][:].reshape(-1)], dtype="S"); coords = h["coords"][:]
        emb = np.load(f"{HEST}/emb/{task}__{sid}__phikon.npy").astype(np.float32); assert len(emb) == len(bc) == len(coords)
        with h5py.File(out, "w") as h:
            h.create_dataset("embeddings", data=emb); h.create_dataset("barcodes", data=bc.reshape(-1, 1)); h.create_dataset("coords", data=coords)
        n += 1
    print(task, n, "samples written", flush=True)
