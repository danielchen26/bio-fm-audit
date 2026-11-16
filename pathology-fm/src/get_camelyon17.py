"""Download the 50 lesion-annotated CAMELYON17 slides + their tumour masks.

Why this subset: the annotations are stratified exactly 10 slides per centre
across all 5 contributing hospitals, which makes CENTRE the experimental
variable. That is the confound stain normalisation is actually for, and it is
what the single-cohort BreaKHis study could not test.

Bandwidth is the binding constraint, not disk. Measured on this machine: 17 MB/s
single-stream, and 8 concurrent streams gave the SAME 17 MB/s aggregate -- the
link is saturated, so parallelism buys nothing and only risks partial files.
Sequential with resume it is.

Resume is not optional. 151 GB at 17 MB/s is ~2.5 h; anything that dies at 90%
without resume wastes two hours. Same lesson as mid-run checkpointing.
"""
from __future__ import annotations

import os
import argparse
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = "https://camelyon-dataset.s3.us-west-2.amazonaws.com"
PREFIX = "CAMELYON17"
# CAMELYON17 slides, tiles and caches (about 180 GB). Set C17_ROOT to keep them on another disk.
DEST = Path(os.environ.get("C17_ROOT", Path(__file__).resolve().parent.parent / "data" / "camelyon17"))


def list_keys(prefix: str) -> dict[str, int]:
    """S3 ListObjectsV2 over the public endpoint, with continuation."""
    out, token = {}, None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        url = f"{BUCKET}/?list-type=2&max-keys=1000&prefix={prefix}"
        if token:
            url += f"&continuation-token={urllib.request.quote(token, safe='')}"
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{ns}Contents"):
            out[c.findtext(f"{ns}Key")] = int(c.findtext(f"{ns}Size"))
        if root.findtext(f"{ns}IsTruncated") != "true":
            return out
        token = root.findtext(f"{ns}NextContinuationToken")


def centre_of(slide_id: str) -> int:
    """CAMELYON17 training patients 0-99, 20 per centre, in order."""
    return int(slide_id.split("_")[1]) // 20


def fetch(key: str, dest: Path, expect: int, retries: int = 5) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        have = dest.stat().st_size if dest.exists() else 0
        if have == expect:
            return True
        if have > expect:                      # corrupt/partial from a bad run
            dest.unlink()
            have = 0
        # -C - resumes; --retry rides out transient S3 hiccups
        cmd = ["curl", "-sS", "-C", "-", "--retry", "3", "--retry-delay", "5",
               "-o", str(dest), f"{BUCKET}/{key}"]
        subprocess.run(cmd, check=False)
        if dest.exists() and dest.stat().st_size == expect:
            return True
        print(f"    retry {attempt + 1}/{retries} for {dest.name} "
              f"({dest.stat().st_size if dest.exists() else 0}/{expect})", flush=True)
        time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all 50 annotated slides")
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--negatives", type=int, default=0,
                    help="also fetch N negative slides per centre")
    a = ap.parse_args()
    dest = Path(a.dest)

    print("[c17] listing bucket ...", flush=True)
    ann = list_keys(f"{PREFIX}/annotations/")
    imgs = list_keys(f"{PREFIX}/images/")
    masks = list_keys(f"{PREFIX}/masks/")

    ids = sorted(Path(k).stem for k in ann)

    if a.negatives:
        # The annotated set is 100% positive by construction, so slide-level
        # grading has no negative class. Pull N negative slides per centre from
        # stages.csv to complete the 4-class task.
        import csv as _csv
        neg_by_c: dict[int, list[str]] = {}
        with open(Path(a.dest) / "stages.csv") as fh:
            for row in _csv.DictReader(fh):
                k = row["patient"].strip()
                if k.endswith(".tif") and row["stage"].strip() == "negative":
                    neg_by_c.setdefault(int(row["center"]), []).append(k[:-4])
        picked = []
        for c in sorted(neg_by_c):
            picked += sorted(neg_by_c[c])[:a.negatives]
        print(f"[c17] + {len(picked)} negative slides ({a.negatives} per centre)")
        ids = ids + picked
    per_centre: dict[int, int] = {}
    for i in ids:
        per_centre[centre_of(i)] = per_centre.get(centre_of(i), 0) + 1
    print(f"[c17] {len(ids)} annotated slides; per-centre {dict(sorted(per_centre.items()))}")

    if a.limit:
        # keep centres balanced when subsetting
        by_centre: dict[int, list[str]] = {}
        for i in ids:
            by_centre.setdefault(centre_of(i), []).append(i)
        k = max(1, a.limit // len(by_centre))
        ids = [x for c in sorted(by_centre) for x in by_centre[c][:k]]
        print(f"[c17] subset to {len(ids)} slides ({k} per centre)")

    # Round-robin across centres. Patient index maps monotonically to centre
    # (0-19 -> centre 0, ...), so plain sorted order downloads all of centre 0
    # first and leave-one-centre-out is impossible until the very end. Interleaving
    # makes the centre experiment runnable at ~40% of the transfer instead of 100%.
    by_c: dict[int, list[str]] = {}
    for i in ids:
        by_c.setdefault(centre_of(i), []).append(i)
    ids = [x for tup in zip(*(by_c[c] for c in sorted(by_c))) for x in tup]
    print(f"[c17] centre-interleaved order; first 5: {ids[:5]}")

    plan = []
    for sid in ids:
        ik, mk = f"{PREFIX}/images/{sid}.tif", f"{PREFIX}/masks/{sid}_mask.tif"
        if ik not in imgs:
            print(f"[c17] WARNING no image for {sid}")
            continue
        plan.append((ik, dest / "images" / f"{sid}.tif", imgs[ik]))
        if mk in masks:
            plan.append((mk, dest / "masks" / f"{sid}_mask.tif", masks[mk]))

    total = sum(sz for _, _, sz in plan)
    done = sum(sz for _, p, sz in plan if p.exists() and p.stat().st_size == sz)
    print(f"[c17] {len(plan)} files, {total / 1e9:.1f} GB total, "
          f"{done / 1e9:.1f} GB already present, "
          f"{(total - done) / 1e6 / 17 / 60:.0f} min remaining at 17 MB/s", flush=True)

    t0, moved, failed = time.time(), 0, []
    for n, (key, path, size) in enumerate(plan, 1):
        if path.exists() and path.stat().st_size == size:
            continue
        ok = fetch(key, path, size)
        if not ok:
            failed.append(key)
            print(f"[c17] FAILED {key}", flush=True)
            continue
        moved += size
        el = time.time() - t0
        rate = moved / 1e6 / max(el, 1e-9)
        left = (total - done - moved) / 1e6 / max(rate, 1e-9) / 60
        print(f"[{n:3d}/{len(plan)}] {path.name:34s} {size / 1e9:5.2f} GB  "
              f"{rate:5.1f} MB/s  ~{left:5.1f} min left", flush=True)

    print(f"\n[c17] done. {moved / 1e9:.1f} GB fetched in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"[c17] {len(failed)} FAILURES: {failed}")
        sys.exit(1)
    have = [p for _, p, sz in plan if p.exists() and p.stat().st_size == sz]
    print(f"[c17] verified {len(have)}/{len(plan)} files match expected size")


if __name__ == "__main__":
    main()
