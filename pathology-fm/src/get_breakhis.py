#!/usr/bin/env python3
"""Acquire the BreaKHis_v1 breast-cancer histopathology dataset and build a manifest.

Downloads BreaKHis_v1.tar.gz (Spanhol et al., UFPR Vision & Robotics Lab), extracts it
into data/BreaKHis_v1/, parses every image filename, and writes
data/breakhis_manifest.csv with columns:

    path,patient_id,label,subtype,magnification,seq

Filename grammar (e.g. SOB_B_TA-14-4659-40-001.png):

    SOB _ <B|M> _ <SUBTYPE> - <patient id> - <magnification> - <sequence> .png

Patient ids are of the form "14-4659" and may carry a trailing letter group
(e.g. "14-16184CD"), so magnification and sequence are anchored from the right.

Run:  python src/get_breakhis.py
"""

from __future__ import annotations

import re
import shutil
import sys
import tarfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
ARCHIVE = DATA_DIR / "BreaKHis_v1.tar.gz"
EXTRACT_ROOT = DATA_DIR / "BreaKHis_v1"
MANIFEST = DATA_DIR / "breakhis_manifest.csv"
SOURCE_URL_FILE = DATA_DIR / "BreaKHis_v1.tar.gz.source"

# Candidate sources, tried in order.
CANDIDATE_URLS = [
    "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
    "https://web.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz",
]
LANDING_PAGE = (
    "https://web.inf.ufpr.br/vri/databases/"
    "breast-cancer-histopathological-database-breakhis/"
)

# ---------------------------------------------------------------------------
# Filename grammar
# ---------------------------------------------------------------------------

BENIGN_SUBTYPES = {"A", "F", "PT", "TA"}
MALIGNANT_SUBTYPES = {"DC", "LC", "MC", "PC"}
VALID_MAGNIFICATIONS = {40, 100, 200, 400}

# Magnification and sequence are anchored at the right so a greedy patient-id
# capture cannot swallow them.
FILENAME_RE = re.compile(
    r"^SOB_(?P<cls>[BM])_(?P<subtype>[A-Z]+)-(?P<patient_id>.+)"
    r"-(?P<magnification>40|100|200|400)-(?P<seq>\d+)$"
)


class ParseError(Exception):
    """Raised for a filename that does not satisfy the BreakHis grammar."""


def parse_filename(name: str) -> dict:
    """Parse one BreakHis image filename (with extension) into its fields.

    Raises ParseError with a specific reason on anything unexpected.
    """
    stem, dot, ext = name.rpartition(".")
    if not dot or ext.lower() != "png":
        raise ParseError(f"not a .png file: {name!r}")

    match = FILENAME_RE.match(stem)
    if match is None:
        raise ParseError(f"does not match SOB_<B|M>_<SUBTYPE>-<patient>-<mag>-<seq>: {name!r}")

    cls = match.group("cls")
    subtype = match.group("subtype")
    patient_id = match.group("patient_id")
    magnification = int(match.group("magnification"))
    seq = match.group("seq")

    if cls == "B":
        label = "benign"
        allowed = BENIGN_SUBTYPES
    else:
        label = "malignant"
        allowed = MALIGNANT_SUBTYPES

    if subtype not in allowed:
        raise ParseError(
            f"subtype {subtype!r} is not a known {label} subtype "
            f"({sorted(allowed)}): {name!r}"
        )
    if magnification not in VALID_MAGNIFICATIONS:
        raise ParseError(f"unexpected magnification {magnification}: {name!r}")
    if not re.fullmatch(r"\d+-\d+[A-Z]*", patient_id):
        raise ParseError(f"patient id {patient_id!r} is not <digits>-<digits>[LETTERS]: {name!r}")

    return {
        "patient_id": patient_id,
        "label": label,
        "subtype": subtype,
        "magnification": magnification,
        "seq": seq,
    }


# ---------------------------------------------------------------------------
# Download / extract
# ---------------------------------------------------------------------------


def _human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TiB"


def scrape_landing_page() -> list[str]:
    """Find tar.gz / zip download links advertised on the official landing page."""
    print(f"[download] scraping {LANDING_PAGE}")
    try:
        resp = requests.get(LANDING_PAGE, timeout=60)
    except requests.RequestException as exc:
        print(f"[download]   landing page request failed: {exc}")
        return []
    print(f"[download]   HTTP {resp.status_code}")
    if resp.status_code != 200:
        return []

    hrefs = re.findall(r'href=["\']([^"\']+)["\']', resp.text, flags=re.I)
    found: list[str] = []
    for href in hrefs:
        if re.search(r"\.(tar\.gz|tgz|zip)$", href, flags=re.I):
            url = requests.compat.urljoin(resp.url, href)
            if url not in found:
                found.append(url)
    for url in found:
        print(f"[download]   candidate from page: {url}")
    return found


def download_archive() -> str:
    """Download the archive if absent. Returns the URL that worked."""
    urls = list(CANDIDATE_URLS)
    attempted: list[str] = []

    while True:
        for url in urls:
            if url in attempted:
                continue
            attempted.append(url)
            print(f"[download] trying {url}")
            try:
                with requests.get(url, stream=True, timeout=(30, 300)) as resp:
                    print(f"[download]   HTTP {resp.status_code}")
                    if resp.status_code != 200:
                        continue
                    ctype = resp.headers.get("Content-Type", "")
                    total = int(resp.headers.get("Content-Length", 0))
                    print(f"[download]   Content-Type: {ctype}  Content-Length: {total}")
                    if "text/html" in ctype.lower():
                        print("[download]   looks like an HTML error page, skipping")
                        continue

                    tmp = ARCHIVE.with_suffix(ARCHIVE.suffix + ".part")
                    got = 0
                    last = time.monotonic()
                    with open(tmp, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=4 << 20):
                            fh.write(chunk)
                            got += len(chunk)
                            now = time.monotonic()
                            if now - last > 15:
                                pct = f" ({100 * got / total:.1f}%)" if total else ""
                                print(f"[download]   {_human(got)}{pct}")
                                last = now
                    tmp.replace(ARCHIVE)
                    print(f"[download]   done: {_human(got)} -> {ARCHIVE}")
                    return url
            except requests.RequestException as exc:
                print(f"[download]   failed: {exc}")

        extra = [u for u in scrape_landing_page() if u not in attempted]
        if not extra:
            raise SystemExit(
                "[download] FATAL: no usable BreaKHis_v1 archive URL. Tried:\n  "
                + "\n  ".join(attempted)
            )
        urls = extra


def ensure_archive() -> str:
    """Ensure the archive is on disk. Returns the URL it actually came from."""
    if ARCHIVE.exists() and ARCHIVE.stat().st_size > 0:
        print(f"[download] archive already present: {ARCHIVE} ({_human(ARCHIVE.stat().st_size)})")
        if SOURCE_URL_FILE.exists():
            return SOURCE_URL_FILE.read_text().strip()
        return "(unknown: archive was already on disk with no recorded source URL)"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = download_archive()
    SOURCE_URL_FILE.write_text(url + "\n")
    return url


def find_image_root() -> Path | None:
    """Return the extracted BreaKHis_v1 directory if it already holds images."""
    if EXTRACT_ROOT.is_dir() and any(EXTRACT_ROOT.rglob("*.png")):
        return EXTRACT_ROOT
    return None


def extract_archive() -> None:
    if find_image_root() is not None:
        print(f"[extract] already extracted at {EXTRACT_ROOT}")
        return

    staging = DATA_DIR / "_breakhis_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    print(f"[extract] extracting {ARCHIVE} ...")
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        members = 0
        for member in tar:
            name = member.name
            # Reject absolute paths and traversal.
            if name.startswith("/") or ".." in Path(name).parts:
                print(f"[extract]   skipping unsafe member: {name!r}")
                continue
            if "__MACOSX" in name or Path(name).name.startswith("._"):
                continue
            tar.extract(member, path=staging, filter="data")
            members += 1
            if members % 2000 == 0:
                print(f"[extract]   {members} members")
    print(f"[extract]   extracted {members} members")

    # The tarball contains a top-level BreaKHis_v1/ directory; normalise either way.
    inner = staging / "BreaKHis_v1"
    source = inner if inner.is_dir() else staging
    if EXTRACT_ROOT.exists():
        shutil.rmtree(EXTRACT_ROOT)
    source.rename(EXTRACT_ROOT)
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    print(f"[extract]   image tree at {EXTRACT_ROOT}")


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest() -> tuple[pd.DataFrame, int]:
    files = sorted(p for p in EXTRACT_ROOT.rglob("*.png") if not p.name.startswith("._"))
    print(f"[manifest] found {len(files)} .png files under {EXTRACT_ROOT}")
    if not files:
        raise SystemExit("[manifest] FATAL: no .png files found after extraction")

    rows = []
    failures: list[tuple[str, str]] = []
    dir_label_mismatch: list[str] = []
    dir_mag_mismatch: list[str] = []

    for path in files:
        try:
            fields = parse_filename(path.name)
        except ParseError as exc:
            failures.append((str(path.relative_to(PROJECT_DIR)), str(exc)))
            continue

        rel = path.relative_to(PROJECT_DIR).as_posix()
        parts = {p.lower() for p in path.parts}
        # Cross-check parsed label / magnification against the directory tree.
        if fields["label"] in ("benign", "malignant"):
            other = "malignant" if fields["label"] == "benign" else "benign"
            if other in parts and fields["label"] not in parts:
                dir_label_mismatch.append(rel)
        if f"{fields['magnification']}x" not in parts:
            dir_mag_mismatch.append(rel)

        rows.append({"path": rel, **fields})

    if failures:
        print(f"\n[manifest] FATAL: {len(failures)} filename(s) could not be parsed:")
        for rel, why in failures:
            print(f"  UNPARSED: {rel}\n            {why}")
        raise SystemExit(1)
    print("[manifest] all filenames parsed successfully (0 failures)")

    if dir_label_mismatch:
        print(f"[manifest] WARNING: {len(dir_label_mismatch)} label/dir mismatches, e.g.:")
        for rel in dir_label_mismatch[:10]:
            print(f"    {rel}")
    else:
        print("[manifest] label cross-check vs directory tree: OK")
    if dir_mag_mismatch:
        print(f"[manifest] WARNING: {len(dir_mag_mismatch)} magnification/dir mismatches, e.g.:")
        for rel in dir_mag_mismatch[:10]:
            print(f"    {rel}")
    else:
        print("[manifest] magnification cross-check vs directory tree: OK")

    df = pd.DataFrame(rows, columns=["path", "patient_id", "label", "subtype", "magnification", "seq"])
    df["magnification"] = df["magnification"].astype(int)
    df = df.sort_values(["patient_id", "magnification", "seq"], kind="stable").reset_index(drop=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    print(f"[manifest] wrote {MANIFEST} ({len(df)} rows)")
    return df, len(files)


# ---------------------------------------------------------------------------
# Verification / reporting
# ---------------------------------------------------------------------------


def sample_dimensions(df: pd.DataFrame, n: int = 5) -> list[tuple[str, int, int, str]]:
    # Spread the samples across the manifest rather than taking the first n.
    step = max(1, len(df) // n)
    picks = [df.iloc[min(i * step, len(df) - 1)] for i in range(n)]
    out = []
    for row in picks:
        with Image.open(PROJECT_DIR / row["path"]) as im:
            out.append((row["path"], im.width, im.height, im.mode))
    return out


def report(df: pd.DataFrame, n_files: int, url: str) -> None:
    per_patient = df.groupby("patient_id").size()
    labels_per_patient = df.groupby("patient_id")["label"].nunique()
    crossover = sorted(labels_per_patient[labels_per_patient > 1].index)
    patients_per_label = df.groupby("label")["patient_id"].nunique()
    subtypes_per_patient = df.groupby("patient_id")["subtype"].nunique()
    multi_subtype = sorted(subtypes_per_patient[subtypes_per_patient > 1].index)
    slide_dirs = len([d for d in EXTRACT_ROOT.rglob("SOB_*") if d.is_dir()])
    dims = sample_dimensions(df)

    verified = pd.read_csv(MANIFEST)

    print("\n" + "=" * 68)
    print("BREAKHIS (BreaKHis_v1) ACQUISITION SUMMARY")
    print("=" * 68)
    print(f"download url that worked        : {url}")
    print(f"archive                         : {ARCHIVE} ({_human(ARCHIVE.stat().st_size)})")
    print(f"extracted image tree            : {EXTRACT_ROOT}")
    print(f"manifest                        : {MANIFEST}")
    print(f"manifest columns                : {list(verified.columns)}")
    print()
    print(f"total image files on disk       : {n_files}")
    print(f"total manifest rows             : {len(df)}")
    print(f"manifest rows re-read by pandas : {len(verified)}")
    print(f"row count == file count         : {len(df) == n_files}")
    print(f"distinct patient_id             : {df['patient_id'].nunique()}")
    print(f"slide directories (SOB_*)       : {slide_dirs}")
    if multi_subtype:
        print(f"patient_id spanning >1 subtype  : {len(multi_subtype)} -> {multi_subtype}")
        print("  this is why the {} slide folders collapse to {} distinct patient ids;"
              .format(slide_dirs, df['patient_id'].nunique()))
        print("  the literature's '82 patients' counts slides, not unique patient ids.")
        print("  IMPORTANT: split by patient_id, not by slide, to avoid leakage.")
    else:
        print("patient_id spanning >1 subtype  : 0")
    print()
    print("images per magnification:")
    for mag in sorted(df["magnification"].unique()):
        print(f"  {mag:>3}X : {int((df['magnification'] == mag).sum())}")
    print()
    print("images per label:")
    for label, count in df["label"].value_counts().sort_index().items():
        print(f"  {label:<10} : {count}")
    print()
    print("images per subtype:")
    sub = df.groupby(["label", "subtype"]).size().sort_index()
    for (label, subtype), count in sub.items():
        print(f"  {subtype:<3} ({label:<9}) : {count}")
    print()
    print("patients per label:")
    for label in sorted(patients_per_label.index):
        print(f"  {label:<10} : {int(patients_per_label[label])}")
    print()
    print(f"min images per patient          : {int(per_patient.min())} "
          f"(patient {per_patient.idxmin()})")
    print(f"max images per patient          : {int(per_patient.max())} "
          f"(patient {per_patient.idxmax()})")
    print()
    print("PATIENT/LABEL LEAKAGE CHECK (no patient_id under both labels):")
    if crossover:
        print(f"  FAIL - {len(crossover)} patient_id(s) appear under BOTH labels: {crossover}")
    else:
        print("  PASS - 0 patient_id values appear under both labels; "
              "every patient is exclusively benign or malignant.")
    print()
    print("sample image pixel dimensions (PIL):")
    for rel, w, h, mode in dims:
        print(f"  {w} x {h}  mode={mode}  {rel}")
    print("=" * 68)


def main() -> None:
    url = ensure_archive()
    extract_archive()
    df, n_files = build_manifest()
    report(df, n_files, url)


if __name__ == "__main__":
    sys.exit(main())
