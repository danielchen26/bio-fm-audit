#!/bin/sh
# How the corrected ImageNet rows (README section 5) were produced.
#
#   OLD   a clean checkout of this repository at commit 34c4032, the last one before the fix,
#         with pathology-fm/data holding patches96.npz, breakhis_manifest.csv, hest_sample.npz and hest_sample224.npz
#   FIX   the fix as a patch:  git diff 34c4032 HEAD -- pathology-fm/src/bench.py pathology-fm/src/c17bench.py \
#                                                      pathology-fm/src/c17site.py pathology-fm/src/xmodal.py > normfix.patch
#   OUT   where the six logs go (copies are in evidence/logs/imagenet_fix_*.log)
#
# Order matters: the as-released runs must first reproduce the published rows, otherwise the corrected ones mean nothing.
OLD=${1:?path to a checkout at 34c4032}; FIX=${2:?path to normfix.patch}; OUT=${3:-.}
PY=${PYTHON:-python}
OUT=$(cd "$OUT" && pwd); FIX=$(cd "$(dirname "$FIX")" && pwd)/$(basename "$FIX")
cd "$OLD/pathology-fm/src" || exit 1
$PY bench.py --encoders imagenet                        > "$OUT/bench_imagenet_asreleased.log" 2>&1
$PY xmodal.py --morph imagenet --src hest_sample.npz    > "$OUT/xmodal_imagenet_p96_asreleased.log" 2>&1
$PY xmodal.py --morph imagenet --src hest_sample224.npz > "$OUT/xmodal_imagenet_p224_asreleased.log" 2>&1
(cd "$OLD" && git apply "$FIX") || exit 2
$PY bench.py --encoders imagenet                        > "$OUT/bench_imagenet_fixed.log" 2>&1
$PY xmodal.py --morph imagenet --src hest_sample.npz    > "$OUT/xmodal_imagenet_p96_fixed.log" 2>&1
$PY xmodal.py --morph imagenet --src hest_sample224.npz > "$OUT/xmodal_imagenet_p224_fixed.log" 2>&1
