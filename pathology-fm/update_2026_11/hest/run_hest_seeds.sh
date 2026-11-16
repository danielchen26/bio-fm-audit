#!/bin/sh
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
for s in 1 2; do for t in LUNG PAAD SKCM; do $PY hest_flow.py --task $t --seed $s > hest_flow_${t}_s$s.log 2>&1; done; done
echo ALLDONE > hest_seeds_ALLDONE
