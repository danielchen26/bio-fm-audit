#!/bin/sh
# Phase 4, our model: replicate $1 (seed = replicate), six budgets per split, budget selected on the validation strip.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
for T in LUNG SKCM PAAD COAD READ LYMPH_IDC; do $PY hest_flow_val.py --task $T --rep $1 > flowval_${T}_r$1.log 2>&1; done
echo ALLDONE > flowval_r$1_ALLDONE
