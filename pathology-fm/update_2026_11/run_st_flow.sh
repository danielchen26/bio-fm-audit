#!/bin/sh
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
$PY st_cv2.py --slide BRCA > st_cv2_BRCA.log 2>&1
$PY st_cv2.py --slide V2LN > st_cv2_V2LN.log 2>&1
echo ALLDONE > st_flow_ALLDONE
