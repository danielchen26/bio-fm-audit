#!/bin/sh
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
for t in "$@"; do $PY hest_flow.py --task $t > hest_flow_$t.log 2>&1; done
echo ALLDONE > hest_flow_ALLDONE
