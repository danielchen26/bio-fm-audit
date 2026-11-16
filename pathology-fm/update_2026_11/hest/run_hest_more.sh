#!/bin/sh
# approved scope: the six small HEST-Benchmark tasks. Adds READ, COAD, LYMPH_IDC (about 8 GB), same pre-registered rule.
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
HF_HUB_DISABLE_TELEMETRY=1 $PY hest_download.py READ COAD LYMPH_IDC > hest_download_more.log 2>&1
while [ ! -f hest_seeds_ALLDONE ]; do sleep 10; done
HF_HUB_DISABLE_TELEMETRY=1 $PY hest_embed.py > hest_embed_more.log 2>&1
$PY hest_ridge.py > hest_ridge_all.log 2>&1
for t in READ COAD LYMPH_IDC; do $PY hest_flow.py --task $t > hest_flow_$t.log 2>&1; done
echo ALLDONE > hest_more_ALLDONE
