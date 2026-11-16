#!/bin/sh
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
while ! grep -q "^done" st_build.log; do sleep 5; done
HF_HUB_DISABLE_TELEMETRY=1 $PY st_embed.py > st_embed.log 2>&1
for s in V1LN BRCA V2LN; do $PY st_eval.py $s > st_eval_$s.log 2>&1 & done
wait
echo ALLDONE > st_ALLDONE
