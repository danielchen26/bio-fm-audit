#!/bin/sh
# launch the pre-registered evaluation for each encoder as soon as its embedding exists
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python}
for n in phikon raw in_sup_vits in_dino_vits path_dino_vits_lunit in_dino_vitb in_sup_vitb path_dino_vitb_kaiko path_dinov2_vitl_phikon2; do
  ( if [ "$n" != phikon ] && [ "$n" != raw ]; then while [ ! -f c17_emb_$n.npy ]; do sleep 5; done; fi
    $PY c17_grid_eval.py $n 2 > c17_grid_$n.log 2>&1 ) &
done
wait
echo ALLDONE > c17_grid_ALLDONE
