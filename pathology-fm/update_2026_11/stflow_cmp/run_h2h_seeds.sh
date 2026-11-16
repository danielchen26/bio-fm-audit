#!/bin/sh
# Robustness seeds for the public STFlow code: same protocol as run_h2h.sh, only --seed and the output folder change.
cd "$(dirname "$0")" || exit 1
for S in 2 3; do
  for T in LUNG SKCM PAAD COAD READ LYMPH_IDC; do
    echo "=== seed $S $T start $(date)"
    PYTHONPATH=STFlow env/bin/python STFlow/stflow/app/flow/train.py --datasets $T --feature_encoder phikon --seed $S \
      --source_dataroot "$(cd ../hest && pwd)/bench_data" --embed_dataroot embed \
      --save_dir results_s$S --exp_code phikon_h2h --device cpu --batch_size 2 --n_layers 4 --n_sample_steps 5
    echo "=== seed $S $T done rc=$? $(date)"
  done
done
echo ALLDONE
