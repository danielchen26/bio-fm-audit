#!/bin/sh
# Phase 4, public STFlow code: replicate $1 (seed = replicate + 1) trains on four fifths of every training slide and logs the
# validation-strip and test curves of every epoch. Same settings as Phase 3; 4 CPU threads so three replicates can run side by side.
cd "$(dirname "$0")" || exit 1
R=$1; S=$((R + 1))
for T in LUNG SKCM PAAD COAD READ LYMPH_IDC; do
  echo "=== rep $R $T start $(date)"
  PYTHONPATH=STFlow env/bin/python STFlow/stflow/app/flow/train_val.py --datasets $T --feature_encoder phikon --seed $S --rep $R --threads 4 \
    --source_dataroot "$(cd ../hest && pwd)/bench_data" --embed_dataroot embed --val_dataroot embed_val \
    --save_dir results_val/rep$R --exp_code phikon_val --device cpu --batch_size 2 --n_layers 4 --n_sample_steps 5
  echo "=== rep $R $T done rc=$? $(date)"
done
echo ALLDONE
