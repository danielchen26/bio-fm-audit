#!/bin/sh
# Public STFlow code on our Phikon features, official HEST splits, README hyper-parameters. One task after another.
cd "$(dirname "$0")" || exit 1
for T in "$@"; do
  echo "=== $T start $(date)"
  PYTHONPATH=STFlow env/bin/python STFlow/stflow/app/flow/train.py --datasets $T --feature_encoder phikon \
    --source_dataroot "$(cd ../hest && pwd)/bench_data" --embed_dataroot embed \
    --save_dir results --exp_code phikon_h2h --device cpu --batch_size 2 --n_layers 4 --n_sample_steps 5
  echo "=== $T done rc=$? $(date)"
done
echo ALLDONE
