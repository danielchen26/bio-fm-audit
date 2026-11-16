"""Local patch of the public STFlow training script (commit 880c2ee) so it runs on our Phikon features on this machine
and records the test curve. Model, loss, sampler, optimiser and hyper-parameters are untouched.
1. --device accepts a string (cpu) instead of a CUDA index.
2. feature_dim table gains phikon: 768.
3. Test Pearson of every epoch is written to curve.json (the released script only keeps the best one).
4. Early stopping on the test patients is switched off, so every run trains the script's default 100 epochs.
   The released rule (best test epoch, patience 20) is recomputed afterwards from curve.json."""
import os
import shutil
P = "STFlow/stflow/app/flow/train.py"
shutil.copy(P, P + ".orig")
s = open(P).read()
R = [("parser.add_argument('--device', type=int, default=0)", "parser.add_argument('--device', type=str, default='cpu')"),
     ('        "uni_v1_official": 1024,\n', '        "uni_v1_official": 1024,\n        "phikon": 768,\n'),
     ("    best_pearson, best_val_dict = -1, None\n", "    best_pearson, best_val_dict = -1, None\n    curve = []\n"),
     ("            val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)\n",
      "            val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)\n"
      "            curve.append(float(val_perf_dict[\"all\"]['pearson_mean']))\n"
      "            with open(os.path.join(val_save_dir, 'curve.json'), 'w') as f:\n"
      "                json.dump({\"test_pearson_per_epoch\": curve}, f)\n"),
     ("                if early_stop_step >= 20:", "                if early_stop_step >= 10 ** 9:")]
for a, b in R:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open(P, "w").write(s)
print("patched")
