"""Phase 4: generate train_val.py from the Phase 3 train.py. Adds (1) --rep / --val_dataroot: train on the four-fifths part and score a
validation strip of every training slide after each epoch, (2) the validation curve in curve.json, (3) --threads. With --rep -1
(default) the script is the Phase 3 script: same data, same order of random draws."""
P = "STFlow/stflow/app/flow/train.py"; s = open(P).read()
R = [("    train_dataset = MultiHESTDataset(sample_id_paths,",
      "    if args.rep >= 0:\n"
      "        for p in sample_id_paths:\n"
      "            p.h5_path = os.path.join(args.val_dataroot, args.dataset, f\"rep{args.rep}\", \"train\", f\"{p.name}.h5\")\n"
      "    train_dataset = MultiHESTDataset(sample_id_paths,"),
     ("    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=padding_batcher())\n",
      "    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=padding_batcher())\n"
      "    inner_loaders = []\n"
      "    if args.rep >= 0:\n"
      "        inner_loaders = [\n"
      "            torch.utils.data.DataLoader(\n"
      "                HESTDataset(\n"
      "                    HESTDatasetPath(name=p.name, h5_path=os.path.join(args.val_dataroot, args.dataset, f\"rep{args.rep}\", \"val\", f\"{p.name}.h5\"),\n"
      "                                    h5ad_path=p.h5ad_path, gene_list_path=p.gene_list_path),\n"
      "                    distribution=\"constant_1.0\", normalize_method=normalize_method, sample_times=1\n"
      "                ),\n"
      "                batch_size=1, collate_fn=padding_batcher()\n"
      "            ) for p in sample_id_paths\n"
      "        ]\n"),
     ("    curve = []\n", "    curve, inner_curve = [], []\n"),
     ("            val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)\n",
      "            if inner_loaders:\n"
      "                inner_curve.append(float(test(args, diffusier, model, inner_loaders)[\"all\"]['pearson_mean']))\n"
      "            val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)\n"),
     ("                json.dump({\"test_pearson_per_epoch\": curve}, f)\n",
      "                json.dump({\"test_pearson_per_epoch\": curve, \"validation_pearson_per_epoch\": inner_curve}, f)\n"),
     ("    parser.add_argument('--device', type=str, default='cpu')\n",
      "    parser.add_argument('--device', type=str, default='cpu')\n"
      "    parser.add_argument('--rep', type=int, default=-1)\n"
      "    parser.add_argument('--val_dataroot', type=str, default='embed_val')\n"
      "    parser.add_argument('--threads', type=int, default=0)\n"),
     ("    args = parser.parse_args()\n", "    args = parser.parse_args()\n    if args.threads > 0:\n        torch.set_num_threads(args.threads)\n")]
for a, b in R:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
open("STFlow/stflow/app/flow/train_val.py", "w").write(s); print("train_val.py written")
