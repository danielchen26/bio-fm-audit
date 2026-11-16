"""Environment compatibility only: with current scipy, pearsonr on float32 arrays returns numpy float32, which json cannot write.
Both json.dump calls of the released train.py get default=float. No numbers change."""
P = "STFlow/stflow/app/flow/train.py"
s = open(P).read()
a = "json.dump(dataset_res, f, sort_keys=True, indent=4)"; b = "json.dump(kfold_results, f, sort_keys=True, indent=4)"
assert s.count(a) == 1 and s.count(b) == 1
s = s.replace(a, "json.dump(dataset_res, f, sort_keys=True, indent=4, default=float)").replace(b, "json.dump(kfold_results, f, sort_keys=True, indent=4, default=float)")
open(P, "w").write(s); print("patched")
