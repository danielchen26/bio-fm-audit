"""Download selected HEST-Benchmark tasks (public, ungated dataset MahmoodLab/hest-bench) into ./bench_data."""
import sys, os
from huggingface_hub import snapshot_download
tasks = sys.argv[1:]
p = snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type="dataset", local_dir="bench_data",
                      allow_patterns=[f"{t}/*" for t in tasks] + [f"{t}/**" for t in tasks], max_workers=8)
print("downloaded to", p, flush=True)
for t in tasks:
    n = sum(len(f) for _, _, f in os.walk(os.path.join(p, t)))
    print(t, n, "files", flush=True)
print("done", flush=True)
