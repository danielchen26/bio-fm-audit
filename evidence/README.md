# Evidence

Raw output backing the numbers quoted in the READMEs. Small enough to version, and
the point of keeping it: a claim is checkable against the run that produced it.

| file | what it backs |
|---|---|
| `verify_results.json` | section 0 of `pathology-fm/README.md` — the only quotable table. Carries snapshot id `9eff6989265d263e`, every claim's pre-registered null, value, SE and tier. |
| `logs/c17bench.log` | CAMELYON17 encoder x split-regime table, 5 seeds, per-centre AUC |
| `logs/frontier.log` | stain-jitter sweep with across-seed error bars and its power verdict |
| `logs/c17_download.log` | 131.8 GB transfer, 100/100 files size-verified, measured 17 MB/s |
| `logs/imagenet_fix_*.log` | the ImageNet correction of November 2026: BreaKHis and cross-modal ImageNet rows, as released (reproducing the published values) and with the input statistics fixed |
| `bench_results.npz` | BreaKHis 4-encoder x 3-regime results |
| `frontier.npz` | stain-jitter frontier rows |
| `hest_schema.md` | spatial-transcriptomics provenance, incl. the axis-swap control that validated the coordinate-to-pixel mapping (0/200 mismatches correct, 50/50 swapped) |

The result files of the November 2026 update are next to its code, in
`pathology-fm/update_2026_11/results/`. Absolute paths in the logs have been
replaced by relative ones.

Not versioned: 212 GB of slide data, 276 MB of checkpoints, and the cached
embeddings. All re-derivable by the scripts in this repository.
