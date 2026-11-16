# Phase 1 pre-registration

Written 2026-11-12 about 01:00 EST, before any grid encoder other than the corrected ImageNet ViT-S was evaluated.

## Already seen (exploratory, not confirmatory)
- Phikon, ImageNet ViT-S with the wrong input normalisation, colour histogram: label efficiency and leave-one-hospital-out (c17_labeleff3.json).
- Corrected ImageNet ViT-S: all-patient AUC and leave-one-hospital-out (c17_normfix_check.json).
- c17_diag.json: per-slide breakdown and pooled-across-hospital AUC. These motivated E3-E5 and stay exploratory.

## CAMELYON17 (49,532 tiles, 39 slides, 34 patients, 5 hospitals; ITC slides excluded)
Probe: the audit.logistic_probe recipe, unchanged. Unit of resampling: patient.
Encoders, each with the input normalisation of its own model config:
colour histogram; in_sup_vits; in_dino_vits; path_dino_vits_lunit; in_dino_vitb; in_sup_vitb; path_dino_vitb_kaiko; phikon (ViT-B, iBOT); phikon2 (ViT-L).

- E1 Label efficiency: budgets 1, 2, 4, 8, 16, all; 5 patient-disjoint splits (seeds 0-4) x 4 subset draws; identical subsets for every encoder.
- E2 All-patient out-of-fold: 5 repeats of grouped 5-fold, every patient tested once per repeat; AUC on pooled out-of-fold scores;
  95% interval from a patient-cluster bootstrap (B = 2000) on repeat 0, plus the spread over repeats.
- E3 Leave-one-hospital-out: per-hospital AUC, pooled AUC, sensitivity and specificity at the fixed threshold 0.5.

Primary contrasts (paired, same splits):
- C1 path_dino_vits_lunit - in_dino_vits (same architecture, same self-supervised method, pretraining data differs)
- C2 path_dino_vitb_kaiko - in_dino_vitb (same architecture, DINO, pretraining data differs)
- C3 phikon - in_dino_vitb (same architecture, iBOT vs DINO)
- C4 phikon - in_sup_vitb
Secondary: size within ImageNet DINO (B - S); phikon2 - phikon.

Claim rule: "pathology pretraining helps at matched size" is supported only if C1 and C2 are both positive with patient-bootstrap
95% intervals excluding 0 in E2, and positive at budgets 2 and 16 in at least 4 of 5 splits in E1. Otherwise: unresolved or negative.

- E4 Hospital shift, using no labels from the new hospital. Methods fixed in advance:
  M0 none; M1 per-hospital feature standardisation; M2 projection removing the span of training-hospital mean differences;
  M3 hospital- and class-balanced loss weights.
  Selection by inner leave-one-hospital-out inside the 4 training hospitals (pooled AUC); the selected method is applied to the held-out
  hospital. All four are reported. Success: Phikon's worst-hospital AUC rises without lowering the mean by more than 0.005.
- E5 Failure analysis: per-slide AUC for every encoder; tile montage for slides with AUC below 0.8.

Caveat fixed in advance: tiles are annotation-guided samples, so tumour prevalence per hospital (26-76%) is a sampling artefact.
Pooled and fixed-threshold numbers are tile-set metrics, not deployment metrics; those need exhaustive tiling (Phase 3).

## Spatial transcriptomics (within slide, 5-fold spatial-block cross-validation, folds as in st_cv.py)
- S1 Gene panels fixed in advance: all 256 genes; top 50 by variance of log1p(CP10K) on the training spots of each fold.
- S2 Baselines: training mean; kNN in feature space; coordinates-only kNN (k = 8); ridge on features.
- S3 Encoder ladder with a ridge head (PCA 256 + ridge, alpha by inner CV on training blocks): random conv and the eight encoders above.
- S4 Head ladder on Phikon: ridge, neighbourhood Transformer + MSE, neighbourhood Transformer + flow.
- S5 Noise ceiling: binomial thinning of raw UMI counts (p = 0.5), 10 draws; split-half reliability with Spearman-Brown;
  report r / sqrt(reliability) per gene.
- S6 Block-size sensitivity: 6 x 6 and 3 x 3 block grids.
- S7 Calibration: split-conformal on one calibration fold; coverage of 90% intervals on the test folds.
- S8 Replication of S2-S5 on the two other local slides (Visium V2 FFPE lymph node; breast cancer block A section 1).

Claim rules: "H&E predicts expression beyond spatial interpolation" needs FM + ridge above coordinates-only on median gene r in
5 of 5 folds on each slide. "Pathology pretraining helps" needs lunit above in_dino_vits and kaiko above in_dino_vitb on median r
in at least 4 of 5 folds.

## Amendments (each added before the affected analysis was run)
- 2026-11-12 01:06 EST, E4/M2: hospital means are class-balanced (average of the normal-tile mean and the tumour-tile mean,
  training labels only), so that the projection does not remove tumour signal through differing tumour prevalence.
- 2026-11-12 01:06 EST, E2: both the pooled out-of-fold AUC and the mean of per-fold AUCs are reported.
- 2026-11-12 01:41 EST, S7: each cross-validation model only predicts its own held-out blocks, so calibration and evaluation are
  two halves of those held-out blocks (alternating in sorted block order, then swapped). Score = |y - mean| / sd per gene.
  Calibration and evaluation blocks are different tissue regions, so the conformal guarantee is approximate; empirical coverage is reported.

# Phase 2 pre-registration: cross-patient spatial transcriptomics on HEST-Benchmark
Written 2026-11-12 20:30 EST, before any HEST number was computed.

- Data: public MahmoodLab/hest-bench, official splits (train_i.csv / test_i.csv, patient-stratified) and official gene lists (var_50genes.json).
- Gate (must pass before anything else is interpreted): with the official protocol (log1p counts, StandardScaler, PCA 256, ridge alpha = 100/(256*50),
  no intercept, mean Pearson over genes, mean over splits) Phikon must reproduce the public leaderboard value of each task within 0.02.
- Model under test: the neighbourhood Transformer + flow head, unchanged, with the training recipe fixed from the within-slide experiments
  (4000 steps, batch 64, d = 256, 4 layers, AdamW 3e-4, one-cycle). No tuning on HEST. Inference uses morphology only; no expression from test patients.
- Primary comparison: flow vs official ridge on the same Phikon features, per split (mean Pearson over the 50 genes).
  Success: flow above ridge in at least 6 of 7 splits over SKCM, LUNG, PAAD, and mean gain above 0.01. Reported regardless.
- Control: ridge on own features concatenated with the mean of the 8 nearest spots' features (context without the Transformer or the flow head).
- Further tasks (READ, COAD, IDC, LYMPH_IDC, PRAD, CCRCC) are added as disk allows and reported with the same rule; nothing is dropped after the fact.

## Phase 3 pre-registration: head-to-head with the public STFlow code (written 2026-11-13, before any STFlow run)

Question: on identical inputs, how does our fixed-recipe flow model compare with a published method of the same class?

- Code: github.com/Graph-and-Geometric-Learning/STFlow at commit 880c2ee, hyper-parameters of its README command
  (batch_size 2, n_layers 4, n_sample_steps 5, all other defaults, seed 1). Local patch = stflow_cmp/patch_train.py (device, phikon feature size,
  per-epoch test curve, early stop off); model, loss, sampler and optimiser untouched.
- Inputs: the same frozen Phikon embeddings, official HEST-Benchmark splits and 50 genes, target log1p(counts), score = mean Pearson over genes
  on pooled test spots. Tasks: LUNG, PAAD, SKCM, COAD, READ, LYMPH_IDC (everything downloaded).
- Two STFlow numbers per split, both reported:
  (a) released rule: best test-patient epoch with patience 20, recomputed from the curve;
  (b) fixed budget: epoch 100, no look at the test patients.
- Primary comparison: our flow model (4000 fixed steps, seed 0, no test-based selection) against (b), because both are selection-free.
  Secondary: against (a), which is how the released script reports.
- Reading rule, fixed now: "on par" if the mean over the six tasks differs by less than 0.01; "ahead"/"behind" if it differs by 0.01 or more
  AND the sign holds in at least 10 of 15 splits; otherwise "mixed". The size of (a) minus (b) is reported as the selection effect.
- Not claimed whatever happens: anything about STFlow's published numbers (those use UNI features we have no access to), or about one seed
  being the whole distribution.

### Phase 3 result, seed 1 (2026-11-13 04:24) and amendment written before any further run

- Result under the rule above: task means over six organs: ours 0.3888, STFlow epoch 100 0.3836, STFlow released rule 0.4365, ridge 0.3647.
  Ours minus STFlow at fixed budget +0.0052 (ours higher in 10 of 15 splits, 4 of 6 organs) -> reading: ON PAR.
  Selection effect (released rule minus epoch 100): +0.053 on task means, +0.055 over splits, largest +0.135.
- Post hoc, not pre-registered: STFlow's task-mean test score by epoch peaks at 0.410 (epoch 36) and is above ours for epochs 10 to 80;
  the script's default 100 epochs is past that peak. Our model's budget (4000 steps) was never varied, so its own curve is unknown.
- Amendment: robustness seeds with the identical protocol: STFlow seeds 2 and 3 (all six tasks); ours seeds 1 and 2 on COAD, READ, LYMPH_IDC
  (seeds 0 to 2 already exist for LUNG, PAAD, SKCM). The same reading rule is then applied to seed-averaged task means.

### Phase 3 result, three seeds per method (2026-11-13 11:45)

- Seed-averaged six-organ means: ours (seeds 0-2) 0.3857, STFlow epoch 100 (seeds 1-3) 0.3861, STFlow released rule 0.4380, ridge 0.3647.
  Per-seed six-organ means: ours 0.3888 / 0.3826 / 0.3857 (sd 0.003); STFlow epoch 100 0.3836 / 0.3880 / 0.3866 (sd 0.002).
- Reading rule on seed means: ours minus STFlow at fixed budget -0.0004, ours higher in 9 of 15 splits -> ON PAR (unchanged from seed 1).
  Per organ: LUNG -0.035, PAAD +0.011, SKCM -0.001, COAD +0.018, READ +0.006, LYMPH_IDC -0.001.
- Selection effect (released rule minus epoch 100): +0.052 on six-organ means, per seed +0.053 / +0.052 / +0.051; largest single run +0.146.
- Post hoc: seed-mean STFlow test curve peaks at 0.403 (epoch 36) and is above our mean for most epochs from 9 on. Our budget was never varied.
- Versus ridge on seed means: ours +0.021 (12 of 15 splits, 5 of 6 organs), STFlow epoch 100 +0.021 (10 of 15 splits, 4 of 6 organs).

## Phase 4 pre-registration: leak-free choice of the training budget for both models (written 2026-11-15, before any run)

Question: Phase 3 found parity at each method's default budget, but STFlow's test curve peaks before its default 100 epochs and our
budget was never varied. Does the comparison change when both budgets are chosen by one rule on data that are not the test patients?

- Validation data: in every training slide one contiguous strip holding a fifth of the spots is removed from training. Strips are five
  equal-count bins along the slide's longer coordinate axis; replicate r in {0, 1, 2} holds out strip {4, 2, 0}[r] (hest/val_strip.py).
  Both models train on the same remaining four fifths and are validated on the same strip, which each model sees as a separate small
  slide (own neighbour graph, image features only, no expression).
- Replicates: replicate r = our seed r and STFlow seed r + 1 (the seeds of Phase 3).
- STFlow: public code and settings of Phase 3; train_val.py is generated from the Phase 3 train.py by stflow_cmp/make_train_val.py
  (adds the validation strip, its curve and a thread cap; nothing else). Budget = epoch 1..100. Rule = the script's own rule
  (best epoch, patience 20, evaluated every epoch) pointed at the validation strip instead of the test patients.
- Ours: unchanged recipe (hest/hest_flow_val.py); budget = total optimiser steps in {250, 500, 1000, 2000, 4000, 8000}, each a full run
  with its own one-cycle schedule. Rule = the budget with the highest validation score.
- Score everywhere: mean Pearson over the 50 genes on pooled spots. Test patients are scored at every budget but read only at the selected one.
- Primary result (A): test score of the model trained on four fifths, at its validation-selected budget.
- Secondary (B): the selected budget re-used with all training spots (STFlow: Phase 3 curve of the same seed at the selected epoch;
  ours: Phase 3 run if 4000 steps were selected, otherwise one new full-data run).
- Reading rule, same as Phase 3, on replicate-averaged values of (A): on par if the six-organ means differ by < 0.01; ahead / behind if
  >= 0.01 and the sign holds in >= 10 of 15 splits; otherwise mixed.
- Reported whatever happens: the selected budgets; regret (test score at the best budget in hindsight minus test score at the selected
  budget) for both models; the per-organ table; (A) next to Phase 3's default-budget numbers.
- Sanity gates before any run counts: (1) train_val.py with validation switched off reproduces the Phase 3 seed-1 test curve of
  LUNG split 0 over the first 5 epochs (max abs difference < 1e-3); (2) hest_flow_val.py in full-data mode at 4000 steps reproduces the
  Phase 3 seed-0 score of LUNG split 0 within 0.01.
- Known limit, stated now: the strip comes from the training patients, so it measures generalisation within those patients. Two tasks
  (LUNG, SKCM) have a single training patient, so a held-out training patient is not available as an alternative there.

### Phase 4 result (2026-11-15 08:30), three replicates, both gates passed before the runs

- Gates: train_val.py with validation off reproduced the Phase 3 curve (max abs difference 3e-8); hest_flow_val.py in full-data mode
  reproduced 0.5404 exactly.
- Scheduling note, no effect on results: the GPU runs of our model and the CPU runs of STFlow slowed each other several-fold, so STFlow
  was paused (SIGSTOP) until our runs had finished and then resumed.
- Six-organ means, replicate-averaged. Default budget, all spots (Phase 3): ours 0.3857, STFlow 0.3861.
  (A) primary, four-fifths model at its validation-selected budget: ours 0.3967, STFlow 0.3925 -> difference +0.0041, ours higher in
  11 of 15 splits, 4 of 6 organs -> reading: ON PAR. By replicate: ours 0.4053 / 0.3927 / 0.3919, STFlow 0.3894 / 0.3970 / 0.3911.
  (B) secondary, selected budget with all training spots: ours 0.4052, STFlow 0.3903 -> +0.0148, 11 of 15 splits -> reading: AHEAD.
  By replicate: ours 0.4092 / 0.4007 / 0.4055, STFlow 0.3900 / 0.3919 / 0.3892 (ours higher in all nine pairings).
- Per organ, ours minus STFlow: (A) LUNG -0.015, PAAD +0.020, SKCM +0.002, COAD +0.012, READ +0.010, LYMPH_IDC -0.004;
  (B) LUNG -0.008, PAAD +0.036, SKCM +0.036, COAD +0.009, READ +0.025, LYMPH_IDC -0.010.
- Selected budgets: ours 2000 steps in 27 of 45 runs, 4000 in 15, 1000 in 2, 500 in 1, never 8000 or 250. STFlow epoch: median 57,
  quartiles 40 to 71, range 12 to 98. STFlow alternative rule (best validation epoch over all 100): 0.3912.
- Regret (best budget in hindsight on the test patients minus selected, four-fifths models): ours 0.017, STFlow 0.043. Hindsight
  values: ours 0.4148, STFlow 0.4318 (STFlow picks among 100 epochs, ours among 6 step counts).
- Gain from choosing the budget, all spots: ours +0.0195 over its default, STFlow +0.0043 over its default.
- Versus ridge with all spots: ours +0.040 (13 of 15 splits, 5 of 6 organs), STFlow +0.026 (10 of 15 splits, 4 of 6 organs).
- As stated beforehand: the validation strip comes from the training patients; in both models it favours a longer budget than the one
  that is best for new patients.
