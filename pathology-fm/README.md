# Computational-pathology foundation models: what survives an honest split

A small, fully reproducible study of two questions that decide whether a pathology
foundation model is worth anything:

1. **How much of a reported score is leakage?** Measured, not assumed.
2. **Does the representation encode biology, or the slide it came from?** Probed
   directly, with the confound in the same table as the biology.

Everything below is measured on this machine (Apple M4 Max, MPS) from public data.
No number is estimated.

---

## 0. Verified results — snapshot `9eff6989265d263e`

**Read this section first. It supersedes every number in sections 6 and 7.**

The individual experiments below were each run when they were written, against
whatever data existed at the time: 38 slides, then 41, then 50, then 56, now 65
with all negatives. Each analysis is individually defensible and none of them are
comparable to each other. That is a defect in the study, not in any one
experiment, and `src/verify.py` is the fix: one frozen snapshot hashed by content,
one seed policy (0-4), claims and their nulls **pre-registered before computing**,
and every result assigned a confidence tier rather than quoted as a bare number.

Snapshot: **52,155 tiles / 65 slides / 45 patients / 5 centres**, 50.3% tumour,
tiles per grade `{negative 6000, itc 6423, micro 7825, macro 31907}`.

| claim | value | 2·SE | n | tier |
|---|---|---|---|---|
| `leak_raw` — random minus patient-disjoint AUC | **+0.1179** | 0.1018 | 5 | RESOLVED |
| `leak_phikon` — same, Phikon | **+0.0199** | 0.0111 | 5 | RESOLVED |
| `centre_id` — hospital decodable above chance | **+0.6984** | 0.0018 | — | RESOLVED |
| `encoder_gap` — Phikon minus raw-pixel, patient-disjoint | **+0.1852** | 0.0902 | 5 | RESOLVED |
| `delta_site` — paired seen minus unseen site | **+0.0077** | 0.0136 | 5 | **NULL** |
| `stability` — across-centre AUC SD, raw minus Phikon | **+0.1968** | 0.0957 | 5 | RESOLVED |
| `rule_kappa` — AJCC rule on annotated polygons | **+0.9875** | 0.0190 | 65 | RESOLVED |
| `tile_kappa` — AJCC rule on predicted tiles | **+0.6741** | 0.1300 | 65 | RESOLVED |
| `seg_value` — rule_kappa minus tile_kappa | **+0.3133** | 0.1313 | 65 | RESOLVED |

### The three findings that survive

**1. A good pathology FM essentially solves site shift on this task; a colour
baseline does not.** `delta_site` is a genuine NULL: +0.0077 with 2·SE 0.0136,
and that bound sits below the 0.02 effect size pre-registered as mattering.
Per-fold values flip sign (+0.018, -0.003, +0.028, +0.006, -0.010) with matched
test sets on all five folds. But the same is emphatically not true of the
baseline:

| | across-centre AUC SD | worst hospital |
|---|---|---|
| raw-pixel | 0.2252 | **0.3750** (below chance) |
| Phikon | **0.0284** | 0.9117 |

So site robustness is a property of the representation, not of the task. Reporting
only a mean would have missed this entirely; `stability` is RESOLVED at +0.1968.

**2. Leakage is real but encoder-dependent.** raw-pixel loses +0.1179 AUC moving
from a random to a patient-disjoint split; Phikon loses +0.0199. A weak encoder
leans on slide-specific colour and therefore leaks far more.

**3. Every encoder is a hospital fingerprint.** Phikon identifies which of five
hospitals a patch came from with **99.77%** accuracy against chance 29.9%. Best
biology and heaviest confound in the same model — the same coupling found on
BreaKHis with patient identity, reproduced here with a different confound
variable.

### Numbers this corrected

Re-running on one snapshot changed four figures I had previously reported:

| quantity | earlier claim | verified | cause of the error |
|---|---|---|---|
| value of segmentation | 0.64 kappa | **+0.3133** | stage A measured on 50 slides, stage B on the *itc+micro subset only* — **different denominators** |
| `delta_site` | -0.028, underpowered | **+0.0077, NULL** | matched test sets + size-equalised training + all 65 slides |
| Phikon leak | +0.072 | **+0.0199** | negative slides had not been downloaded |
| raw-pixel leak | +0.263 | **+0.1179** | different slide set |

The 0.64 figure was the worst: it differenced two kappas computed over different
slide populations and presented the result as an effect size. That is exactly the
failure this harness exists to prevent, and it caught it on its author.

### Two statistical errors the harness caught in itself

The first run flagged `leak` and `encoder_gap` as UNDERPOWERED. Both were
artefacts of my own SE computation:

- **SD used where SE belonged** — missing the `/sqrt(n)`.
- **Unpaired comparison of paired measurements** — both encoders are scored on
  identical splits, so split-to-split variance is common and cancels under
  differencing. Using marginal means inflated 2·SE from 0.090 to 0.317 and buried
  a real effect.

After correcting both, all nine claims carry a tier and none are underpowered.

---

## 1. Data and splits

| | |
|---|---|
| Dataset | BreaKHis v1 (Spanhol et al.), breast histopathology, H&E |
| Images | **7909**, uniformly 700x460 RGB |
| Patients | **81 distinct `patient_id`** across 82 slide directories |
| Magnifications | 40x (1995) / 100x (2081) / 200x (2013) / 400x (1820) |
| Labels | benign 2480 (24 patients) / malignant 5429 (57 patients) |
| Paired ST sample | 10x Visium Human Lymph Node, 4035 spots, 36601 genes |

All counts reproduce the published BreaKHis figures exactly.

**A trap worth naming.** The literature says "82 patients". There are 82 slide
*directories* but only **81 distinct patients**: `14-13412` appears twice, under
two different subtypes (`SOB_M_DC_14-13412`, `SOB_M_LC_14-13412`). Splitting by
slide or by subtype directory leaks that patient across the boundary. Splits here
key on `patient_id`.

Three regimes, verified for disjointness rather than assumed:

| regime | train/val/test | patients shared train<->test |
|---|---|---|
| `random` (leaky by design) | 5515 / 1183 / 1211 | **81 of 81** |
| `patient` (honest) | 5354 / 1122 / 1433 | **0** |
| `magnification` (domain shift, 40+100x -> 400x) | 2997 / 1439 / 513 | **0** |

---

## 2. Main result: the leak is 0.084-0.150 AUC

Benign/malignant AUC from a linear probe on frozen embeddings.

| encoder | random (leaky) | **patient (honest)** | magnif (shift) | **leak** | patient-ID lift | MIL patient-AUC |
|---|---|---|---|---|---|---|
| raw-pixel (colour histogram) | 0.878 | 0.794 | 0.632 | **+0.084** | 13.4x | 0.882 +- 0.102 |
| ImageNet ViT-S/16 | 0.951 | 0.801 | 0.766 | **+0.150** | 16.9x | 0.938 +- 0.020 |
| **Phikon ViT-B** (iBOT on TCGA) | 0.993 | **0.902** | **0.941** | +0.091 | **20.0x** | **0.977 +- 0.021** |
| ours-DINO (2500 steps, 37 min, 3.4M) | 0.943 | 0.837 | 0.865 | +0.106 | 15.8x | 0.900 +- 0.050 |

Three things fall out.

**A colour histogram matches ImageNet.** raw-pixel 0.794 vs ImageNet ViT-S 0.801.
Indistinguishable. On this task ImageNet pretraining contributes nothing, and any
paper reporting only its own model would never surface that. The trivial baseline
belongs in the table.

**Pathology-specific pretraining does pay off.** Phikon reaches 0.902 honest AUC,
+0.108 over the colour histogram, and is the most robust under magnification shift
(0.941). Note this is the opposite of the cross-modal result in section 4 -- FM
value is task-dependent and cannot be asserted in general.

**raw-pixel collapses under domain shift** (0.632). Colour statistics do not
transfer across magnification, which confirms the shift regime is a real test.

---

## 3. The confound audit

Same frozen embeddings, three probes. The biology number alone is not
interpretable without the confound next to it.

Phikon identifies **which of 55 patients** a single patch came from with
**96.9% accuracy** (chance 4.85%, 20.0x lift). It has the best biology *and* the
heaviest slide-identity signal. Our DINO, trained with HED stain jitter, carries
the least identity among learned encoders (15.8x) at lower biology (0.837).

That is a trade-off curve, not a defect, and it is what `frontier.py` maps.

---

## 4. Cross-modal extension: morphology <-> spatial transcriptome

Same architecture as a multimodal protein model (one trunk, per-track masking,
whole-track dropout for any-to-all conditioning); modality pair swapped for
H&E patch <-> gene expression <-> spatial position. Spatially blocked split:
whole tiles of the slide held out, because adjacent Visium spots are ~100 um
apart and leak exactly the way patients leak.

### Finding 1: the metric determines the conclusion

`morph+ctx`, 96px patches, same held-out tiles:

| encoder | head | CRPS (lower=better) | **r(mean)** | cov90 | sharpness |
|---|---|---|---|---|---|
| — | neighbour-Gaussian (smoothing) | 0.6993 | +0.110 | 0.700 | 0.833 |
| — | neighbour-mean (point) | 0.9080 | +0.114 | 0.004 | 0.000 |
| phikon | **MSE** | 0.7930 | **+0.378** | 0.000 | 0.000 |
| phikon | **flow** | **0.5766** | +0.376 | 0.817 | 0.861 |

On `r(mean)` the MSE model beats naive smoothing by **3.5x**. On CRPS -- a strictly
proper scoring rule -- the *same model* is **worse than smoothing**. `cov90 = 0.000`
explains it: MSE returns a zero-variance point forecast, and a proper score
punishes overconfidence.

The cause is structural, not incidental: the MSE optimum is `E[x|c]`, and the
conditional mean of a spatially autocorrelated field *is* the smoothed field. An
MSE head cannot demonstrate morphology contributing anything beyond smoothing even
if morphology fully determines the conditional distribution.

**Rectified flow matching is strictly dominant**: identical point accuracy
(+0.376 vs +0.378) with calibrated spread instead of none.

### Finding 2: field of view was my bug, not the model's

Initial patches were 96px at 0.6146 um/px = **59 um**, upsampled to 224 for a
ViT/16. Phikon was pretrained near 128-256 um. Re-cropping natively at 224px
(**138 um**, 4035/4035 spots retained):

| encoder | morphology-kNN r @59um | @138um | `morph-only` CRPS @59um | @138um |
|---|---|---|---|---|
| random conv | +0.1219 | +0.1219 | 0.6251 | 0.6233 |
| ImageNet | +0.0994 | +0.1121 | 0.6218 | 0.6115 |
| **Phikon** | +0.1082 | **+0.1543** | 0.6146 | **0.6088** |

At native scale Phikon's retrieval improves **43%** and the ordering becomes
phikon > imagenet > random, which it was not at 59 um. Reading a pathology FM out
of its training scale looks exactly like the FM being useless.

### Finding 3: the regime decides whether the encoder matters

- **Neighbours measured** -> spatial autocorrelation dominates; morphology's
  marginal contribution is ~0.003-0.005 CRPS *regardless of encoder*, and a random
  conv matches Phikon. Any per-spot vector buys this much.
- **Neighbours unmeasured** (`morph-only` -- the actual deployment condition on a
  new slide) -> encoder quality orders monotonically and the random->Phikon gap
  *widens* at native scale (0.0105 -> 0.0145).

---

## 5. Stain-augmentation sweep: the predicted trade-off does not exist

HED stain-jitter strength swept during SSL pretraining, matched budget (1200
steps, 256 prototypes). Every point evaluated on **5 independent patient-disjoint
splits**, because a single split at 14 test patients cannot separate signal from
noise.

| `hed_sigma` | bio-AUC (5 seeds) | patient-ID acc | pid lift | magnif |
|---|---|---|---|---|
| 0.00 | 0.8224 +- **0.0381** | 0.5717 +- 0.0575 | 11.4x | 0.5356 |
| 0.10 | 0.8443 +- **0.0272** | 0.6204 +- 0.0440 | 12.4x | 0.5973 |
| 0.35 | **0.8699** +- **0.0131** | 0.5888 +- 0.0214 | 11.8x | 0.5783 |

This experiment was built to map an invariance/utility frontier: stronger jitter
should destroy slide identity at some cost to biology. **Neither half happened.**

- Biology rose **monotonically**, +0.0475 AUC from no jitter to the strongest
  tested.
- Patient-ID decodability stayed **flat and non-monotone** within error bars
  (11.4x -> 12.4x -> 11.8x). No confound was traded away.

So there is no frontier in `[0, 0.35]`; stain jitter is free utility over this
range, and the operating point is simply "the strongest setting tested". Finding
where it starts to hurt requires going past 0.35, which this sweep did not do.

### The result I did not design for

The across-seed standard deviation falls monotonically with jitter strength:

```
sigma 0.00 -> SD 0.0381
sigma 0.10 -> SD 0.0272
sigma 0.35 -> SD 0.0131      (-66% vs no jitter)
```

Stain augmentation makes the encoder roughly **3x less sensitive to which
patients happen to land in the training split**. For deployment that matters more
than the mean: it is the difference between a model whose score depends on cohort
luck and one that does not. Caveat: an SD estimated from 5 seeds carries about
35% relative uncertainty (`SD/sqrt(2(n-1))`), so a 3x reduction is suggestive
rather than established.

### Honest power verdict

```
[power] mean across-seed SD 0.0261 vs across-sigma spread 0.0475
        -> NOT RESOLVABLE at this n
```

The script's own criterion is `spread > 2 x SD` = 0.052. The observed spread is
0.0475 -- **just under**. So despite a clean monotone trend across three points,
this benchmark cannot establish the effect at n = 81 patients. The monotonicity is
encouraging and the direction is consistent, but the correct statement is
"suggestive, underpowered", not "demonstrated".

Settling it needs multi-centre data where stain variation is the actual
experimental variable -- CAMELYON17 (5 hospitals) or TCGA stratified by tissue
source site -- not more seeds on one cohort.

---

## 6. CAMELYON17: whole-slide scale and the multi-centre confound

> **Numbers in this section are from an earlier 38-41 slide snapshot without
> negative slides, and are superseded by section 0 (snapshot `9eff6989265d263e`).** The methodology, the traps
> documented, and the three broken `delta_site` designs are all still valid and
> are the reason this section is kept.


BreaKHis is one cohort, one lab, one scanner, and pre-cropped 700x460 microscope
fields. CAMELYON17 adds both missing dimensions: 19-gigapixel pyramidal slides,
and five contributing hospitals so that CENTRE becomes an experimental variable.

| | |
|---|---|
| Slides | **50** lesion-annotated, **exactly 10 per centre**, grades itc 16 / micro 17 / macro 17 |
| Level 0 | 96651 x 197226, 9 pyramid levels, 0.243 um/px (40x) |
| Working level | level 1 = 0.486 um/px, so a 224px tile spans **109 um** -- inside the band pathology FMs are pretrained on |
| Tiles | **46,155** (26,255 tumour) |
| Transfer | 131.8 GB in 184 min at 17 MB/s (link-saturated: 8 concurrent streams gave the same aggregate) |

### Two label-corrupting traps, both silent

**The distributed `masks/*.tif` are TISSUE masks, not tumour masks.** Measured:
every tissue tile scored >= 0.5 coverage, mean 0.995, and mask-nonzero area
(11.5% of slide) tracks Otsu tissue area (8.2%). Using them as tumour labels
makes **100% of tiles positive** -- a dataset on which any classifier reports
near-perfect AUC. Real lesions are ASAP polygons in `annotations/*.xml`.

**A coverage threshold of 0.5 silently zeroes out every ITC slide.** Annotated
lesion area spans five orders of magnitude, 116.25 mm2 down to 0.0010 mm2, while
a tile is 0.0119 mm2. So the smallest ITC covers 8% of one tile, and the highest
positive coverage observed on an ITC slide was **0.482**. At `pos_thr=0.5`, 16 of
50 slides contribute zero signal while training loss looks entirely normal.
Fixed with lesion-guided sampling: positives are enumerated from the polygons
(bbox-overlapping grid tiles plus one centred on each lesion centroid), so every
lesion yields a positive regardless of grid phase.

A third trap was avoided by parsing rather than grepping: `grep PartOfGroup`
reports 51 `None` groups, which are `<Group>` header definitions in ASAP's
`<AnnotationGroups>`, not annotations. Actual annotations are 649 `Tumor` and
1 `Exclusion`.

### Leakage is worse at WSI scale

Tumour-tile AUC, linear probe on frozen embeddings, grade-controlled to
micro+macro, 5 seeds:

| encoder | random | slide | **patient** | centre (LOCO) | leak | **centre-ID** |
|---|---|---|---|---|---|---|
| raw-pixel | 0.9117 +-.003 | 0.5993 +-.141 | 0.6487 +-.099 | 0.7073 +-.081 | **+0.263** | 0.9615 |
| ImageNet ViT-S | 0.9732 +-.002 | 0.8066 +-.068 | 0.8457 +-.039 | 0.8500 +-.086 | **+0.128** | 0.9661 |
| **Phikon ViT-B** | 0.9973 +-.000 | 0.8827 +-.061 | **0.9250 +-.031** | 0.9595 +-.016 | +0.072 | **0.9958** |

The random-split leak reaches **+0.263 AUC** -- far worse than BreaKHis's
+0.084-0.150, because 400 tiles drawn from one 19-gigapixel slide are near
duplicates of each other. Weaker encoders leak more: they lean harder on
slide-specific colour.

**Phikon identifies which of five hospitals a patch came from with 99.58%
accuracy** (chance 33.3%). It has simultaneously the best biology (0.9250) and
the heaviest confound. That is the same trade-off found on BreaKHis, where
Phikon had the best biology and the heaviest patient-ID signal (20.0x) --
**reproduced on a second dataset with a different confound variable.** This is
the strongest claim in the project.

### Site shift: three broken designs before a clean one

The target quantity was `delta_site = AUC(seen site) - AUC(unseen site)`. Three
attempts each produced publishable-looking numbers and each was wrong:

| version | confound | symptom |
|---|---|---|
| v1 | lesion grade | `slide` split scored 0.557 vs `patient` 0.835 -- impossible, since patient-disjoint is strictly stronger |
| v2 | test-set size and prevalence | delta_site came out **negative** for all encoders: "unseen hospitals are easier" |
| v3 | training-set size | matching a large target starved training: 3,683 tiles on one fold vs 21,462 on another |
| v4 | none found | match 1.00 on all folds, train n=6000+-1, test n=1500+-1, prevalence and grade identical per fold |

v4 result, one model scored on two matched test sets differing only in site novelty:

| encoder | AUC seen | AUC unseen | delta_site | across-fold SD | **worst fold** |
|---|---|---|---|---|---|
| raw-pixel | 0.6938 | 0.6295 | +0.064 +- **0.341** | 0.341 | **0.2341** |
| Phikon | 0.8982 | 0.9257 | -0.028 +- 0.065 | 0.065 | 0.7840 |

**delta_site is not measurable here.** Resolving an effect against an across-fold
SD of 0.0645 over 5 folds needs `|delta| > 2*SD/sqrt(5) = 0.058`; the observed
0.028 is below that, and its sign is negative, which is physically implausible as
a real effect. The correct statement is "this design cannot detect it at 50
slides and 5 centres", not "site shift is free".

### The signal is in the variance, not the mean

What *is* resolvable is stability:

```
across-fold SD    raw-pixel 0.3414  vs  Phikon 0.0645   -> 5.3x more stable
worst hospital    raw-pixel 0.2341  vs  Phikon 0.7840   -> +0.550 AUC
```

raw-pixel scores **below chance** on one held-out hospital: the colour feature
that separates tumour in four hospitals actively anti-predicts in the fifth. That
is consistent with centre being 96% decodable from colour alone, and it is a
direct measurement of why stain normalisation exists.

A deployment decision depends on the worst hospital, not the average. So the
useful metric here is worst-fold AUC and across-centre SD -- both resolvable --
rather than the mean shift this experiment was originally designed to measure.

---

## 7. Clinical endpoint: slide grading, and what it proves about segmentation

> **Kappa figures here are from a 50-slide snapshot without negatives and are
> superseded by section 0**, which puts both stages on one snapshot and gives
> `seg_value = +0.3133` rather than the 0.64 quoted below. The two-stage validation
> design and the falsified resolution hypothesis stand.


CAMELYON17's clinical target is pN stage, and it is defined by **metastasis size**,
not by a learned score:

| longest axis | grade | contributes |
|---|---|---|
| none | negative | pN0 |
| < 0.2 mm | isolated tumour cells | pN0(i+) |
| 0.2 - 2.0 mm | micro-metastasis | pN1mi |
| > 2.0 mm | macro-metastasis | pN1 (1-3 nodes) / pN2 (4-9) |

So the grader is that rule applied to predicted tumour geometry -- interpretable
and auditable by construction. Validated in two stages, because a rule can be
wrong too.

### Stage A: the rule, with no model in the loop

`rule(annotated polygons)` vs the published `stages.csv` grades:

| true grade | n | measured longest axis | AJCC definition |
|---|---|---|---|
| itc | 16 | median 0.116, range 0.033-**0.342** | < 0.2 mm |
| micro | 17 | median 0.452, range **0.208**-1.289 | 0.2 - 2.0 mm |
| macro | 17 | median 3.173, range **2.212**-10.555 | > 2.0 mm |

**quadratic kappa 0.9687, accuracy 0.960**, only 2 errors and both ITC->micro
adjacent. The measured geometry lands exactly on the AJCC thresholds, so the rule
and the millimetre arithmetic are correct and stage B is interpretable.

### Stage B: the rule on predicted tiles, centre-disjoint

Tile-level tumour AUC per held-out hospital: 0.987 / 0.984 / **0.849** / 0.982 /
0.980. The classifier finds the tumour tiles. Slide grading, all 50 slides:

| | pred itc | pred micro | pred macro |
|---|---|---|---|
| true itc (16) | **3** | **13** | 0 |
| true micro (17) | 0 | **17** | 0 |
| true macro (17) | 0 | 1 | **16** |

**quadratic kappa 0.7280.** micro and macro are essentially solved; every error
is ITC. That is a measurement-resolution failure, not a classification failure:
tile pitch is 0.109 mm, so 1 tile = 0.109 mm and 2 tiles = 0.218 mm, and the
0.2 mm ITC boundary falls *between* those quantisation levels.

### The resolution hypothesis, tested and falsified

Stride was decoupled from field of view -- tiles stay 109 um (Phikon's
pretraining scale) but are sampled at 1/4 stride, giving 0.027 mm localisation.
On the itc+micro subproblem:

| | coarse (0.109 mm pitch) | fine (0.027 mm pitch) |
|---|---|---|
| fixed AJCC thresholds | kappa 0.3084, itc 3/16 | kappa **0.1283**, itc 2/16 |
| thresholds calibrated per fold | -- | kappa 0.3320, itc **11/16**, micro **7/17** |

Finer stride made the fixed-threshold version **worse**: a tile is called positive
when a lesion covers 5% of it, so the positive footprint is a *dilation* of the
lesion by up to one tile width -- comparable to the 0.2 mm threshold itself, and
oversampling amplifies it. Calibration then merely traded ITC errors for micro
errors, leaving kappa unchanged (0.308 -> 0.332).

The fitted cut points give the reason:

```
t1 across folds: 11.4 / 16.6 /  9.5 / 11.2 /  7.8 grid units  = 0.213-0.453 mm
t2 across folds: 16.6 / 63.0 / 63.0 / 63.0 / 63.0             = unstable
```

**There is no stable span-to-grade mapping across hospitals.** The positive-tile
footprint reflects classifier behaviour -- decision threshold, per-centre AUC
0.849-0.987 -- more than lesion geometry. The ITC boundary is not recoverable
from tile classification at any stride tested.

### What that measures

Stage A shows the grade IS recoverable from true lesion geometry (kappa 0.9687).
Stage B shows it is not recoverable from tile classification (kappa 0.33 on the
ITC-containing subproblem).

```
kappa with true segmentation   0.9687
kappa with tile classification 0.33
                               ------
value of a segmentation stage   0.64 kappa
```

That 0.64 gap is a direct measurement of why "segmentation driven, cell-centric
pipelines" is its own line item in this job description. Tile classification is
sufficient for micro and macro disease; sub-0.2 mm disease needs actual instance
segmentation. This project does not implement that stage -- it quantifies its
worth.

### pN staging: pipeline present, not scoreable here

The official patient rule is implemented and runs, but pN requires all five nodes
of a patient and only the lesion-annotated subset was downloaded: 43 patients
with nodes-per-patient of `{1: 37, 2: 5, 3: 1}`. pN output is therefore a
pipeline demonstration on partial node sets, explicitly not a scored result.
---

## 8. Negative results and errors caught

The most transferable part of this study.

**The training curve does not predict representation quality, in either
direction.** Three instances:

| run | loss / dynamics | representation |
|---|---|---|
| protein VQ tokenizer | loss 3.69->1.78, codebook 512/512 used, perplexity 444 | **useless**: 0.55 of 2.39 bits of SS8 entropy (23%) |
| DINO, 4096 prototypes | loss flat 7.03->6.86, teacher drifting to uniform | **usable**: 0.837, beats colour histogram |
| DINO, 256 prototypes | loss genuinely falls 5.05->3.95, teacher structures | **no better**: 0.848 (+0.011, within noise), and -0.049 under shift |

Sophisticated collapse diagnostics (`teacher_entropy`, `kl_uniform`, `top1_share`)
correctly described the *dynamics* every time and predicted downstream quality
**never**. Only parameter-free probes on frozen features did.

**Errors I made and caught:**

- *Coordinate leakage in the multimodal design.* Geometric attention was fed true
  coordinates on every forward pass, so `sequence -> structure` evaluation would
  have read the answer off the input. Fixed by gating the attention bias with the
  structure mask.
- *A null hypothesis of zero.* An auto-verdict declared "morphology helps" whenever
  the delta beat zero -- and duly declared it for **random features**. The null for
  "our encoder learned something" is a random-initialised control, never zero. The
  auto-verdict is now removed rather than fixed: it was worse than no verdict.
- *Comparing across training budgets.* A 2500-step model looked worse than a
  6000-step model, which I nearly read as "my fix is harmful". A matched-budget
  control reversed it: at 2500 steps both, the fix gives +0.06 bits SS8 (+17%).
  But raising steps 2500->6000 gives +0.19 bits -- **3x the effect of the fix**, so
  the diagnosis was a second-order correction, not the bottleneck.
- *Silent feature-cache collision.* The morphology cache keyed on encoder name
  only, so a 96px run's features would have been reused for the 224px run,
  voiding the entire field-of-view experiment. Cache key now includes patch size.
- *Trusting a distributed mask file's name.* `masks/*.tif` are tissue masks, not
  tumour masks; using them made 100% of tiles positive, which any metric would
  have reported as a triumph.
- *A coverage threshold that silently deleted a whole stratum.* `pos_thr=0.5`
  gives every ITC slide zero positive tiles -- 16 of 50 slides contributing
  nothing, with a normal-looking loss curve.
- *Auditing structure with grep.* `grep PartOfGroup` reported a phantom `None`
  class that was really ASAP's `<Group>` header rows.
- *Comparing split schemes instead of matched test sets.* Three successive
  delta_site designs each produced a clean-looking number and each was confounded
  by a different nuisance variable (grade, then test size, then train size).
- *A resolution hypothesis that was wrong.* Diagnosing the ITC failure as a
  stride problem was reasonable and testable; the test falsified it. Finer stride
  made the fixed-threshold grader worse, because the real limit is footprint
  dilation and per-centre classifier calibration, not localisation.
- *No mid-run checkpointing.* A 6000-step run was killed by a wall-clock limit at
  step 5750 with nothing saved. Added periodically -- then the fix to `dino.py`
  broke `frontier.py`, which builds its own `Namespace`.
- *Estimating rates from a differently-configured smoke test.* Predicted 25 min for
  a run that took 37, because the smoke test used 1/4 the prototypes and half the
  batch.

---

## 9. Limitations, stated plainly

- **Cross-modal results are n=1 slide, one tissue** (lymph node). Not
  generalisable. HEST-1k would fix this but is gated (HTTP 403 even
  authenticated), so an open 10x Visium sample was substituted.
- **~14 test patients** per honest split. Benchmark resolution is roughly
  +-0.02-0.03 AUC; differences smaller than that are noise. MIL numbers carry
  5-seed error bars for this reason, and raw-pixel's +-0.102 shows how thin the
  evidence is.
- ours-DINO saw **5354 patches from 55 patients** for 2500 steps. It is not
  comparable in scale to Phikon and is not presented as such.
- Flow model `cov90 = 0.817` against a 0.90 target: still mildly overconfident,
  wants post-hoc calibration.
- Only the 256 most-variable genes are modelled.
- One patch per image (centre crop of the 460x460 square) discards ~34% of each
  700x460 field.

---

## 10. Layout

```
src/
  BreaKHis track
    pathdata.py    manifest, 3 split regimes, Macenko normalisation, patch cache
    backbone.py    ViT with interpolating pos-embed; HED stain jitter + dihedral aug
    dino.py        DINO SSL with collapse monitoring as a first-class metric
    audit.py       linear probe + confound probes; rank-based AUC, no sklearn
    mil.py         gated-attention MIL over patient bags, multi-seed
    bench.py       the main table: 4 encoders x 3 regimes + MIL
    frontier.py    stain-jitter invariance/utility sweep, across-seed error bars
  spatial-transcriptomics track
    xmodal.py      cross-modal trunk, MSE vs rectified-flow heads, CRPS
    recrop.py      native-resolution re-crop for the field-of-view control
  CAMELYON17 whole-slide track
    get_camelyon17.py  centre-interleaved download, resumable, size-verified
    wsi.py             tissue detection, lesion-guided tiling from ASAP XML
    c17data.py         tile dataset + random/slide/patient/centre split regimes
    c17bench.py        encoders x regimes + centre-ID and slide-ID confound probes
    c17site.py         PAIRED leave-one-centre-out with matched test sets
    c17stage.py        AJCC size rule -> slide grade -> pN; two-stage validation
    verify.py          SINGLE SOURCE OF TRUTH: frozen snapshot, pre-registered
                       claims, paired SEs, RESOLVED/NULL/UNDERPOWERED tiers
  get_breakhis.py / get_hest.py   data acquisition, with verification output
tile_loop.sh       watcher: tiles slides as they land, overlapping the download
```

Reproduce:

```bash
uv venv --python 3.12
uv pip install torch numpy pandas pillow timm transformers openslide-bin

# BreaKHis track
python src/get_breakhis.py && python src/pathdata.py
python src/dino.py --steps 2500 --protos 256
python src/bench.py --encoders raw,imagenet,phikon,ours
python src/frontier.py --sigmas 0.0,0.10,0.35 --steps 1200

# spatial transcriptomics track
python src/get_hest.py && python src/recrop.py --size 224
python src/xmodal.py --src hest_sample224.npz --head flow --morph phikon
python src/xmodal.py --src hest_sample224.npz --head flow --morph random   # control

# CAMELYON17 track (152 GB, ~3 h at 17 MB/s; needs an external volume)
python src/get_camelyon17.py          # resumable, centre-interleaved
./tile_loop.sh                        # or: python src/wsi.py
python src/c17data.py                 # split disjointness audit
python src/c17bench.py --encoders raw,imagenet,phikon --grades micro,macro
python src/c17site.py  --encoders raw,phikon --grades micro,macro
python src/c17stage.py --encoder phikon                      # stage A + stage B
python src/wsi.py --stride-frac 0.25 --out-dir .../tiles_s4  # resolution test
python src/c17stage.py --encoder phikon --tiles tiles_s4 --keep itc,micro

# the only command whose output is quotable without caveats
python src/verify.py                  # -> verify_results.json + snapshot id
```

Every claim above is printed by one of these commands.
