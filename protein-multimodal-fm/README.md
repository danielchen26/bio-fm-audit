# A multimodal protein foundation model, built from scratch and measured

This is the half of the repository that *builds* a foundation model rather than
auditing someone else's. It is an ESM3-style multimodal generative masked model:
four discrete tracks, one shared trunk, per-track noise schedules, any-to-all
conditioning.

It also contains the single most transferable result in the project, which is a
negative one:

> The structure tokenizer reached a **perfect-looking** training state — loss
> 3.69 → 1.78, all 512 codebook entries in use, codebook perplexity 444 of 512 —
> while its tokens carried **0.55 of 2.39 bits** of secondary-structure entropy.
> **23%.** Nothing in the loss, the codebook usage, or the perplexity showed this.

Everything below was measured on one laptop (Apple M4 Max, MPS).

---

## What it is

| track | vocabulary | source |
|---|---|---|
| sequence | 20 aa + unknown | measured |
| structure | 512 VQ codes | learned tokenizer over backbone geometry |
| SS8 | 8 DSSP states | derived from coordinates |
| SASA | 4 relative-burial bins | Shrake–Rupley from coordinates |

Trained as a generative masked model: each track is corrupted with its own cosine
noise schedule, plus 15% whole-track dropout. That dropout is what buys any-to-all
conditioning — inverse folding, forward folding, and function-prompted
co-generation all fall out of one set of weights instead of three models.

```
protein:   sequence  <-> structure tokens <-> SS8 / SASA
```

The same trunk, with the modality pair swapped, becomes the cross-modal
morphology ↔ transcriptome model in `../pathology-fm/src/xmodal.py`.

### Scale, stated honestly

| | this | ESM3 |
|---|---|---|
| parameters | ~19M | 98B |
| training tokens | 8.6 × 10⁵ residues | 7.71 × 10¹¹ |
| proteins | 5.8 × 10³ | 2.78 × 10⁹ |
| codebook | 512 | 4096 |

Six orders of magnitude apart. The goal was **mechanism completeness**, not scale:
every mechanism in the real recipe is present and each one is independently
verifiable. The engineering scales 1:1 to a multi-GPU run. **The numbers do not.**

---

## Data, and why the split is the first thing to get right

5,770 proteins / 857k residues from RCSB, one representative per **30%-sequence-identity
cluster**, so a random split over representatives *is* a homology-aware split.
Train 4,616 / val 577 / test 577.

Two independent validations before any training:

**DSSP assignment, checked against ubiquitin (1UBQ).** The implementation is
Kabsch–Sander hydrogen-bond energies with standard priority H > B > E > G > I > T > S.

```
seq  MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG
ss8  -EEEEEESSS-EEEEE-TSSSBHHHHHHHHHHHH--TGGGEEEEESSEE-TSSSBTGGG--TSS-EEEEEE--S--
```

α-helix assigned at residues **23–34** and strands at 2–7 / 12–16 / 41–45 / 48–49
/ 66–71 — ubiquitin's canonical topology, residue for residue.

**Local frame geometry, checked against chemistry** across 800 proteins. The frames
and the FAPE projection share an einsum that is easy to get silently wrong:

| quantity | measured | literature |
|---|---|---|
| \|CA–C\| | **1.524 Å** | 1.525 Å |
| \|CA–N\| | **1.459 Å** | 1.458 Å |
| ∠ N–CA–C | **111.1°** | 111.2° |

Plus the structural invariants the construction requires: CA exactly at the
origin, C exactly on +e₁, N exactly in the e₁e₂ plane. An earlier einsum
subscript collision would have broken all six of these numbers at once.

---

## The structure tokenizer, and how it failed

Geometric encoder (SE(3)-invariant local neighbourhood, K=16) → EMA vector
quantiser → decoder trained with FAPE + distogram + distance-map losses.

### Collapse, caught and survived

```
[    1] codes 472/512  ppl 385.7   <- data-initialised codebook
[  250] codes   1/512  ppl   1.0   <- total collapse: every residue -> one code
[  500] codes  32/512  ppl   8.3   <- dead-code revival begins
[ 1250] codes 426/512  ppl 260.3   <- recovered
[ 6000] codes 512/512  ppl 444.5
```

Two mechanisms prevented a permanent collapse: **codebook initialisation from
data** (random init is the primary cause) and **dead-code revival**. Without them
the run parks at `codes 1/512` — every structure token identical, the downstream
model reporting a normal loss while the structure track carries literally zero
information.

**This is why `codebook_usage` and `perplexity` are logged as first-class metrics,
not inferred from the loss.**

### And then the tokens were still thin

The recovered tokenizer looked healthy on every training metric. A parameter-free
probe — a plain lookup table, which cannot overfit in any interesting way — says
otherwise:

| probe (held-out, 30%-identity disjoint) | lookup accuracy | marginal baseline | information |
|---|---|---|---|
| token → SS8 | 53.2% | 34.0% | **0.55 / 2.39 bits (23%)** |
| token → Ramachandran-36 | 39.9% | 34.5% | **0.44 / 3.15 bits (14%)** |

512 codes is 9 bits of capacity delivering 0.55 bits of SS8. Capacity was not the
constraint.

### A hypothesis of mine, tested and killed

I assumed the information *was* in the tokens and the frame-prediction head was
the weak link. Testable: rebuild coordinates from the predicted **distogram** via
classical MDS, bypassing the frame head entirely.

| reconstruction path | CA-RMSD |
|---|---|
| predicted frames | 10.98 Å |
| predicted distogram → MDS | **12.48 Å** — *worse* |
| contact F1 (<8 Å, \|i−j\| ≥ 6) | 0.302 |

Hypothesis falsified. The tokens themselves were thin. Had I trusted the guess I
would have spent a training cycle enlarging the wrong component.

### The diagnosed fix, and a matched-budget control that saved me

Root cause: the encoder consumes **local** geometry while the decoder is asked for
**global** coordinates, so the gradient reaching the encoder never pressures it to
preserve local conformation. Fix: add a local-reconstruction head that rebuilds the
neighbourhood the encoder consumed — an autoencoder in the proper sense.

Naively comparing runs said the fix was **harmful**:

| run | local head | steps | I(token; SS8) |
|---|---|---|---|
| v1 | off | 6000 | 0.55 bits |
| v2 | **on** | 2500 | 0.42 bits |

At matched budget it reverses:

| run | local head | steps | I(token; SS8) |
|---|---|---|---|
| v1b | off | 2500 | 0.36 bits |
| v2 | **on** | 2500 | **0.42 bits (+17%)** |
| v1 | off | **6000** | **0.55 bits** |

So the fix works (+0.06 bits SS8, +0.08 Ramachandran) but it is a **second-order**
correction: raising the step count 2500 → 6000 buys +0.19 bits, three times more.
The real bottleneck was undertraining.

**Without the matched-budget control I would have concluded my own fix was harmful
— the exact opposite of the truth.** Comparisons across different training budgets
are meaningless, and they are easy to accept because they look like diligence.

---

## The multimodal trunk

`multimodal.py` — shared trunk, RMSNorm / RoPE / QK-norm / SwiGLU, one head per
track, geometric attention bias in the first block.

### A leakage bug I wrote and then caught

Geometric attention conditions on true coordinates. Fed unconditionally, the
`sequence → structure` evaluation reads its answer off its own input, and the
model would have posted an excellent forward-folding number while learning
nothing. The bias is now **gated by the structure mask**: geometry is visible only
between residues whose structure token is visible.

```python
g = struct_given if struct_given is not None else mask
vis = (g & mask)[:, None, :, None] & (g & mask)[:, None, None, :]
bias = bias * vis
```

Any multimodal design where one track is a deterministic function of another needs
this check. It does not raise an error and it does not perturb the loss curve.

### Status: implemented, not evaluated

**The trunk trains and the evaluation harness is written, but the any-to-all
results table was never produced.** Two runs — the 4-track trunk and its
sequence-only ablation — were cancelled mid-flight when the project pivoted to
computational pathology. What exists:

| component | status |
|---|---|
| data pipeline, homology-aware splits | ✅ verified against ubiquitin and bond chemistry |
| structure tokenizer | ✅ trained, and **measured** (0.55/2.39 bits) |
| tokenizer fix + matched-budget control | ✅ measured (+17%, second-order) |
| multimodal trunk | ✅ implemented, smoke-tested |
| any-to-all conditional table | ❌ **not run** |
| SS8-prompted closed-loop generation | ❌ **not run** (`decode.py` written) |

Reporting this as absent rather than quietly omitting it: the trunk's numbers do
not exist, so nothing about its performance is claimed here.

---

## Layout

```
src/
  data.py              RCSB 30%-identity clusters -> backbone -> DSSP SS8 -> SASA
  nn.py                RMSNorm, RoPE, QK-norm attention, SwiGLU, local frames, FAPE
  struct_tokenizer.py  geometric encoder + EMA VQ + FAPE/distogram/local decoder
  multimodal.py        shared trunk, per-track noise schedule, any-to-all eval
  dataset.py           token-budget batching
  diag_local.py        parameter-free token information probe -- the key tool here
  diag_tokenizer.py    frames-vs-distogram-MDS reconstruction comparison
  decode.py            SS8 prompt -> co-generate -> decode 3D -> re-run DSSP
```

```bash
python src/data.py                                     # build + verify the dataset
python src/struct_tokenizer.py --steps 6000            # tokenizer
python src/diag_local.py vqvae.pt                      # the number that matters
python src/struct_tokenizer.py --local-weight 0.0 --steps 2500 --out vqvae1b.pt
python src/multimodal.py --steps 6000                  # trunk (table not produced)
```

---

## What this half establishes

1. A multimodal generative masked model over four biological tracks, from scratch,
   with the mechanisms that actually matter present and separately verifiable:
   homology-aware splits, VQ collapse control, per-track noise schedules,
   whole-track dropout for any-to-all conditioning, leakage gating.
2. **Never trust a pretraining loss for an intermediate component.** Build a
   parameter-free information probe and log it alongside the loss. This tokenizer
   would have shipped into the trunk carrying 23% of the entropy it appeared to.
3. **Never compare across training budgets.** A matched-budget control reversed the
   sign of my own conclusion.
4. **Test your diagnosis before acting on it.** Two of my three hypotheses here
   were falsified by cheap experiments that took minutes.

The parameter-free-probe idea transfers directly: in `../pathology-fm` it becomes
the confound audit that finds Phikon identifying the source hospital at 99.77%.
