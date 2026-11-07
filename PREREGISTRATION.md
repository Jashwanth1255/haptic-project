# Preregistration

Recorded 2026-10-31, before any real Cluster cloth data was processed.

## DTD textile subset

Twelve DTD classes form the textile subset:

1. braided
2. fibrous
3. frilly
4. gauzy
5. interlaced
6. knitted
7. lacelike
8. matted
9. meshed
10. pleated
11. woven
12. wrinkled

**Selection rule.** Classes describing fabric structure are included. Print
patterns such as striped, chequered, paisley and polka-dotted are excluded,
because a print carries no information about how a fabric feels.

**Disclosure.** The list was chosen after the 47-class DTD fold 1 test results
had been seen. It is fixed in advance of the Cluster cloth data only.

**This list is fixed.** It must not be revised after seeing results from the
real Cluster cloth data.

## Prediction: haptic clusters mirror visual clusters

Recorded 2026-11-01, before any haptic descriptors exist.

In haptic descriptor space, two groups will each cluster together:

- **Pile:** fibrous, matted
- **Weave:** braided, woven, interlaced, meshed, knitted

The groups were read from the visual 12×12 confusion matrix on DTD fold 1. The
visual pattern is therefore the source of the prediction, not evidence for it.
Only the haptic result tests it.

**Mapping.** These are DTD class names, and Cluster's cloth materials carry no
DTD labels. Each of the 16 materials (IDs 102–117) must be assigned to pile,
weave or neither from its name and scan alone, and recorded here, before any
descriptor is computed. The assignment must not change afterwards.

**Test.** Use the velocity-normalised descriptors, standardised, with one centroid
per material and Euclidean distance. For each group, compare mean within-group
distance with mean distance to the other assigned materials, using a one-sided
permutation test on the group labels, Holm-corrected across the two groups at
alpha 0.05.

**Outcome.** The prediction is supported only if both groups pass. A group with
fewer than 3 materials cannot be tested, and that is reported as untested. A
failed prediction is reported as failed.

## Objective 1 gate thresholds

As set in `haptic/separability.py`:

| Constant | Value | Meaning |
|---|---|---|
| `ALPHA` | 0.05 | ANOVA significance level, applied to Holm-corrected p-values |
| `SILHOUETTE_MIN` | 0.30 | Minimum silhouette score in descriptor space |
| `COLLAPSE_RATIO` | 1.0 | Centroid distance / pooled spread; a pair below this has collapsed |
| `MIN_SEPARABLE` | 8 | Minimum effective classes after merging collapsed pairs |

## Prediction: the drape group does not transfer

Recorded 2026-11-03, before any haptic descriptors exist.

**Status.** This entry is **exploratory**. The pile and weave prediction above is
**confirmatory**: it was recorded on 2026-11-01 from the DTD fold 1 confusion
matrix. The drape group was found afterwards, post hoc, in the confusion matrix
pooled over folds 2–10, which was not expected.

- **Drape:** gauzy, pleated, wrinkled

**Prediction.** Pile and weave will cluster in haptic descriptor space, as
registered above. Drape will not. Drape describes how fabric hangs and folds in
three dimensions, and a probe sliding over a flat sample does not measure that.

**Mapping.** This extends the assignment rule above to four labels: pile, weave,
drape or neither. Each Cluster material gets one, from its name and scan alone,
before any descriptor is computed.

**Test.** Same as above: within-group against between-group distance, one-sided
permutation test. Drape is tested on its own at alpha 0.05, outside the Holm family
of the confirmatory test, which stays as registered. Report the within/between
distance ratio beside the p-value. With few materials, a non-significant drape
result is weak evidence that the group does not cluster, and must be described as
weak. A drape group with fewer than 3 materials is reported as untested.

## Prediction: representability ordering

Recorded 2026-11-03, before any perceptual rating or haptic descriptor exists.

**Status.** Confirmatory.

The five perceptual attributes will rank by Stage-2 representability:

roughness ≈ bumpiness > slipperiness > softness > warmth

Warmth is a control. It depends on thermal properties, which neither an
accelerometer nor a vibration motor carries, so it should not be representable.

**Measure.** For each attribute, Stage-2 representability is the
leave-one-material-out R² of predicting that attribute's mean rating from the four
velocity-normalised haptic descriptors. Each attribute also gets a permutation
test against the mean predictor.

**Outcome.** The ordering holds if all three of these hold in the point estimates:

- the lower of roughness and bumpiness is above slipperiness
- slipperiness is above softness
- softness is above warmth

"≈" means no prediction is made about the order of roughness and bumpiness. The
control holds if warmth does not beat the mean predictor, meaning its permutation
p is at or above 0.05. Report each part separately, and report a failure as a
failure. With 16 materials the ordering is a description, not a significance test.

If the pilot drops an attribute for low rater agreement (ICC(2,k) below 0.50), the
ordering is judged over the remaining attributes, and the drop is reported.

## Cluster cloth group assignment

Recorded 2026-11-04, before any descriptor was computed from real Cluster data.
Only the file names, timestamps and position traces had been read; no acceleration
spectrum, descriptor or separability figure existed.

Assigned from each material's name (`documents/texture_list.pdf`) and its 300 dpi
scan (`images_scan_area/img<id>.jpg`, viewed whole and at 4x), using the labels
and rule registered above.

| Group | Materials | n |
|---|---|---|
| Pile | 102 Velvet, 103 Corduroy | 2 |
| Weave | 104 Cotton, 105 Cotton broadcloth, 111 Mesh, 112 Wool, 114 Polyester, 116 Linen, 117 Cotton linen | 7 |
| Drape | 106 Cotton gauze, 107 Chiffon, 109 Georgette, 110 Organdy | 4 |
| Neither | 108 Lace, 113 Nylon, 115 Silk | 3 |

Where the scan decided it:

- 104 Cotton is a gridded muslin weave with a floral print. The print is ignored.
- 112 Wool is a woven twill pinstripe suiting, not a pile, despite the fibre name.
- 114 Polyester shows a fine woven structure at 4x.
- 108 Lace is an eyelet embroidery on a plain ground. DTD "lacelike" is in no group.
- 113 Nylon and 115 Silk show no structure at scan resolution.

**Consequence, stated before the data.** Pile has 2 materials, below the minimum
of 3, so pile is reported as untested. The confirmatory prediction needs both pile
and weave to pass, so it cannot be reported as supported whatever the data show;
weave is still tested and reported. Drape (4) is tested as the exploratory entry
specifies.

This assignment must not change.

## Confirmation: cloth spectra shift with velocity after whitening

Written 2026-11-06, after run `velocity_all_materials/20261105T202123_412283Z`
showed (post hoc) that cloth's rig-whitened spectral-shift exponent, median 0.218,
exceeds the other 102 materials' median of −0.022. That finding came from
directions 0° and 90°, first repeat only. **This entry is written before any of the
held-out recordings below have been passed through the shift analysis.**

**Held-out data.** The 16 cloth materials (102–117): every recording at directions
45, 135, 180, 225, 270 and 315°, both repeats, plus the second repeat at 0° and 90°.
None of these entered the post-hoc result.

**Method, fixed now.** `haptic.velocity.shift_exponents` exactly as used in that
run. The whitening reference is the median log spectrum over all 118 materials at
the same velocity and force, from the original subset (0°/90°, first repeat), so the
reference contains no held-out data. One exponent per material is the mean over its
(force, direction, repeat) cells.

**Predictions** (both one-sided, α = 0.05, Holm across the two):

1. The median cloth exponent on held-out data is above 0 (Wilcoxon signed-rank on
   16 per-material values).
2. Cloth's held-out exponents exceed the other 102 materials' exponents from the
   original run (Mann-Whitney U).

Both passing confirms that cloth carries a velocity-scaled component that the rest
of the dataset lacks. Either failing means the post-hoc result is reported as not
replicated. In neither case is the exponent near 1 expected, and none is claimed.

## Benchmark and findings round 2 (four analyses)

Written 2026-11-07, before any of the analyses below has been run. Nothing in
them has been computed yet. Seen so far: one audio file's length and format (to
confirm that audio and position share a clock), and the descriptor-based results
already in docs/RESULTS.md. The thresholds and predictions below must not change.

### A. Does looks predict feel? (118 materials)

**Visual representation.** ImageNet ResNet-50, the same torchvision weights as
`vision.train`, not fine-tuned. Penultimate 2048-d features of the 100 crops the
dataset provides per material (`images/crop_images/img{id}/`), averaged per
material. Distance: cosine.

**Haptic representation (primary).** Per-material mean log Welch spectrum on
`haptic.velocity.SHIFT_GRID` (20–1000 Hz), averaged over that material's in-band
recordings at directions 0° and 90°, first repeat, all velocities and forces (the
`velocity_all_materials` subset). Distance: Euclidean. Secondary, reported
descriptively: per-material mean of the four standardised raw descriptors,
Euclidean.

**Tests.** Spearman Mantel correlation between the two distance matrices, 9,999
permutations, one-sided (r > 0), Holm across the three:

1. All 118 materials, free permutation. **Prediction: r > 0, significant.**
2. All 118 materials, permutations restricted within the nine categories, so
   the shared category structure cannot produce the correlation.
   **Prediction: not significant.**
3. The 16 cloth materials only. **Prediction: not significant.**

### B. Audio versus acceleration (16 cloth materials)

Same 2,493 recordings as gate run `20261104T153936_758197Z`, same steady
window, same CNN, training and splits as `haptic/learned.py`, three seeds. Input:
the noise-cancelled mono `audio` channel at 44.1 kHz. Log STFT (n_fft 2048, hop
512), pooled into 64 log-spaced bands. Two variants: `audio_full` (20–8000 Hz)
and `audio_band` (20–1000 Hz, the acceleration band).

**Prediction (confirmatory):** `audio_full` direction-held-out accuracy exceeds
acceleration's. The test is one-sided Wilcoxon signed-rank over the 8 direction
folds (seed-averaged accuracy), against run
`o1_learned_baseline/20261106T214141_196418Z`, α = 0.05. `audio_band` is
reported descriptively, with no prediction.

### C. Unseen velocity and frequency warping (16 cloth materials)

Same recordings. Leave-one-velocity-out: 5 folds, three seeds, same CNN. Input:
log-frequency spectrogram (n_fft 1024, hop 128, 48 log-spaced bands, 20–1000 Hz).
Before the CNN, the frequency axis of each recording is warped to f·(v/40)^k, by
linear interpolation on log f. Three conditions:
- k = 0 (no warp);
- k = 0.338, the held-out median from
  `velocity_confirm_cloth/20261105T203252_596852Z`;
- k = 1 (f = v/λ).

A recording counts as correct when the majority of its three seeds are correct.
Predictions, one-sided exact sign test on recordings where two conditions
disagree, Holm across the two:

1. **k = 0.338 beats k = 0.**
2. **k = 0.338 beats k = 1.**

Disclosure: 0.338 was estimated from cloth recordings that overlap these. It was
not fitted to classification accuracy.

### D. Whole-dataset identification benchmark (118 materials), descriptive

Analysis plan fixed now; no predictions.

- **Data:** every recording of all 118 materials: 8 directions, 2 repeats, 5
  velocities, 2 forces. Same loader rules as O1: under 0.5 s of steady sliding is
  skipped, out-of-band is excluded.
- **Tasks:** 118-way material; 9-way category (IDs as in the dataset paper).
- **Splits:** direction held out (8 folds), and repeat 0 → 1.
- **Models:**
  - logistic regression on 4 raw descriptors;
  - logistic regression on 16 log bands + RMS;
  - logistic regression on both;
  - the CNN from `haptic/learned.py`, three seeds.
- **Reported:** accuracy, macro-F1, top-5 (material task), per-category recall,
  chance (1/118, 1/9) and the majority-class baseline.

## Round 3: three further analyses (E, F, H)

Registered 2026-11-07, before any of the quantities below was computed. Already
seen at registration: the round-2 results, including the per-fold accuracies of
the direction-held-out splits (90° and 270° hardest for the cloth CNN). No
direction-pair distance, rendered-envelope feature or cross-force accuracy had
been computed.

Data for E and H: `features.csv` from the round-2 features kernel (code
`c0efe48`), in-band recordings only: 18,569 recordings, 118 materials. Features: 16
log band powers (`band00`–`band15`) plus `log_rms`, standardised over the whole
table.

### E. Direction structure: opposite vs perpendicular strokes

For each (material, velocity, force, repeat), take the feature vector at each of
the 8 directions. Opposite pairs: (θ, θ+180°), 4 per cell. Perpendicular pairs:
(θ, θ+90°), 8 per cell. Euclidean distance; per material, average over cells to
get d_opp and d_perp. A cell missing a direction drops only the pairs that need it.

1. **Prediction (cloth, 16 materials):** d_perp > d_opp, i.e. a stroke along the
   same axis (warp or weft) resembles its reverse more than the crossing axis.
   One-sided Wilcoxon signed-rank over materials.
2. **Other 102 materials:** two-sided Wilcoxon, no direction predicted.

Holm across the two. Reported descriptively: diagonal pairs (45°, 135°) and the
repeat-to-repeat distance at the same direction, as a noise floor.

### F. What survives the on/off channel (16 cloth materials)

The 2,493 gate recordings (run `20261104T153936_758197Z`), same steady window.
Two sources per recording, both band-passed to 20–1000 Hz:
- acceleration (as in O1);
- the contact audio.

For each source, the amplitude envelope (`render.signal_envelope`) of the first
500 ms is rendered with scheme B (`render.encode_b`): tick 10 ms, level from RMS
intensity relative to that source's 95th-percentile intensity. It is then passed
through the assumed actuator lag (`render.achievable_envelope`, τ = 25 ms).
Features: log modulation power in 12 log-spaced bands from 1 to 50 Hz, plus the
log mean level. Computed on the intended envelope and on the achievable
(rendered) envelope. Model: standardised logistic regression, 16-way,
direction-held-out folds.

1. **Rendered acceleration identifies fabrics above chance:** per-fold accuracy
   > 1/16, one-sided Wilcoxon over the 8 folds.
2. **Rendering loses identity:** rendered < intended accuracy, acceleration,
   one-sided Wilcoxon over the 8 folds.
3. **Audio vs acceleration after rendering:** two-sided Wilcoxon over the 8
   folds, no direction predicted.

Holm across the three.

### H. Force transfer

Logistic regression on bands + `log_rms` + the 4 raw descriptors (the
`bands+raw4` set of D). Train on force A, repeat 0. Test on repeat 1 at force A
(same) and at force B (cross). Run both ways round, A = 0.5 N and A = 1.0 N, and
pool the predictions. Per-material recall, same vs cross.

1. **Prediction (cloth, 16 materials):** cross-force recall < same-force recall.
   One-sided Wilcoxon over materials.
2. **Prediction (all 118 materials, 118-way):** the same, over 118 materials.

Holm across the two. Reported descriptively: the same comparison after
subtracting `log_rms` from every band (level removed).

### G. Scan → product-photo transfer (registered after E, F, H; before any G data were processed)

Registered 2026-11-07. Photo source: DeepFashion-MultiModal, Kaggle mirror
`silverstone1903/deep-fashion-multimodal` (CC BY-NC-SA 4.0; 12,278 front-view
product photos). This mirror has no separate fabric annotation file. Each
caption names the fabric of each garment it describes ("... with cotton fabric
..."). A photo is labelled with a fabric only when every fabric its caption names
is that one fabric. Only the caption label counts were seen before registration
(cotton 13,528 mentions, denim 2,247, chiffon 1,832, knitting 989, leather 391,
furry 120). No image had been opened.

**Classes:** the three fabrics present in both sources.
- cotton: Cluster 104 cotton, 105 cotton broadcloth, 106 cotton gauze;
- chiffon: Cluster 107;
- leather: Cluster 87–101.

Cotton linen (117) is excluded as mixed.

**Photos:** up to 100 per class, drawn at random (seed 0) from the qualifying
photos; all of them if fewer qualify.

**Scans:** the dataset's 100 crops per material. Crops 1–70 train, 71–100 test,
within each material. With one chiffon material, a held-out-material split is
impossible, so the scan test is within-material and optimistic. That works for
the hypothesis, not against it.

**Model:** frozen ImageNet ResNet-50 (`IMAGENET1K_V2`, as in A), 2048-d features,
standardised logistic regression trained on scan training crops. Classes are
weighted to balance (cotton 3 materials, chiffon 1, leather 15).

**Metric:** balanced accuracy (mean per-class recall); chance 1/3.

1. **Prediction (headline):** balanced accuracy on photos < balanced accuracy on
   held-out scan crops. One-sided, 10,000 class-stratified bootstrap resamples of
   both test sets; p = share of resamples with photo ≥ scan.

Descriptive, no prediction:
- photo balanced accuracy with a bootstrap 95% CI against chance;
- the reverse direction (train on photos, test on all scan crops);
- photo-to-photo 5-fold cross-validated balanced accuracy, as a within-domain
  reference.

## Round 4: five further analyses (I, J, K, L, M)

Registered 2026-11-07, after rounds 1–3 and before any quantity below was
computed. All run on CPU.

### I. Does removing the rig's shared response help identification?

Features: the 16 log bands plus `log_rms` from `features.csv` (in-band, 18,569
recordings). Whitening: from each feature, subtract its median over all 118
materials at the same (velocity, force, direction). This is label-free, but it
uses the test direction's recordings of every material; we disclose that. Model:
standardised logistic regression, direction held out (8 folds).

1. **Cloth, 16-way:** whitened > raw accuracy, one-sided Wilcoxon over the 8 folds.
2. **All 118, 118-way:** whitened > raw, the same test.

Holm across the two.

### J. Scan → photo transfer, stricter and with a photo-trained model

Same photos and classes as G (run `photo_transfer/20261107T144009_695846Z`).

1. **Leave-one-material-out scans.** Two classes, cotton (3 materials) vs leather
   (15); chiffon has a single material and drops out. ResNet-50 features as in G,
   class-balanced logistic regression. Each scan material is predicted by a model
   trained on the other materials. Photos are predicted by the model trained on
   all scan materials. **Prediction: photo balanced accuracy < LOMO scan balanced
   accuracy**, by a one-sided class-stratified bootstrap as in G.
2. **CLIP zero-shot.** `openai/clip-vit-base-patch32` (Hugging Face), prompt
   "a photo of {cotton|chiffon|leather} fabric", argmax over the 3 classes, on all
   1,900 scan crops and the 211 photos. **Prediction: photo balanced accuracy >
   scan balanced accuracy** (CLIP is trained on web photographs), by the same
   bootstrap.

Holm across J1 and J2. Descriptive: a CLIP linear probe trained on scans and
tested on photos.

### K. Test–retest reliability of the features (118 materials), descriptive

For each feature (4 raw descriptors, 16 bands, `log_rms`; logs of positive
descriptors): ICC(3,1) of repeat 0 vs repeat 1 over all (material, velocity,
force, direction) cells; and split-half reliability of per-material means,
Spearman–Brown corrected. The good-reliability threshold is fixed now at ICC ≥ 0.75.
We report the features that meet it. No prediction.

### L. Image → category → descriptor at 118-material scale

Leave-one-material-out. A logistic regression on per-crop ResNet-50 features
(IMAGENET1K_V2) is trained on the crops of the other 117 materials. It predicts
the held-out material's category by majority vote over its 100 crops. The
predicted descriptor is the mean of that category's other materials. The
descriptor is each material's mean of log centroid, log intensity, pattern and
log duration over its in-band recordings. Baselines: the mean of all other
materials; and, as an oracle, the true category's mean.

- **Prediction:** per dimension, absolute error of the image-predicted descriptor
  < the all-materials mean. One-sided Wilcoxon over 118 materials, Holm across
  the 4 dimensions.
- Also reported: category accuracy from images, and the oracle errors.

### M. Contact audio at 118-material scale (Kaggle CPU)

Contact audio for every recording of all 118 materials, cut to the steady window
of the position file, with the same loader rules. Features: 16 log band shares
over 20–1000 Hz plus log RMS (as for acceleration); and 16 log band shares over
20–8000 Hz plus log RMS. Logistic regression, direction held out, 118-way
material.

- **Prediction:** audio 20–8000 Hz > acceleration bands (D's `bands`, run
  `benchmark118_linear/20261107T103411_132525Z`), one-sided Wilcoxon over the 8
  folds.
- Reported descriptively: audio 20–1000 Hz, the 9-way category task, and the
  repeat split.

## Round 5: consequences of rig whitening (N, O)

Registered 2026-11-07, after round 4's analysis I showed whitening helps linear
identification. Not yet computed. Whitening as in I: per feature, subtract the
cross-material median at the same (velocity, force, direction).

### N. Does whitening rescue the O1 separability gate? (16 cloth materials)

The O1 gate is fixed and failed (run `o1_gate/20261104T153936_758197Z`); this does
not change it. We recompute the gate's own metrics (pooled silhouette, collapsed
pairs out of 120, effective classes, same thresholds) on whitened versions of:
- the 4 raw descriptors (log centroid, log intensity, entropy, log duration);
- the 17 band features.

Whitening is computed over all 118 materials and applied to the cloth recordings.

**Prediction:** silhouette on whitened band features > silhouette on raw band
features. Tested by a permutation test that permutes the whitened/raw assignment
within each recording's pair of silhouette values (10,000 permutations), one-sided.
The 4-descriptor version is reported descriptively. We expect it to stay below
the 0.30 threshold, but make no formal prediction.

### O. Velocity invariance after whitening (open question 2; 118 materials)

For each material and feature, the coefficient of variation across the 5
velocities of the per-velocity mean. For log-scale features we use the SD
instead, because their mean can be near zero. Computed on raw and on whitened
features.

**Prediction:** across materials, the median over features of the
velocity-induced SD is lower after whitening. One-sided Wilcoxon over 118
materials, using each material's median over the 17 band features. Also reported
per category.
