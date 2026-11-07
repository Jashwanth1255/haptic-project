# Results on real data

All numbers come from saved runs under `results/`; the run is named beside each
table. Computed on 2026-11-04 on CPU. Nothing here is synthetic. Where an analysis
was not pre-registered it is marked **exploratory**.

Data: Cluster Haptic Texture Dataset, cloth IDs 102–117 (16 materials), Hugging
Face mirror revision `b7c2fb70`. 2,560 recordings; 47 skipped for under 0.5 s of
steady sliding; 20 more out of band (Velvet 12, Corduroy 5, Cotton 2, Lace 1).
2,493 recordings enter the analyses.

Order of events, verifiable in git: the cloth group assignment was committed
(`a2610b8`) before any descriptor was computed. Gate thresholds are unchanged
since 2026-10-31.

---

## O1 — Haptic descriptor separability: the gate fails

Run `o1_gate/20261104T153936_758197Z`.

| | Raw descriptors | Velocity-normalised |
|---|---|---|
| Silhouette, pooled (threshold 0.30) | −0.164 | −0.124 |
| Silhouette per velocity, 20–60 mm/s | −0.19 to −0.15 | −0.18 to −0.15 |
| Collapsed pairs (separation < 1.0) | 120 of 120 | 120 of 120 |
| Effective classes (need 8) | 1 | 1 |
| Verdict | **FAIL** | **FAIL** |

The largest pairwise separation is 0.57 (Lace–Wool, raw), well under the 1.0
collapse threshold. Coarsening the class list, as the plan prescribes for a
failed gate, cannot help: every pair overlaps, so all 16 merge into one class.

Every dimension differs across materials (Holm p < 1e-15), but the effects are
small:

| Dimension | ω² | Dimension (normalised) | ω² |
|---|---|---|---|
| frequency_hz | 0.160 | wavelength_mm | 0.038 |
| intensity | 0.067 | roughness_index | 0.084 |
| pattern | 0.080 | pattern | 0.080 |
| duration_ms | 0.120 | burst_spacing_mm | 0.099 |

**Where the variance goes** (`o1_real_analysis/20261104T155118_372566Z`, type II
sums of squares, logged where positive):

| Dimension | Material | Direction | Material × direction | Velocity | Force | Residual |
|---|---|---|---|---|---|---|
| frequency_hz | 0.153 | 0.101 | 0.142 | 0.019 | 0.003 | 0.582 |
| intensity | 0.079 | 0.029 | 0.085 | 0.556 | 0.011 | 0.241 |
| pattern | 0.085 | 0.225 | 0.073 | 0.109 | 0.020 | 0.489 |
| duration_ms | 0.085 | 0.206 | 0.071 | 0.026 | 0.007 | 0.605 |
| wavelength_mm | 0.035 | 0.023 | 0.032 | 0.778 | 0.001 | 0.132 |

Material explains 8–15% of each raw dimension. Sliding direction and its
interaction with material explain as much again, and most is residual
(recording-to-recording). Velocity normalisation makes wavelength 78% velocity:
see O3.

**Checked and ruled out as the cause.** A spectral component at about 5 Hz per
mm/s (a 0.2 mm spatial period) appears in every material and is the tallest peak
in 26% of recordings, probably from the drive (not verified). It carries only
about 4% of in-band power, no more than its share of the band, so it does not
explain the collapse.

### Pre-registered group test

Same run. Velocity-normalised, standardised, mean centroid per material,
Euclidean; exact permutation over all relabellings.

| Group | n | Within / between | p | Holm p | Result |
|---|---|---|---|---|---|
| Pile (confirmatory) | 2 | — | — | — | untested (< 3 materials, stated before data) |
| Weave (confirmatory) | 7 | 0.806 | 0.114 | 0.228 | does not cluster |
| Drape (exploratory) | 4 | 0.900 | 0.281 | — | does not cluster, as predicted (weak evidence) |

**Confirmatory prediction: failed** for weave; pile untested. Sensitivity to the
one reading the registration left open: weave p = 0.056 against pile only, 0.100
against all 16. No reading reaches 0.05. Raw descriptors give the same picture
(weave p = 0.129).

### Exploratory: how much identity is in the signal at all

Not pre-registered. Material identification, multinomial logistic regression,
cross-validated with **sliding direction held out** (8 folds). Chance 0.0625.

| Features | Accuracy | Fold SD |
|---|---|---|
| 4 raw descriptors | 0.138 | 0.032 |
| 4 normalised descriptors | 0.154 | 0.042 |
| 16-band spectrum + RMS | 0.218 | 0.049 |
| Spectrum + raw descriptors | 0.248 | 0.060 |

The signal holds some identity, about four times chance with the richer spectrum,
and the four-number descriptor keeps roughly half of it. Best recall: Corduroy and
Wool 0.45, Nylon 0.42. Worst: Cotton broadcloth 0.08, Organdy 0.10, Polyester 0.12.

#### Learned baseline: a CNN on the spectrogram (exploratory)

Run `o1_learned_baseline/20261106T214141_196418Z` (Kaggle T4, code
`312c86f+uncommitted sources-sha256:6ace53cfef3f`). It uses the same 2,493 in-band
recordings as gate run `20261104T153936_758197Z`. A small CNN reads the log
spectrogram (20–1000 Hz, 0.5 s windows), and velocity and force go to its head.
Each recording is scored by averaging its windows. Three seeds. Not pre-registered;
added to answer "is it the descriptor or the signal?"

| Split | Accuracy (mean over 3 seeds) | SD over seeds | SD over folds | Fold range |
|---|---|---|---|---|
| Sliding direction held out (8 folds) | **0.378** | 0.003 | 0.080 | 0.215–0.455 |
| Repeat 0 → repeat 1 (all directions seen) | 0.542 | 0.007 | — | 0.533–0.547 |

- The CNN reaches about six times chance with direction held out, against 0.248 for
  the best hand-built feature set under the same split. So the four-number
  descriptor throws away identity that the signal carries; the O1 negative result is
  about the descriptor, not only the signal. Even so, 62% of held-out-direction
  recordings are still misidentified.
- Direction matters. The 90° and 270° folds score 0.267 and 0.241, while the
  other six score 0.386–0.453. Holding out a sliding direction perpendicular to most
  of the training data costs the most. This is an observation, not a tested effect.
- The repeat split, with every direction seen, gives an upper bound of 0.542.
- Per-material recall with direction held out: best Corduroy 0.66, Mesh 0.53;
  worst Cotton broadcloth 0.14, Organdy 0.23. Corduroy is also the best and Cotton
  broadcloth and Organdy are also the worst under the logistic regression above.
  Wool and Nylon are not near the top here (0.38 each).

---

## O2 — Scan domain: the classifiers on Cluster's 300 dpi scans

Run `o2_cluster_scans/20261104T161427_584869Z`. 100 crops of 224 px (about 19 mm)
per material, 1,600 in all.

**Material (MINC-2500, fold 1).** Every crop is cloth.

| Test set | "Fabric" recall |
|---|---|
| MINC-2500 test photographs (`results/o2/minc2500_fold1_seed0`) | 0.756 (189/250) |
| Cluster cloth scan crops | **0.033** (52/1,600) |

No material is called fabric by majority. The scans are read as carpet (velvet,
corduroy, wool, polyester, cotton linen), wallpaper (cotton, gauze, chiffon, lace,
organdy, nylon), tile (mesh, linen) or painted. **Exploratory check of field of
view** (`o2_cluster_scan_scale/20261104T161606_361379Z`): whole 100 mm scans give
1/16 fabric and 50 mm quadrants 3/64, so crop scale does not explain the gap. A
photo-trained material model does not transfer to flat, evenly lit flatbed scans.

**Texture (DTD, 10 folds), mapped to the registered groups.**

| Measure | Mean ± SD over folds | Baseline |
|---|---|---|
| Crop group accuracy, all 16 | 0.531 ± 0.043 | 0.438 (always "weave") |
| Crop group accuracy, 13 assigned | 0.652 ± 0.053 | 0.538 |
| Per-material majority, all 16 | 0.531 ± 0.044 | 0.438 |

Corduroy is called pile 99% of the time, mesh/wool/polyester weave 94–100%,
organdy drape 91%. Velvet is called weave (90%), and all three "neither" materials
mostly drape. This is 4-way group accuracy against labels assigned by eye; it is
not comparable with the 12-way DTD test accuracy (0.812 ± 0.025, folds 2–10).

**Not done: the product-photo side** (P6). It needs ≥ 50 product photographs
labelled independently by two people. `python -m vision.cli transfer` is ready for
them.

---

## O3 — Image → descriptor retrieval, and velocity

Run `o3_real/20261104T161615_092994Z`. 16 materials × 100 crop views; descriptor
per material is the mean of in-band recordings at 40 mm/s and 1.0 N.

**Closed set** (which material is this crop?), 4 view folds. Chance 0.0625 / 0.1875.

| Strategy | Top-1 | Top-3 |
|---|---|---|
| Class pair (MINC × DTD) | 0.473 ± 0.007 | 0.759 ± 0.015 |
| Cosine (DTD ResNet-50 features) | 0.990 ± 0.005 | 0.999 ± 0.001 |

Caveat: query and database crops come from the same scan, so cosine mostly
recognises the scan. This is an upper bound, not cross-image retrieval.

**Open set** (material held out; how good is the nearest substitute's descriptor?).
Standardised MAE, mean over 16 folds:

| Dimension | Class pair | Cosine | Mean of database | Random substitute |
|---|---|---|---|---|
| frequency_hz | 1.012 | 1.001 | 0.874 | 1.134 |
| intensity | 1.193 | 1.069 | 0.891 | 1.130 |
| pattern | 0.917 | 0.951 | 0.790 | 1.068 |
| duration_ms | 1.095 | 1.063 | 0.843 | 1.078 |

Neither strategy beats predicting the mean descriptor. Class pair is significantly
worse than the mean for intensity and duration (Wilcoxon, Holm p = 0.026). Cosine
and class pair do not differ (all Holm p = 1). This follows from O1: when
descriptors barely separate materials, nothing retrieved by appearance can beat the
average.

### Velocity invariance (open question 2)

| Measure | Result |
|---|---|
| Exponent of frequency on velocity (f = v/λ predicts 1) | median −0.03, range −0.32 to 0.35 |
| Exponent of intensity on velocity | median 1.34 |
| CV across 20–60 mm/s: frequency_hz | 0.063 |
| CV: wavelength_mm (normalised) | 0.398 |
| CV: intensity / roughness_index | 0.530 / 0.373 |

**Real cloth does not obey f = v/λ on this rig.** The spectral centroid is nearly
velocity-invariant, so dividing velocity by it makes wavelength six times more
velocity-dependent than the raw frequency. Velocity-conditioned prediction loses to
velocity-agnostic for frequency at 20, 30, 50 and 60 mm/s (Holm p ≤ 0.006, error up
234%), and wins for intensity at 20 and 30 mm/s (error down 30–49%). The same holds
for an unseen query velocity and for the cosine substitute. The synthetic O1 result
(silhouette +0.43 normalised) was built on the relation these data reject.

### Velocity across all 118 materials: rig or cloth?

Run `velocity_all_materials/20261105T202123_412283Z`. Added 2026-11-05 to tell
apart two explanations of the cloth result. Subset identical for every material:
directions 0° and 90°, first repeat, five velocities, two forces. 2,322 descriptors
from 118 materials; 38 files skipped, 7 out of band. (Two earlier runs of the same
script, `…200348…` and `…201244…`, hold the same centroid numbers and lack the later
tests; the latest supersedes them.)

**Centroid exponent** (pre-planned for this script):

| | Result |
|---|---|
| Median exponent, cloth (n=16) / others (n=102) | −0.039 / −0.084 |
| Materials within 0.25 of 1 / of 0 | 0 of 118 / 0.92 |
| Others vs 1 (Wilcoxon) | p = 1.8×10⁻¹⁸ |
| Cloth vs rest (Mann-Whitney) | p = 0.13 |
| Kruskal-Wallis across 9 categories | p = 1.2×10⁻⁴ |
| Centroid at 40 mm/s, category medians | 241–266 Hz (every category) |

**No material, wood and stone included, has a spectral centroid that follows
sliding speed, and every category's centroid sits at 240–270 Hz.** The centroid of
this rig's acceleration describes the probe, not the surface. That is a property
of the dataset, and anyone using Cluster's acceleration for texture rendering needs
to know it.

**Spectral-shift exponent** (EXPLORATORY, added 2026-11-06). The centroid is one
number; a fixed resonance could dominate it while a texture peak beneath still
scales. This aligns whole log spectra at each velocity against 40 mm/s and fits the
best log-frequency shift on log(v/40). "Whitened" first subtracts the median log
spectrum of all 118 materials at the same velocity and force, the part shared by
every surface. Tests on synthetic tones (scaffolding only) recover 1 for a scaling
tone, 0 for a fixed one, and recover scaling under a shared resonance only after
whitening.

| | Raw | Whitened |
|---|---|---|
| Median over 118 materials [IQR] | 0.159 [0.106, 0.204] | −0.001 [−0.135, 0.084] |
| Within 0.25 of 1 | 0 of 118 | 0 of 118 |
| vs 1 (Wilcoxon) | p = 4.2×10⁻²¹ | p = 4.2×10⁻²¹ |
| Cloth median / rest median | 0.117 / 0.164 | 0.218 / −0.022 |
| Cloth vs rest (Mann-Whitney) | p = 0.12 | p = 3.0×10⁻⁴ |
| Cloth vs 0 (Wilcoxon) | | p = 0.009 |

Not one material's spectrum moves as v/λ. **Post hoc, and to be reported as
such:** the cloth-vs-rest test on the whitened exponent was added after seeing
the figure. Once the shared spectrum is removed, cloth is the only category that
shifts upward with speed at all, median 0.22, with a cluster of materials at
0.3–0.65.

**Confirmed on held-out data.** Run `velocity_confirm_cloth/20261105T203252_596852Z`,
pre-registered in PREREGISTRATION.md ("Confirmation: cloth spectra shift with
velocity after whitening") before the held-out recordings were analysed. Held out:
cloth at the six other directions, both repeats, plus the second repeat at 0° and
90°. 2,240 recordings, 40 skipped, giving 444 (material, force, direction, repeat)
cells. The whitening reference comes only from the original subset.

| | Result |
|---|---|
| Cloth held-out median exponent [IQR] | **0.338** [0.258, 0.388] |
| Cloth materials above 0 | 16 of 16 (range 0.131–0.573) |
| 1. Cloth > 0 (Wilcoxon, one-sided) | p = 1.5×10⁻⁵, Holm 1.5×10⁻⁵ |
| 2. Cloth > other 102 materials (Mann-Whitney, one-sided) | p = 5.4×10⁻¹⁰, Holm 1.1×10⁻⁹ |

Both predictions pass. **Relative to the response shared by all 118 surfaces,
cloth's spectrum moves up with sliding speed, about a third as far as f = v/λ
predicts; no other category's does.** Highest: Corduroy 0.57, Cotton 0.54, Lace
0.51, Velvet 0.48. Lowest: Mesh 0.13, Chiffon 0.13. That ordering, coarse
structure high and sheer fabric low, is descriptive and was not predicted. What
this does not show: the mechanism. A partial log-frequency shift is also what a
velocity-dependent change in spectral tilt would produce, such as damping by
compliant fabric changing with speed. Identifying it needs the surface profile,
which the dataset does not have. Figure: `figures/paper/fig3_velocity_exponents`
(hollow markers).

---

## Round 2, A — Does looks predict feel? (pre-registered)

Run `looks_feel/20261107T082554_607613Z`. Pre-registered in PREREGISTRATION.md
("Benchmark and findings round 2", A, commit `545702e`) before it was run.
Visual distance is the cosine distance between per-material mean ImageNet
ResNet-50 features of the dataset's 100 crops. Haptic distance is the Euclidean
distance between per-material mean log acceleration spectra (2,315 in-band
recordings, directions 0°/90°, first repeat). Tests are one-sided Spearman Mantel
tests with 9,999 permutations, Holm-corrected.

| Test | r | p | Holm p | Predicted | Outcome |
|---|---|---|---|---|---|
| All 118 materials, free permutation | +0.154 | 0.0011 | 0.0033 | significant | **held** |
| All 118, permuted within category | +0.154 | 0.211 | 0.422 | not significant | **held** |
| 16 cloth materials only | −0.095 | 0.833 | 0.833 | not significant | **held** |

Secondary, descriptive: with the four-number descriptor in place of the spectrum,
all 118 give r = +0.117 (p = 0.011) and cloth gives r = −0.018 (p = 0.49). The
spectrum and descriptor distances agree with each other at r = +0.749.

**Reading.** Materials that look alike vibrate a little alike, but only because
both distances separate categories: wood looks and feels unlike cloth. Permuting
within category removes the effect. Among the 16 fabrics, visual similarity
carries no information about vibration similarity. This is why appearance-based
retrieval in O3 could not beat the mean descriptor. It also limits any
image-to-vibration system that relies on generic ImageNet features to tell one
fabric from another.

---

## Round 2, B — Contact audio vs acceleration (pre-registered)

Run `audio_vs_accel/20261107T110948_431117Z` (Kaggle T4, code `c0efe48`). Same
2,493 cloth gate recordings, same CNN and direction-held-out folds as
`o1_learned_baseline`, 3 seeds. Chance 6.25% (16 classes).

| Input | Accuracy | SD over folds | Fold range | Macro-F1 |
|---|---|---|---|---|
| Audio, 20–8000 Hz (`audio_full`) | **0.720** | 0.048 | 0.640–0.807 | 0.719 |
| Audio, 20–1000 Hz (`audio_band`) | 0.573 | 0.056 | 0.462–0.656 | 0.572 |
| Acceleration, rerun with the new trainer | 0.381 | 0.073 | 0.235–0.458 | 0.375 |

The acceleration rerun reproduces the committed baseline (0.378) to within 0.003,
so the vectorised trainer did not change the result.

Confirmatory test (audio_full > o1_learned_baseline, one-sided Wilcoxon over the
8 direction folds): audio wins every fold by 0.22–0.45, p = 0.0039. **Prediction
held.**

`audio_band` was not part of the confirmatory test and is reported descriptively.
It shows that most of audio's advantage remains inside the 20–1000 Hz band a phone
actuator can reproduce. So the gap is not only high-frequency content. Audio here
is the dataset's noise-cancelled contact microphone, a different sensor from the
rubber-tip accelerometer. This result says which recorded signal identifies a
fabric best; it does not say which one feels right when rendered.

## Round 2, C — Recognition at an unseen velocity (pre-registered)

Run `velocity_warp/20261107T111741_349222Z`. Leave-one-velocity-out (5 folds),
log-frequency spectrogram warped by f·(v/40)^k before the CNN, 3 seeds. A
recording counts as correct if the majority of seeds classify it correctly. Tests
are exact one-sided sign tests on the recordings where the two conditions
disagree, Holm-corrected.

| Warp k | Accuracy | 20 mm/s | 30 | 40 | 50 | 60 |
|---|---|---|---|---|---|---|
| 0 (no warp) | 0.279 | 0.126 | 0.306 | 0.315 | 0.386 | 0.267 |
| **0.338 (confirmed exponent)** | **0.305** | 0.095 | 0.316 | 0.354 | 0.421 | 0.347 |
| 1 (pure f = v/λ) | 0.284 | 0.085 | 0.228 | 0.286 | 0.491 | 0.345 |

| Prediction | Wins / losses | p | Holm p | Outcome |
|---|---|---|---|---|
| k = 0.338 beats k = 0 | 307 / 251 | 0.0099 | 0.020 | **held** |
| k = 0.338 beats k = 1 | 331 / 292 | 0.064 | 0.064 | **not supported** |

**Reading.** Warping by the exponent measured in the velocity analysis helps a
classifier generalise to a velocity it never saw. The gain is modest (+2.6
points), and every condition fails at the 20 mm/s extreme, where the test velocity
lies outside the training range. The advantage over the pure kinematic warp (k = 1)
was not significant, so the claim stops at "partial warp beats no warp". It does
not extend to "partial warp beats full warp". Fold SD (0.10–0.15) is larger than
any of the differences, so per-fold comparisons should not be read on their own.

## Round 2, D — 118-material benchmark (descriptive)

Runs `benchmark118_linear/20261107T103411_132525Z` (CPU) and
`benchmark118_cnn/20261107T125519_960150Z` (T4). 18,569 in-band recordings: all 118
materials, 8 directions, 5 velocities, 2 forces, 2 repeats. 266 files failed to
load and 45 were out of band. Majority baselines: 0.009 for material, 0.323 for
category (wood is the largest category).

| Task | Split | raw4 LR | bands LR | bands+raw4 LR | CNN (3 seeds) |
|---|---|---|---|---|---|
| Material, 118-way | direction held out | 0.031 | 0.078 | 0.083 | **0.202** (top-5 0.448) |
| Material, 118-way | repeat | 0.035 | 0.107 | 0.111 | **0.338** (top-5 0.700) |
| Category, 9-way | direction held out | 0.347 | 0.354 | 0.355 | **0.449** (macro-F1 0.301) |
| Category, 9-way | repeat | 0.352 | 0.387 | 0.394 | **0.631** (macro-F1 0.425) |

CNN category recall with direction held out: cloth 0.71, wood 0.54, leather 0.52,
polymer 0.49, stone 0.28, metal 0.22, glass 0.04, ceramic 0.04, composite 0.00.

**Reading.** The four-number descriptor barely beats the majority baseline at
category level (0.347 vs 0.323), so it is a rendering parameterisation, not a
recogniser. The CNN reaches 20% on 118-way material identification across held-out
directions, 24× chance. That is a usable baseline for this dataset, but well short
of reliable identification. Cloth is the most recognisable category from
vibration. The three smallest categories (composite 3, glass 5, ceramic 5
materials) are almost never recognised. The gap between the direction and repeat
splits (0.20 vs 0.34) shows that sliding direction changes the signal far more
than repeating the same stroke does.

---

## Round 3, E — Direction structure (pre-registered)

Run `direction_structure/20261107T140954_426324Z`. Registered in PREREGISTRATION.md
("Round 3", commit `d3b3bf9`) before any distance was computed. 18,569 in-band
recordings; features are 16 log bands + log RMS, standardised. Distances are
Euclidean, within a (material, velocity, force, repeat) cell, averaged per
material.

| Median per material | Opposite (θ, θ+180°) | Perpendicular (θ, θ+90°) | Diagonal (±45°, ±135°) | Same direction, other repeat |
|---|---|---|---|---|
| Cloth (16) | 3.22 | 3.83 | 4.91 | 1.66 |
| Other (102) | 2.86 | 4.00 | 5.83 | 1.33 |

- Cloth, perpendicular > opposite: 16/16 materials, Wilcoxon p = 1.5×10⁻⁵ (Holm
  1.5×10⁻⁵). **Prediction held.**
- Other 102 materials: 102/102, p = 1.8×10⁻¹⁸ (Holm 3.6×10⁻¹⁸).

**Reading. The prediction held, but the reason we registered for it does not
hold up.** We expected a fabric's warp and weft to make strokes along one axis
alike. The effect, however, is present in every category, and it is strongest
in materials with no weave: glass (median gap 1.60), metal (1.49), ceramic (1.47).
Cloth shows it least (0.63). Diagonal strokes are the most dissimilar of all. They
also have a lower centroid (median 223 Hz vs 251 Hz for 0°/90°/180°/270°), so the
diagonal moves of the stage, which drive both axes at once, produce a different
signal. Direction in this dataset is therefore dominated by the stage's motion
axes, not by the surface. This is a second rig property, alongside the 240–270 Hz
centroid. It likely contributes to the hard 90°/270° folds and the large
direction-vs-repeat gap in D, though we have not tested that link. Even opposite
strokes are about twice as far apart as two repeats of the same stroke.

## Round 3, H — Force transfer (pre-registered)

Run `force_transfer/20261107T140925_493108Z`. Logistic regression on the
`bands+raw4` features. The model is trained at one force, repeat 0, and tested on
repeat 1 at the same or the other force; both ways round, pooled.

| | Same force | Other force | Materials lower at other force | Wilcoxon p (Holm) |
|---|---|---|---|---|
| Cloth, 16-way | 0.324 | 0.224 | 12/16 | 0.0046 (0.0046) |
| All 118, 118-way | 0.126 | 0.067 | 84/118 | 3.9×10⁻¹³ (7.8×10⁻¹³) |

**Both predictions held.** Changing the force from 0.5 N to 1.0 N, or back, costs
about a third of the cloth accuracy and half of the 118-way accuracy.
Descriptive check: we removed level by subtracting log RMS from every band and
dropping the intensity, RMS and duration features, keeping pattern and centroid.
Accuracy then falls to 0.294 → 0.219 (cloth) and 0.109 → 0.057 (118). Most of the
force cost therefore remains after level is removed: force changes the spectral
shape, not only the amplitude. A renderer conditioned on a single recorded force
will be mismatched to a user who presses differently.

## Round 3, F — What survives the on/off channel (pre-registered)

Run `onoff_channel/20261107T142357_330804Z`. The 2,493 cloth gate recordings,
every one rendered. For each source, both band-passed to 20–1000 Hz, the first
500 ms of the amplitude envelope is encoded with scheme B (10 ms tick) and passed
through the assumed 25 ms actuator lag. Features: log modulation power in 12
bands (1–50 Hz) plus log mean level. 16-way logistic regression, direction held
out. Chance 0.0625.

| Source | Intended envelope | Rendered (on/off + lag) |
|---|---|---|
| Acceleration | 0.118 | 0.099 |
| Contact audio | 0.099 | 0.091 |

| Prediction | p | Holm p | Outcome |
|---|---|---|---|
| Rendered acceleration above chance | 0.0039 | 0.012 | **held** |
| Rendering loses identity (acceleration) | 0.016 | 0.031 | **held** |
| Audio vs acceleration after rendering (two-sided) | 0.31 | 0.31 | no difference |

**Reading.** An on/off phone vibration keeps a trace of fabric identity: 9.9%
against 6.25% chance on 16 fabrics. It is a small trace. The envelope itself
carries little to begin with (11.8%), far less than the spectrum (37.8% for the
CNN), because the actuator channel discards the carrier frequency and keeps only
the slow amplitude pattern. Rendering removes about a sixth of what the envelope
had. Audio's large advantage in B (72% vs 38%) disappears here. It lives in the
spectrum, which an on/off motor cannot reproduce. A one-motor phone is therefore
an amplitude-pattern display: whatever makes fabrics distinct in the recordings
mostly does not reach the fingertip through this channel. Untested: whether
people can perceive the residual 10% (O5).

## Round 3, G — Scan → product-photo transfer (pre-registered; the headline question)

Run `photo_transfer/20261107T144009_695846Z`. Registered in PREREGISTRATION.md
("Round 3", G, commit `2b1b3c7`) before any image was opened. Three fabrics present
in both sources:
- cotton: Cluster 104–106;
- chiffon: Cluster 107;
- leather: Cluster 87–101.

The scan training set is crops 1–70 of each material, with crops 71–100 held
out. The photos are DeepFashion-MultiModal product photos whose caption names
exactly one fabric, drawn with seed 0: 100 cotton, 100 chiffon, and only 11
leather, because only 11 qualified. The model is a frozen ImageNet ResNet-50 with
class-balanced logistic regression. The metric is balanced accuracy; chance is
0.333.

| Test set | Balanced accuracy | 95% bootstrap CI | Recall: cotton / chiffon / leather |
|---|---|---|---|
| Held-out scan crops | **1.000** | [1.000, 1.000] | 1.00 / 1.00 / 1.00 |
| Product photos | **0.234** | [0.173, 0.311] | 0.17 / 0.44 / 0.09 |

- Photo < scan: in all 10,000 bootstrap resamples the photo score fell below the
  scan score (p < 10⁻⁴). **Prediction held.** The gap is 0.77.
- Descriptive:
  - train on photos, test on all scan crops: 0.526;
  - photo-to-photo 5-fold cross-validation: 0.544.

**Reading.** A classifier that is perfect on held-out crops of the scanned
fabrics falls *below chance* on product photographs of the same fabric types.
The photo CI lies entirely under 0.333. Most cotton shirts are called chiffon (65
of 100), and leather is almost never recognised. The scan score is optimistic, as
the pre-registration said: the test crops come from the same materials, so the
model can memorise a swatch rather than learn "cotton". Even so, the transfer
does not merely degrade; it fails. The same features separate the fabrics within
the photo domain (0.544), so the photos carry fabric information. A model trained
on 300 dpi lab scans simply does not reach it. The reverse direction (photos →
scans, 0.526) works better than scans → photos.

Limits:
- three classes, with only 11 leather photos;
- the photo labels come from captions;
- a frozen ImageNet backbone, not fine-tuned.

This is the first measurement of open question 1. It supports the warning that
lab-scan haptic datasets cannot be paired with shop photographs without domain
adaptation.

## Round 4, I — Removing the rig's shared response (pre-registered)

Run `rig_whitening/20261107T152134_578254Z`. Plan: PREREGISTRATION.md "Round 4"
(commit `a86cb06`). From each of the 17 band features (16 bands + log RMS), we
subtract its median over all 118 materials at the same (velocity, force,
direction). This is label-free; it does use the test direction's recordings of
every material. Logistic regression, direction held out.

| | Raw | Whitened | Folds improved | p (Holm) |
|---|---|---|---|---|
| Cloth, 16-way | 0.218 | **0.306** | 8/8 | 0.0039 (0.0078) |
| All 118, 118-way | 0.078 | **0.133** | 8/8 | 0.0039 (0.0078) |

**Both predictions held.** Removing what every surface shares at a given speed,
force and direction raises linear identification by 40% (cloth) and 70% (118
materials). That's a gain of 9 and 5.5 points, from a one-line, label-free
correction. With direction held out, the whitened linear model (0.306) closes most
of the gap to the 16-fabric CNN (0.378). The rig properties found in round 1
(fixed centroid) and round 3 (motion axes) are not just curiosities: correcting
for them is the single most effective step for this dataset.

## Round 4, K — Test–retest reliability (descriptive)

Run `reliability/20261107T152135_235452Z`. ICC(3,1) of repeat 0 vs repeat 1
over 9,145 (material, velocity, force, direction) cells. Split-half
(Spearman–Brown) reliability of per-material means over 118 materials. Threshold
fixed in advance: ICC ≥ 0.75.

- **19 of 21 features meet the threshold.**
  - Centroid 0.90, intensity 0.95, entropy 0.94.
  - Bands 1–15: 0.85–0.96.
  - Log RMS 0.95.
- **Two fall short:**
  - envelope period (`duration_ms`), ICC 0.56;
  - the lowest band (20–25 Hz), 0.68.
- Per-material means are highly reliable for every feature (split-half 0.97–0.99).

**Reading.** The recordings are not noisy. Repeating a stroke reproduces the
features almost exactly. So the poor material separation in O1 (silhouette −0.16,
material ω² 0.07–0.16) is not measurement noise. It comes from systematic
variation with condition: direction, velocity and force, which E, H and I show
are largely rig- and contact-driven. The one weak descriptor, the envelope
period, is also the one the on/off renderer depends on most. That compounds F's
finding that little identity survives rendering.

## Round 4, J — Scan → photo, stricter, and with a photo-trained model (pre-registered)

Run `photo_transfer_strict/20261107T154450_969423Z`.

**J1, leave-one-material-out scans (cotton vs leather).** This removes G's
optimism: every scan material is now predicted by a model that never saw it.

| Test | Balanced accuracy | 95% CI |
|---|---|---|
| Unseen scan materials | 0.675 | [0.647, 0.705] |
| Product photos (100 cotton, 11 leather) | 0.287 | [0.141, 0.439] |

Chance is 0.5. Photo < scan in every bootstrap resample (Holm p < 10⁻⁴).
**Prediction held.** With unseen swatches the scan score drops from 1.000 to
0.675, so G's perfect scan score was swatch memorisation, as disclosed. The photo
score stays below chance either way.

**J2, CLIP zero-shot** ("a photo of {cotton|chiffon|leather} fabric").

| Test | Balanced accuracy | 95% CI | Recall: cotton / chiffon / leather |
|---|---|---|---|
| Scan crops (1,900) | 0.460 | [0.451, 0.468] | 1.00 / 0.00 / 0.38 |
| Product photos (211) | **0.635** | [0.528, 0.740] | 0.58 / 0.78 / 0.55 |

Photo > scan (Holm p = 0.0007). **Prediction held.**

Descriptive: we trained a linear probe on the scans' CLIP features. It is perfect
on held-out scan crops (1.000) and at chance on photos (0.326).

**Reading.** The domain gap runs in both directions and does not depend on the
backbone:
- Trained on scans (ResNet or CLIP features), a model is useless on photos.
- A model built from web photographs (CLIP) works on product photos (0.64, chance
  0.33) but not on the scans (0.46; it calls every chiffon scan something else).

Together with G this answers open question 1. Lab scans and shop photos are
different visual domains for fabric. A shopping pipeline should recognise the
fabric with a photo-domain model, never with one trained on the haptic dataset's
scans, and only then look up the haptic signature. Limits: three classes, 11
leather photos, labels taken from captions.

## Round 4, L — Image → category → descriptor at 118 scale (pre-registered)

Run `image_to_descriptor/20261107T153643_203552Z`. Leave-one-material-out:
1. A logistic regression on ResNet-50 crop features from the other 117 materials
   predicts the held-out material's category, by majority vote over its 100 crops.
2. That category's mean descriptor (from the other materials) is the prediction.

Baselines: the all-materials mean, and an oracle that knows the true category.

- **Category from image:** 0.729 (majority baseline 0.322). Wood and stone are
  always right, cloth 0.56, leather 0.53; glass and composite are never right.

| Dimension | MAE, image category | MAE, global mean | MAE, oracle category | Image better | p (Holm) | Held |
|---|---|---|---|---|---|---|
| log centroid | 0.040 | 0.048 | 0.039 | 74/118 | 0.0086 (0.034) | **yes** |
| log intensity | 0.111 | 0.118 | 0.099 | 67/118 | 0.32 (0.41) | no |
| pattern (entropy) | 0.037 | 0.042 | 0.032 | 73/118 | 0.16 (0.41) | no |
| log duration | 0.207 | 0.213 | 0.168 | 74/118 | 0.14 (0.41) | no |

**One of four predictions held.** Looking at a material's image gets its category
right three times in four. Even so, that improves the predicted vibration only
for the spectral centroid, and only by a little (MAE 0.048 → 0.040 in log Hz,
about 8%). The oracle shows why: even the *true* category barely beats the global
mean. Category means are close together, which matches round 1's finding that
the centroid sits at 240–270 Hz for every category. At 118-material scale,
choosing a vibration by recognising the material category is nearly as good as
it gets, and it is only slightly better than playing the average vibration for
everything. This is the 118-material version of O3's null retrieval result.

## Round 5, N — Does whitening rescue the O1 gate? (pre-registered: no)

Run `gate_whitened/20261107T155735_142723Z`. The gate's own metrics and
thresholds were recomputed on the 2,493 cloth recordings, before and after
whitening. The registered gate result is unchanged.

| Features | Silhouette raw | Silhouette whitened | Effective classes (raw → whitened) | Collapsed pairs after |
|---|---|---|---|---|
| 4 descriptors (logs) | −0.115 | −0.182 | 1 → 1 | 120/120 |
| 17 band features | −0.072 | −0.145 | 1 → 1 | 120/120 |

Prediction (bands, whitened > raw): mean per-recording change −0.073, p = 1.0.
**Prediction failed. Whitening makes the clusters worse, not better.** The raw
4-descriptor silhouette here (−0.115) differs from the gate's −0.164 because this
uses log descriptors from the round-2 extraction; the gate's own number stands.

**Reading.** Whitening helps a *trained* classifier (I) but hurts *unsupervised*
cluster quality. Our explanation, untested, comes from O below: cloth keeps a
genuine velocity dependence that the other materials lack (k ≈ 0.34, confirmed
in round 1). Subtracting a median that the 102 non-cloth materials dominate
therefore over-corrects cloth at the velocity extremes and spreads each fabric's
cloud. Either way, no simple normalisation makes the 16 fabrics form separable
clusters.

## Round 5, O — Velocity invariance after whitening (pre-registered; open question 2)

Run `velocity_invariance/20261107T155735_373772Z`. For each material: the SD
across the 5 velocities of the per-velocity mean of each band feature, then the
median over the 17 features.

- Median over materials: raw **0.541** → whitened **0.145**. It is lower in 115
  of 118 materials (Wilcoxon p = 2.7×10⁻²¹). **Prediction held.**
- By category (median, raw → whitened):

| Category | Raw | Whitened |
|---|---|---|
| composite | 0.54 | 0.08 |
| wood | 0.54 | 0.10 |
| ceramic | 0.57 | 0.12 |
| stone | 0.50 | 0.13 |
| leather | 0.53 | 0.16 |
| metal | 0.72 | 0.17 |
| polymer | 0.64 | 0.17 |
| glass | 0.73 | 0.18 |
| **cloth** | 0.41 | **0.29** |

**Reading.** About 73% of the velocity dependence of the spectral features is
shared by all surfaces, so it belongs to the rig. Removing it makes most materials
nearly speed-invariant. Cloth is the exception: it keeps twice the residual
velocity dependence of any other category. That is the same cloth-specific
velocity effect confirmed in round 1 and used in C, now seen at the level of band
features. For rendering, this means:
- non-cloth textures can be stored at one speed once the rig response is removed;
- fabrics need speed-conditioned signatures.

## Round 4, M — Contact audio at 118-material scale (pre-registered)

Run `audio118/20261107T171843_782790Z`, computed on a Kaggle CPU kernel
(`jashwanth113/cluster-audio118`; all 37,760 files fetched, 18,614 audio
recordings, 18,569 matched to in-band acceleration). Logistic regression,
direction held out (8 folds), 118-way material identification, chance 0.0085.

| Features | Accuracy | Macro-F1 | Top-5 |
|---|---|---|---|
| Acceleration bands (20–1000 Hz) | 0.078 | 0.057 | 0.233 |
| Audio bands, 20–1000 Hz | 0.103 | 0.090 | 0.294 |
| Audio bands, 20–8000 Hz | **0.225** | **0.209** | **0.472** |

- **Prediction (audio 20–8000 Hz > acceleration bands): held.** Higher in 8 of 8
  folds, one-sided Wilcoxon p = 0.0039 against both the registered run
  (`benchmark118_linear/20261107T103411_132525Z`, 0.078) and the matched rerun.
- Descriptive: audio in the same 20–1000 Hz band beats acceleration only modestly
  (0.103 vs 0.078). Most of audio's advantage comes from content above 1 kHz,
  which a phone actuator cannot render. Repeat split: 0.267 audio 8k vs 0.107
  acceleration. 9-way category (direction split): 0.485 vs 0.354, majority 0.323.
- This extends round 2's B (cloth only, 0.720 vs 0.378) to all 118 materials.
  The identity information is mostly in frequencies above the vibrotactile range,
  which bounds what any rendered vibration can carry.

## O4 — What survives a binary vibration channel (open question 3)

Run `o4_fidelity_real/20261104T155413_654754Z`. 80 in-band real recordings (20,
40, 60 mm/s; 1.0 N; 0° and 90°), 800 ms patterns, 10 ms tick, motor lag assumed.

- **9 of 80** real envelopes are renderable: most modulation sits above the 25 Hz
  switching limit of a 10 ms tick.
- On renderable envelopes, median log-spectral distance: B 8.4 dB, D (random
  control) 7.5 dB, A 15.7 dB, C 20.5 dB. Wasserstein: B 1.9 Hz, D 2.5 Hz.
- The metric does not separate the best scheme from the random control. Whether
  people can is the P0 phone test and Stage 3 of the study.

---

## What this means for the paper

1. **Headline candidate (O2):** a material classifier trained on photographs calls
   0.756 of photographed fabric "fabric", but 0.033 of 300 dpi flatbed cloth scans,
   at every field of view tried. The scan → photo gap is real and large in the
   direction measured. The reverse direction (scan-trained → product photos) still
   needs the labelled photo set.
2. **Negative result (O1):** four-number vibrotactile descriptors from a sliding
   probe do not separate 16 cloth materials (silhouette −0.16, 1 effective class).
   Direction and repetition variance swamp material. A richer spectrum reaches 0.25
   accuracy against 0.0625 chance, and a CNN on the spectrogram reaches 0.378 with
   direction held out (exploratory). The loss is in the compact descriptor; the
   signal carries more identity than four numbers keep. This is publishable as is.
3. **Dataset finding (velocity, all 118 materials):** on this rig the spectral
   centroid of acceleration does not follow sliding speed for any material (0 of
   118 within 0.25 of the f = v/λ prediction, centroid 240–270 Hz everywhere), so
   v/f normalisation hurts. This is a property of the measurement, not of cloth,
   and it matters for anyone rendering from Cluster.
4. **Confirmed positive result (velocity, cloth):** after removing the spectrum
   shared by all surfaces, cloth alone shifts upward with speed, by about a third
   of v/λ (held-out median exponent 0.338, 16 of 16 materials above 0, Holm p =
   1.5×10⁻⁵ and 1.1×10⁻⁹). It was found post hoc, then pre-registered and
   confirmed on 2,240 held-out recordings. This is the strongest single result in
   the project.
5. **Pre-registration outcome:** confirmatory prediction failed (weave p = 0.23
   Holm) with pile untestable; drape as predicted, weakly.

## Still blocked, needing people

- P0 phone test; P6 product photos and two labellers; P7–P9 ethics, pilot, main
  study; Stages 1–3 and the representability profile (P10–P13) need ratings.
- P5 DINOv2/CLIP embeddings: not run. They feed Stage 1, which needs ratings, so
  they are not needed yet. They can run on CPU, so no GPU is required.
