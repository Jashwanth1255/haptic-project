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
