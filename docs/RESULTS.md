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
   accuracy against 0.0625 chance. This is publishable as is.
3. **Negative result (O3, velocity):** cloth texture vibration on this rig is
   velocity-invariant in frequency, not spatially periodic, so v/f normalisation
   hurts. Report this.
4. **Pre-registration outcome:** confirmatory prediction failed (weave p = 0.23
   Holm) with pile untestable; drape as predicted, weakly.

## Still blocked, needing people

- P0 phone test; P6 product photos and two labellers; P7–P9 ethics, pilot, main
  study; Stages 1–3 and the representability profile (P10–P13) need ratings.
- P5 DINOv2/CLIP embeddings: not run. They feed Stage 1, which needs ratings, so
  they are not needed yet. They can run on CPU, so no GPU is required.
