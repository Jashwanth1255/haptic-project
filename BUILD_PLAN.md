# Build plan — ordered, with prompts

Work top to bottom; the order is by dependency and risk, not by objective
number.

Steps marked **HUMAN** need a person, not a session.

---

## Week 1 — unblock everything

### P0 · HUMAN · Phone test (30 minutes, highest risk in the project)

No code. Load the Objective 4 page on one Android phone. Play encoding schemes
A, B, C and D for two contrasting fabrics. Have two people say whether they can
tell them apart.

**Why first:** Objective 5 assumes the encoders produce perceptibly different
output. Nobody has verified that on real hardware. The 25 ms motor lag is an
assumption, the 10–20 ms granularity is an assumption, and the fidelity numbers
are synthetic. If A, B and C feel identical on a real LRA, the perception study
has nothing to measure — and you want to know that this week, not in month five.

Record: phone model, Android version, whether each pair was distinguishable.

### P1 · HUMAN · Cluster dataset (two browser tabs)

**Steps 1 and 2 done 2026-11-03**, recorded in `docs/CLUSTER_DATASET.md`: the
licence is Apache 2.0, and the 16 cloth names are listed there and in
`haptic/loader.py`. Step 3 done 2026-11-04 with
`python fetch_cluster.py --out data/cluster` (586 MB); the timing checks on the
real files are in the same notes.

1. Open figshare DOI `10.6084/m9.figshare.29438288` and record the **exact dataset
   licence**. The article appears CC BY-NC-ND; the dataset licence is separate. If
   it forbids derivatives you can use the data but may not redistribute a derived
   corpus — which changes what you can publish alongside the paper, not the paper.
2. Download `documents/texture_list.pdf` from
   `github.com/cluster-lab/Cluster-Haptic-Texture-Dataset` and write out **the 16
   cloth material names** (IDs 102–117). These decide your class list.
3. Download the cloth subset.

### P2 · Results persistence

> Every experiment in this repository currently prints its tables and saves
> nothing — the Objective 3 numbers and the Objective 4 fidelity figures exist
> only in closed terminals. Add `io/results.py` with a `save_run(name, config,
> metrics, artifacts)` function that writes to `results/<name>/<timestamp>/` a
> `config.json`, a `metrics.json`, and the current git commit SHA and dirty-tree
> flag. Refactor every existing experiment entry point to call it. Add a
> `load_run` and a `list_runs`. Write tests. This is small but it blocks writing
> the paper — every number quoted there must be traceable to a run.

### P3 · Pre-registration additions

> Add two dated entries to PREREGISTRATION.md, both before any Cluster descriptor
> exists. First, the three-cluster prediction: the pooled DTD confusion matrix
> showed pile (fibrous, matted), weave (braided, woven, interlaced, meshed,
> knitted) and drape (gauzy, pleated, wrinkled) clusters. Predict that pile and
> weave will transfer to haptic descriptor space and drape will not, because
> drape describes three-dimensional fabric behaviour that a sliding contact
> probe does not measure. Second, the representability ordering: predict that
> Stage-2 representability will rank roughness ≈ bumpiness > slipperiness >
> softness > warmth, with warmth as a physically unrepresentable control.
> Mark clearly which predictions are confirmatory (pile/weave, registered
> earlier) and which are exploratory (the drape cluster, found post hoc in
> pooled data).

---

## Weeks 2–3 — the go/no-go and the headline

### P4 · Objective 1 gate on real cloth

> Run the Objective 1 pipeline on the Cluster cloth subset, IDs 102–117. First
> `python -m haptic.cli inspect --root <dataset>` and adjust FILENAME_PATTERNS in
> loader.py if the paths do not parse. Then run the gate. Report: how many
> recordings were flagged out of band and for which materials, the ANOVA with
> Holm-corrected p and omega-squared per descriptor dimension, the pooled and
> per-velocity silhouette for both raw and velocity-normalised descriptors, and
> the pairwise separation matrix with collapsed pairs named. Do not adjust the
> thresholds in separability.py — they were fixed before any real data was seen.
> Save the run with save_run.

**Done 2026-11-04: the gate failed** (silhouette −0.16 raw, −0.12 normalised;
all 120 pairs collapse, 1 effective class). Group assignment registered first
(`a2610b8`). Every real-data number is in `docs/RESULTS.md`.

**This is the go/no-go.** If fewer than 8 materials separate, coarsen the class
list, report the grouping, and continue — a collapsed class list is a finding
about how much fabric identity survives a four-number descriptor, not a failure.

### P5 · Embedding extraction (Kaggle GPU)

> Using vision/encoders.py, extract DINOv2 and CLIP embeddings for every Cluster
> cloth surface scan and for the product-photograph set. Concatenate with
> per-encoder L2 normalisation. Cache to disk as EmbeddingSet files. Also compute
> the CLIP zero-shot attribute scores for the same images and save them
> separately — they are the perception baseline, not the perception model. Ask me
> before starting the GPU run. Verify that image_ids align between the DINOv2 and
> CLIP sets before concatenating.

### P6 · The transfer gap — partially unblocked, do it now

> Measure the domain-shift gap that is the paper's headline. Curate at least 50
> product photographs from DeepFashion or Fashionpedia covering the 12
> pre-registered textile classes, labelled by two people independently with
> agreement reported. Then evaluate the existing DTD-trained classifier on them
> and report accuracy against the DTD test-split accuracy already measured
> (0.812 ± 0.025 on the 12-way restricted evaluation, folds 2–10). Report the gap
> per class, not only in aggregate — we expect woven, already the weakest class
> at 0.575 recall, to degrade worst. Save the run.

The Cluster-scans-versus-product-photos comparison comes later, once P5 is done.
This DTD-versus-product-photos version needs nothing you do not already have.

---

## Weeks 3–4 — instrument and pilot

### P7 · Prepare the rating study

> Prepare study/rating_instrument.html for use. Replace the STIMULI list in
> study/stimuli.js (and PATTERNS, from real descriptors) with the
> real fabric images — it must include the Cluster surface scans, or the
> perception head and the haptic regression train on disjoint material sets and
> no end-to-end claim is possible. Add product photographs alongside them.
> Replace the placeholder ethics paragraph with the institution's required text.
> Delete the generated noise images in study/images. Then re-run the browser test
> to confirm the page still works with the real stimulus count, and check it on a
> phone-width viewport.

### P8 · HUMAN · Ethics and pilot

Submit the ethics application. Then run the pilot with about 10 participants —
ratings and vibration discrimination in one session, as designed.

> Analyse the pilot: run study/reliability.py over the exported CSVs and report
> intra-rater Spearman and ICC(2,k) per attribute. Flag any attribute with ICC
> below 0.50 — raters do not agree on it, and no model should be expected to
> predict it. Also report median time per image and any participant who looks
> like they clicked through. Recommend instrument changes before the main study.

**Do not skip the pilot.** Perception studies that skip it discover the
instrument was broken after the data is collected.

---

## Month 5 — main study and the three stages

### P9 · HUMAN · Main study

40 recruited, 30 usable, Android only.

### P10 · Stage 1 — perception from images

> Build the perception head: ridge regression from the concatenated DINOv2 + CLIP
> embedding to the five mean perceptual attributes, evaluated leave-one-image-out.
> Report Spearman correlation per attribute between predicted and held-out human
> ratings, **and report each against that attribute's ICC(2,k) ceiling from
> study/reliability.py**. A correlation quoted without the human ceiling beside
> it is uninterpretable. Compare against the CLIP zero-shot baseline from P5.
> Save the run.

### P11 · Stage 2 — perception and haptics

> Using vision/regression.py, run the Stage 2 analysis on the real Cluster cloth
> descriptors. Three parts. First, regress each perceptual attribute from the four
> haptic descriptors, leave-one-material-out, with the permutation test — this
> measures which attributes are present in the measurement. Second, run the full
> ablation: embedding only, perception only, embedding plus perception, predicting
> the haptic descriptors. Third, run the Mantel test in haptic/crossmodal.py
> between the visual confusion matrix and the haptic descriptor distance matrix
> over the same materials. Use velocity-normalised descriptors and report the raw
> comparison alongside. Report everything against the mean-predictor baseline.
> Save all three runs.

### P12 · Stage 3 — rendered discrimination

> Analyse the vibration discrimination data: d-prime per fabric pair and per
> encoding scheme, with scheme D as the random control, and a significance test
> between schemes. Then d-prime along each perceptual attribute dimension —
> whether participants' discrimination tracks roughness, bumpiness and so on.
> Report device model breakdown, since LRA and ERM actuators differ. Save the run.

---

## Month 6 — assembly

### P13 · Representability profiles

> Assemble the three-stage representability profile per attribute: Stage 1
> (Spearman against the ICC ceiling), Stage 2 (R² against the mean-predictor
> baseline), Stage 3 (d-prime). Produce one figure with five grouped bars, one
> per attribute, three stages each, with warmth marked as the unrepresentable
> control and roughness as the reference ceiling. Then state plainly whether the
> pre-registered ordering — roughness ≈ bumpiness > slipperiness > softness >
> warmth — held, and report it either way. Load every number from saved runs; do
> not recompute anything for the figure.

### P14 · The paper

Separate workstream, and the draft has its own defect list: 175 raw `{{...}}`
tags, no references section, the 76.36/83.06 contradiction, two different "Lee
et al.", and the PDF paste residue. Fix those before any new content goes in.

---

## What is already built

| Component | Where |
|---|---|
| Descriptor extraction + O1 gate | `haptic/descriptor.py`, `haptic/separability.py`, `haptic/cli.py` |
| Velocity normalisation | `haptic/descriptor.py` |
| Synthetic signals (scaffolding only) | `haptic/signals.py`, `haptic/synth_o3.py` |
| Cluster loader | `haptic/loader.py` |
| Saved runs (P2) | `runlog/` |
| ResNet-50 baseline results | `results/o2/` |
| DTD to product-photo transfer (P6) | `vision/transfer.py`, `python -m vision.cli transfer` |
| DINOv2 + CLIP + zero-shot baseline (P5) | `vision/encoders.py` |
| Small-N-safe regression + ablation | `vision/regression.py` |
| Stage 1 and Stage 2 runners (P10, P11) | `study/stages.py` |
| Rating instrument (P7) | `study/rating_instrument.html`, stimuli in `study/stimuli.js` |
| Placeholder stimuli | `make_study_stimuli.py` |
| Reliability / ICC (P8) | `study/reliability.py` |
| Stage 3 discrimination d' (P12) | `study/discrimination.py` |
| Representability profile (P13) | `study/representability.py` |
| Mantel test | `haptic/crossmodal.py` |
| Pre-registered group test | `haptic/groups.py`, `make_o1_real.py` |
| Exploratory identity analysis | `haptic/identity.py` |
| Classifiers on Cluster scans | `vision/cluster_scans.py` |
| O3 on real data | `make_o3_real.py` |
| O4 fidelity on real signals | `make_o4_table.py --real data/cluster` |
| Real-data results, with run ids | `docs/RESULTS.md` |

388 tests passing. P5 and P7 onwards are tested on invented data only
and has not run on real data.

## Standing rules

- Never present synthetic-data output as a finding. Label it scaffolding.
- Never quote a number that is not in a saved run.
- A negative result is publishable; tuning until it looks good is not.
- Gate thresholds and pre-registered predictions do not move after data is seen.
- Ask before using GPU; most steps here are CPU work.
