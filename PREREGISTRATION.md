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
