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
