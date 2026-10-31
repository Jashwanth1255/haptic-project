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

## Objective 1 gate thresholds

As set in `haptic/separability.py`:

| Constant | Value | Meaning |
|---|---|---|
| `ALPHA` | 0.05 | ANOVA significance level, applied to Holm-corrected p-values |
| `SILHOUETTE_MIN` | 0.30 | Minimum silhouette score in descriptor space |
| `COLLAPSE_RATIO` | 1.0 | Centroid distance / pooled spread; a pair below this has collapsed |
| `MIN_SEPARABLE` | 8 | Minimum effective classes after merging collapsed pairs |
