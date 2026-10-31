# Objective 1 — haptic descriptor extraction and separability gate

Team 15. Working code for Objective 1 of the tactile virtualization project.

```
haptic/
  descriptor.py     band-pass -> FFT -> RMS -> 4-D descriptor, + velocity normalisation
  signals.py        synthetic texture signals (test scaffolding ONLY)
  separability.py   ANOVA, silhouette, pairwise collapse -> the O1 go/no-go gate
  loader.py         Cluster Haptic Texture Dataset loader
  cli.py            command line
tests/              26 tests, all passing
make_figure.py      produces objective1_velocity_dependence.png
```

## Run it

```bash
pip install numpy scipy scikit-learn pandas matplotlib pytest
python -m pytest tests/ -q                       # 26 passed
python -m haptic.cli gate --synthetic            # end-to-end on synthetic signals
python make_figure.py
```

On the real dataset, once downloaded:

```bash
python -m haptic.cli inspect --root /path/to/texture_dataset   # DO THIS FIRST
python -m haptic.cli gate    --root /path/to/texture_dataset
```

## The four descriptor dimensions

| Dimension | Definition | Meaning |
|---|---|---|
| `frequency_hz` | spectral centroid of band-passed acceleration | how fine the texture is |
| `intensity` | RMS acceleration in band | how strong |
| `pattern` | normalised spectral entropy, [0,1] | low = periodic (rib), high = stochastic (pile) |
| `duration_ms` | dominant envelope modulation period | timing the renderer needs |

Analysis band is 20–1000 Hz: below 20 Hz is hand motion rather than texture, above
1000 Hz the skin does not respond.

Two deliberate choices worth defending at a review:

**`pattern` is spectral entropy, not spectral flatness.** Flatness (Wiener entropy) is
the more common measure and was tried first. It is unusable here: these signals are
narrowband, so out-of-band bins drive the geometric mean to zero and flatness returns
~0.00 for every material regardless of structure. Entropy spreads 0.31–0.67 across the
same materials. Flatness is still reported in dB as an auxiliary feature.

**Effect size is omega-squared, not eta-squared.** Eta-squared is biased upward by
small samples — its null expectation is about (k−1)/(n−1), so a handful of recordings
of *identical* materials produces eta² ≈ 0.25, which reads as a moderate effect and is
not one. Omega-squared removes that bias and can go negative. Holm–Bonferroni
correction is applied across the four dimensions.

## What the pipeline found

Running the gate on synthetic signals produced one result worth carrying into the
project design:

| | pooled silhouette | effective classes |
|---|---|---|
| raw descriptors | **+0.02** | 1 of 12 |
| velocity-normalised | **+0.43** | 9 of 12 |

Per velocity, raw descriptors score 0.18–0.33. Pooled across 20–60 mm/s they collapse
to zero. The materials are highly separable at any *fixed* exploration condition and
not separable at all once velocities are mixed.

The cause is mechanical: a tool sliding at velocity `v` over a surface of spatial
wavelength `λ` excites `f = v / λ`, so the centroid is a property of the stroke, not
the material. Dividing the velocity back out gives candidate material constants:

```
wavelength_mm    = v / frequency_hz
burst_spacing_mm = v * duration_ms / 1000
roughness_index  = intensity / (force * sqrt(v / 40))
pattern          unchanged — already dimensionless
```

### Read this before quoting any of the above

**On synthetic data the normalisation works by construction.** `signals.py` was written
with `f = v / λ` in it, so recovering λ proves the arithmetic and nothing else. Whether
real fabric measurements obey it closely enough for separability to survive pooling is
an open empirical question — and answering it on Cluster's 20–60 mm/s grid is exactly
Objective 3's experiment. Do not report the synthetic number as evidence.

What the synthetic run legitimately establishes: the pipeline runs end to end, the
extractor recovers parameters it is handed (26 tests), and the gate correctly fails
when materials are indistinguishable.

## Band-edge validity

Descriptors carry an `in_band` flag. A surface with a long spatial wavelength explored
slowly excites a fundamental below the analysis band — corduroy at a 2.4 mm period and
20 mm/s sits at 8 Hz — and every descriptor for that recording then describes filter
skirt rather than texture. 66 of 360 synthetic recordings were flagged, all of them
coarse materials at low velocity.

**Expect this on the real cloth subset too**, and decide now whether to exclude those
recordings or widen the band. Do not let them average in silently.

## Using the real dataset

Cluster Haptic Texture Dataset, *Scientific Data* 13, 756 (2026).
Paper DOI `10.1038/s41597-026-06760-z` · data DOI `10.6084/m9.figshare.29438288` ·
code `github.com/cluster-lab/Cluster-Haptic-Texture-Dataset`

Cloth materials are IDs 102–117 (16 of 118). Acceleration is 3-axis at ~6 kHz.

**The on-disk layout is not yet verified** — the figshare record could not be opened
when this was written. `loader.py` fails loudly rather than silently returning nothing.
Run `inspect` first; if the paths do not parse, add a pattern to `FILENAME_PATTERNS`.
That is the only part that should need changing.

Still to confirm:

1. **Dataset licence** on the figshare record. The article appears to be CC BY-NC-ND;
   the dataset licence is separate. If it forbids derivatives you can still use the
   data but may not be able to release a derived corpus.
2. **The 16 cloth names**, in `documents/texture_list.pdf` in the code repository.
   `loader.load_names()` will pick them up from `texture_list.xlsx` automatically.

## Gate thresholds

Set in `separability.py`. Fix them before touching real data so the gate cannot be
moved after seeing the result.

```python
ALPHA           = 0.05   # significance, Holm-corrected
SILHOUETTE_MIN  = 0.30
COLLAPSE_RATIO  = 1.0    # centroid distance / pooled spread
MIN_SEPARABLE   = 8      # project needs at least this many distinct classes
```

A failing gate is not a failed project. It means the class list must be coarsened —
group the collapsed pairs and report the grouping, which is itself a reportable finding
about how much fabric identity survives a four-number descriptor.
