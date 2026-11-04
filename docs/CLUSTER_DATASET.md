# Cluster Haptic Texture Dataset: verified facts

Checked on 2026-11-03. Each line names where it was checked.

## Licence

**Apache 2.0** for the data. The figshare record (`10.6084/m9.figshare.29438288`,
version 6, API field `license`) and the Hugging Face mirror
(`tamago117/cluster-haptic-texture-dataset`, card `license: apache-2.0`) both say so.
Apache 2.0 allows derivatives and redistribution with attribution and the licence
notice, so a derived descriptor corpus can be released alongside the paper. The
article itself has its own licence, which is separate.

## Files

| Source | Form | Size |
|---|---|---|
| figshare | `cluster_haptic_texture_dataset.zip`, all 118 materials | 15.2 GB |
| figshare | `cluster_haptic_texture_dataset_mini.zip`, 10 materials | 760 MB |
| Hugging Face mirror | one Parquet file per recording | 9.9 GB in all |

The mirror lets us fetch the cloth materials alone: `fetch_cluster.py`, pinned to
revision `b7c2fb70`. Accel, position and images for the 16 cloth materials come
to 6,736 files, about 590 MB.

## Layout

From the code repository's `documents/dataset_details.md`:

- Sensor files: `<id>_<direction>_<velocity>_<force>_<repeat>`. Direction is 0–315°
  in steps of 45, velocity 20–60 mm/s, **force in mN** (500 or 1000), repeat 0 or 1.
  That gives 160 recordings per material and 2,560 for the cloth subset, which
  matches the mirror's file count.
- Acceleration: CSV `time,X,Y,Z`, in g, nominally 6 kHz. The documented sample rows
  are unevenly spaced, so `haptic.loader` resamples every recording and reports
  its measured rate.
- Position: CSV `time,X,Y`, in mm, at 100 Hz. The loader uses it to keep only the
  stretch sliding at the nominal speed.
- Images: `images_scan_area/img<id>.jpg` (the 100×100 mm scan area), and
  `crop_images/img<id>/` (100 crops per material, for texture classification).

## Checked on the downloaded files (2026-11-04)

All 6,736 files fetched (586 MB): 160 recordings for each of the 16 materials,
each with its position file. A timing survey of all 2,560 recordings, which
computed no descriptors:

- **Time is `time_ns`** (int64 nanoseconds) in the mirror, not `time` in seconds.
- **Accel rate is about 8.6 kHz, not the documented 6 kHz.** Median step 111 µs on
  every file (about 9 kHz), mean rate 8,642 Hz (range 8,493–8,767), with dropouts
  taking about 0.5% of each recording (longest gap 5.4 ms). The loader low-passes
  before bringing the signal down to the 6 kHz analysis grid, so 3–4.5 kHz content
  cannot alias into the 20–1000 Hz band.
- **Accel and position share a time origin:** position starts a median 0.01 ms after
  accel (range −5.0 to +1.4 ms) and ends a median 4.3 ms before it. Every steady
  window falls inside the acceleration recording.
- **Steady sliding** lasts a median 2.0 s at 20 mm/s down to 1.1 s at 60 mm/s.
  47 recordings (1.8%) have under 0.5 s and are skipped. All are at 30–60 mm/s,
  and no material loses more than 6 of its 160.

## Cloth materials (IDs 102–117)

From `documents/texture_list.pdf`, category "Cloths":

| ID | Name | ID | Name |
|---|---|---|---|
| 102 | Velvet | 110 | Organdy |
| 103 | Corduroy | 111 | Mesh |
| 104 | Cotton | 112 | Wool |
| 105 | Cotton broadcloth | 113 | Nylon |
| 106 | Cotton gauze | 114 | Polyester |
| 107 | Chiffon | 115 | Silk |
| 108 | Lace | 116 | Linen |
| 109 | Georgette | 117 | Cotton linen |

Several names are fibres (cotton, wool, nylon, polyester, silk), not constructions.
A fibre name does not say whether the fabric is woven, knitted or pile.

## Group assignment: a draft, not registered

PREREGISTRATION.md requires each material to be assigned to pile, weave, drape or
neither **from its name and scan alone, before any descriptor is computed**. The
scans have not been looked at yet. This draft uses names only, for the team to
correct against the scans and then record in PREREGISTRATION.md:

| Group | From the name | Unsure until the scan is seen |
|---|---|---|
| Pile | Velvet, Corduroy | Wool |
| Weave | Cotton broadcloth, Linen, Cotton linen, Mesh | Cotton |
| Drape | Chiffon, Georgette, Organdy, Cotton gauze | |
| Neither | Lace | Nylon, Polyester, Silk |

Corduroy has cut-pile wales, but a texture classifier may well call it ribbed or
"pleated". Draft drape has 4 materials and pile 2–3. A group with fewer than 3
materials is reported as untested.
