# Objective 3: visual-to-haptic association

**No result from this code is valid until real Cluster cloth descriptors exist.**
Everything so far has run on `haptic/synth_o3.py`, where the images and the descriptors
are both invented. Objective 1's gate has to pass on real data first. Do not quote any
number from `make_o3_synthetic.py`.

## Strategies (`haptic/association.py`)

| Strategy | Query | Score for each material |
|---|---|---|
| Class pair | MINC and DTD class probabilities | P(its material class) × P(its texture class) |
| Cosine | ResNet-50 embedding from `vision/embed.py` | cosine with its mean reference embedding |

A real embedding is the MINC and DTD models' L2-normalised 2048-d vectors joined together.

Materials with the same class pair cannot be told apart by the lookup, and that is
likely on real cloth because MINC may call nearly all of it "fabric". Ties get
fractional credit in top-k, and a tied substitute is the mean of the tied descriptors.
That mean shrinks towards the database average, which lowers its open-set MAE, so
report how many distinct pairs exist next to any class-pair score.

## Protocols

They answer different questions, so neither replaces the other.

**Closed-set: did it retrieve the right material?** Every material is in the database.
Views are split into 4 folds. Each fold queries in turn and the rest build the database,
so a query never shares a view with the database. Report top-1 and top-3 per fold, with
chance at 1/N and 3/N.

**Open-set: how wrong is the nearest substitute?** Leave-one-material-out, one fold per
material: 16 folds with 15 materials in the database. The held-out material's descriptor
and views are absent from that fold's database. Report MAE per dimension against the
held-out truth, both raw and divided by the spread of that fold's database.

Report the mean, std, min and max across folds, and the per-fold table. With 16
materials one odd fabric can move a mean. Folds overlap, so the std is spread between
folds and not a standard error.

Open-set MAE only means something beside its two baselines: the database mean, and a
random substitute. A retrieval method that loses to the mean is worse than ignoring
the image.

### Real data needs disjoint views

Cluster has one scan per material. Cut each scan into a grid with `vision.embed.tiles`,
which keeps pixels disjoint. Without this, retrieval only finds the same image again.

Scan tiles are all the same imaging condition, so closed-set accuracy here says nothing
about product photos. That is Objective 2's transfer question.

## Velocity experiment (`haptic/velocity.py`)

| Approach | Descriptor returned |
|---|---|
| Agnostic | the material's descriptor averaged over velocities, the same at every query velocity |
| Conditioned | the normalised descriptor, with the query velocity re-applied by `descriptor.denormalise` |

Both use the same reference recordings at the query's force. Lower repeats build the
reference and higher repeats are queries, which on Cluster are the sliding directions.
Errors are averaged within a material before testing, so n is the material count.
`compare` gives MAE, percentage improvement and a Holm-corrected Wilcoxon test per
velocity and dimension, with a verdict. Run it with the right material given, to isolate
velocity, and with an open-set substitute, to see the deployed case.

Pass `exclude_query_velocity=True` to also test a speed the reference never saw.

### Why synthetic results here say nothing

`signals.py` contains f = v/λ, so conditioning wins by construction, and recovering
that proves only the arithmetic. The `velocity_free` recordings ignore velocity while
keeping their labels, which makes the relations false. There the experiment must report
"conditioning hurts", and a test requires that it does. That is the evidence it can give
a negative answer on real fabric.

### Reading a null verdict

Holm correction across a 20-cell table needs p < 0.0025. A Wilcoxon test on n pairs
cannot go below 2/2ⁿ, so 8 materials can never reach significance, and `compare` warns
when this applies. Out-of-band exclusions lower n, mostly at slow velocities.

Exclusions also change which materials each velocity covers, so compare `n_materials`
before comparing velocities. Coarse fabrics drop out at 20 mm/s.

## Known issues for real data

- **`duration_ms` is heavy-tailed.** On synthetic signals 98.5% of medians were within
  10% of the true period, but a rare recording locks onto a spurious low-frequency peak,
  such as 18 ms becoming 341 ms. Reference descriptors therefore use the median within a
  velocity. Query errors keep their outliers, so scoring stays honest. Expect this to be
  worse on real recordings.
- **`in_band` only checks the centroid.** A recording can pass with a bad envelope:
  synthetic `corduroy-wide` at 20 mm/s passes with `duration_ms` about 2.5 times too long.
  Consider an envelope validity check in Objective 1 before running this.
- **The class pairs in `synth_o3` were assigned by hand from fabric names**, so
  similar fabrics share a class by design and class-pair open-set scores are flattering.

## Running it

```bash
python make_o3_synthetic.py
```

The output is scaffolding: it shows the protocols run and can fail.
