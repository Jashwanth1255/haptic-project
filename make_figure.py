"""Figure: why raw descriptors cannot be pooled across sliding velocity."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from haptic import descriptor as D, separability as S

df = pd.read_csv("descriptors_synthetic.csv")
df = df[df["in_band"]]

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Objective 1 — velocity dependence of haptic descriptors\n"
             "SYNTHETIC SIGNALS: scaffolding to verify the pipeline, not a result",
             fontsize=11, fontweight="bold")

show = ["satin-slick", "plain-weave-fine", "twill-diagonal", "corduroy-wide",
        "pile-velvet", "felt-matte"]
cmap = plt.get_cmap("tab10")
colors = {m: cmap(i) for i, m in enumerate(show)}

# (a) raw centroid rises with velocity
a = ax[0, 0]
for m in show:
    g = df[df.material_name == m].groupby("velocity_mm_s")["frequency_hz"]
    a.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="o",
               capsize=3, label=m, color=colors[m])
a.set_xlabel("sliding velocity (mm/s)")
a.set_ylabel("spectral centroid (Hz)")
a.set_title("(a) raw 'frequency' moves with velocity", fontsize=10)
a.legend(fontsize=7)
a.grid(alpha=0.3)

# (b) normalised wavelength is flat
b = ax[0, 1]
for m in show:
    g = df[df.material_name == m].groupby("velocity_mm_s")["wavelength_mm"]
    b.errorbar(g.mean().index, g.mean(), yerr=g.std(), marker="s",
               capsize=3, label=m, color=colors[m])
b.set_xlabel("sliding velocity (mm/s)")
b.set_ylabel("recovered wavelength (mm)")
b.set_title("(b) normalised wavelength is flat", fontsize=10)
b.grid(alpha=0.3)

# (c) silhouette per velocity, raw vs normalised
c = ax[1, 0]
raw = S.separability_by_velocity(df, D.Descriptor.CORE)
nrm = S.separability_by_velocity(df, D.NORMALISED_CORE)
labels = [str(v) for v in raw["velocity_mm_s"]]
x = np.arange(len(labels))
c.bar(x - 0.2, raw["silhouette"], 0.4, label="raw", color="#c44")
c.bar(x + 0.2, nrm["silhouette"], 0.4, label="velocity-normalised", color="#48a")
c.axhline(S.SILHOUETTE_MIN, ls="--", c="k", lw=1,
          label=f"gate threshold {S.SILHOUETTE_MIN}")
c.axhline(0, c="k", lw=0.8)
c.set_xticks(x)
c.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
c.set_ylabel("silhouette score")
c.set_title("(c) separability: per velocity, then pooled", fontsize=10)
c.legend(fontsize=8)
c.grid(alpha=0.3, axis="y")

# (d) pooled descriptor space, raw vs normalised
d = ax[1, 1]
for m in show:
    g = df[df.material_name == m]
    d.scatter(g["frequency_hz"], g["intensity"], s=14, alpha=0.6,
              color=colors[m], label=m)
d.set_xlabel("spectral centroid (Hz)")
d.set_ylabel("RMS intensity")
d.set_title("(d) raw space, all velocities pooled — classes smear", fontsize=10)
d.grid(alpha=0.3)

plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig("objective1_velocity_dependence.png", dpi=150)
print("wrote objective1_velocity_dependence.png")

print(f"\npooled raw        : {S.cluster_quality(df, D.Descriptor.CORE):+.3f}")
print(f"pooled normalised : {S.cluster_quality(df, D.NORMALISED_CORE):+.3f}")
