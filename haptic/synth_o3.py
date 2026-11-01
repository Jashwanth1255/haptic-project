"""Synthetic stand-ins for Objective 3. SCAFFOLDING ONLY, NEVER A RESULT.

The Cluster cloth descriptors do not exist yet, and neither do embeddings of
Cluster's scans. Everything here is invented so that `association.py` and
`velocity.py` can be built and tested. Two controls exist to let those experiments
fail, because a generator that only produces the wanted answer tests nothing:

    coupling        how far visual similarity follows haptic similarity. At 0 the
                    embeddings know nothing about touch, so retrieval must fall to
                    its baseline.
    velocity_free   recordings that ignore velocity, so re-applying the velocity
                    relations is wrong and conditioning must lose.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import descriptor as D
from . import signals as G

N_MATERIALS = 16        # as in Cluster's cloth set; the fabrics themselves are fictional
VELOCITIES_MM_S = G.VELOCITIES_MM_S
FORCES_N = G.FORCES_N

EXTRA_FABRICS = (
    G.SyntheticMaterial("canvas-rough", 0.85, 0.66, 1.9, 0.75),
    G.SyntheticMaterial("linen-slub",   0.40, 0.40, 1.2, 0.60),
    G.SyntheticMaterial("chiffon-sheer", 0.18, 0.35, 0.6, 0.40),
    G.SyntheticMaterial("terry-loop",   0.75, 0.25, 2.0, 1.40),
)
FABRICS = (*G.FABRIC_BANK, *EXTRA_FABRICS)

# A plausible classifier outcome: nearly all cloth is one MINC class, and several
# fabrics share a DTD class, so pairs collide as they are likely to on real cloth.
MATERIAL_CLASSES = ("fabric", "carpet", "leather")
TEXTURE_CLASSES = ("woven", "knitted", "braided", "fibrous", "matted", "meshed",
                   "pleated", "gauzy", "interlaced")
FABRIC_CLASSES = {
    "plain-weave-fine": ("fabric", "woven"),
    "plain-weave-coarse": ("fabric", "woven"),
    "satin-slick": ("fabric", "gauzy"),
    "satin-dense": ("fabric", "gauzy"),
    "twill-diagonal": ("fabric", "interlaced"),
    "denim-heavy": ("fabric", "woven"),
    "corduroy-wide": ("fabric", "pleated"),
    "corduroy-narrow": ("fabric", "pleated"),
    "pile-velvet": ("fabric", "fibrous"),
    "pile-fleece": ("carpet", "fibrous"),
    "mesh-open": ("fabric", "meshed"),
    "felt-matte": ("fabric", "matted"),
    "canvas-rough": ("fabric", "braided"),
    "linen-slub": ("fabric", "woven"),
    "chiffon-sheer": ("fabric", "gauzy"),
    "terry-loop": ("carpet", "knitted"),
}


def recordings(repeats: int = 4, seconds: float = 1.5, seed: int = 0,
               velocity_free: bool = False) -> pd.DataFrame:
    """Descriptors over materials x velocities x forces x repeats, with the O1
    normalised columns and `in_band` flag.

    `velocity_free` synthesises every recording at the reference velocity while
    keeping its nominal label, which makes the velocity relations false.
    """
    rows, counter = [], 0
    for i, m in enumerate(FABRICS):
        for v in VELOCITIES_MM_S:
            for fo in FORCES_N:
                for r in range(repeats):
                    counter += 1
                    actual = D.REF_VELOCITY_MM_S if velocity_free else v
                    x = G.synthesize(m, actual, fo, seconds, seed=seed * 100000 + counter)
                    d = D.extract(x, G.FS_HZ, material_id=i, velocity_mm_s=v, force_n=fo)
                    rows.append({**d.as_dict(), "material_name": m.name, "repeat": r})
    return D.normalise_frame(pd.DataFrame(rows))


@dataclass(frozen=True)
class ViewConfig:
    """`n_views` stands for tiles cut from one scan."""
    coupling: float = 0.8
    view_noise: float = 0.3
    n_views: int = 8
    dim: int = 128
    classifier_accuracy: float = 0.8
    seed: int = 0


def _latent() -> np.ndarray:
    """What each fabric is, as standardised generator parameters."""
    z = np.array([[np.log(m.wavelength_mm), m.periodicity, np.log(m.roughness),
                   np.log(m.burst_per_mm)] for m in FABRICS])
    return (z - z.mean(0)) / z.std(0)


def simulated_embeddings(cfg: ViewConfig) -> np.ndarray:
    """(materials, views, dim). Each material is a point built partly from its
    haptic parameters (`coupling`) and partly from unrelated noise, plus view noise."""
    rng = np.random.default_rng(cfg.seed)
    z = _latent()
    linked = z @ rng.normal(size=(z.shape[1], cfg.dim))
    unrelated = rng.normal(size=(len(z), z.shape[1])) @ rng.normal(size=(z.shape[1], cfg.dim))
    linked /= linked.std()
    unrelated /= unrelated.std()
    base = cfg.coupling * linked + np.sqrt(1 - cfg.coupling ** 2) * unrelated
    noise = rng.normal(0, cfg.view_noise, (len(z), cfg.n_views, cfg.dim))
    return base[:, None, :] + noise


def _class_probs(true_index, k: int, cfg: ViewConfig, seed: int) -> np.ndarray:
    """(materials, views, classes). The true class wins with the set accuracy."""
    rng = np.random.default_rng(seed)
    n = len(true_index)
    probs = rng.dirichlet(np.ones(k) * 0.5, (n, cfg.n_views))
    for i, c in enumerate(true_index):
        for v in range(cfg.n_views):
            if rng.random() < cfg.classifier_accuracy:
                probs[i, v] = np.full(k, 0.3 / (k - 1) if k > 1 else 0.0)
                probs[i, v, c] = 0.7
    return probs


def simulated_class_probs(cfg: ViewConfig) -> tuple[np.ndarray, np.ndarray]:
    mats = [MATERIAL_CLASSES.index(FABRIC_CLASSES[m.name][0]) for m in FABRICS]
    texs = [TEXTURE_CLASSES.index(FABRIC_CLASSES[m.name][1]) for m in FABRICS]
    return (_class_probs(mats, len(MATERIAL_CLASSES), cfg, cfg.seed + 1),
            _class_probs(texs, len(TEXTURE_CLASSES), cfg, cfg.seed + 2))
