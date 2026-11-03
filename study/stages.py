"""Stages 1 and 2 of the representability analysis, each saved as a run.

    python -m study.stages stage1 --embeddings E.npz --ratings R.csv --reliability RUN
    python -m study.stages stage2 --descriptors D.csv --ratings R.csv --image-materials M.csv
                                  [--embeddings E.npz] [--confusion C.csv]

Stage 1, perception from images: ridge from the joined DINOv2 + CLIP embedding to
each attribute's mean rating, leave one image out. Every Spearman is reported
beside its sqrt(ICC(2,k)) ceiling, and beside the CLIP zero-shot baseline.

Stage 2, perception and haptics, at the material level:

1. each attribute's mean rating from the four haptic descriptors, leave one
   material out, with a permutation test (`stage2_representability`)
2. the ablation: embedding, perception, and both, predicting the haptic
   descriptors (`stage2_ablation`)
3. the Mantel test between the visual confusion matrix and haptic descriptor
   distance (`stage2_mantel`)

Velocity-normalised descriptors are the primary analysis and raw descriptors are
reported alongside. Do not run Stage 2 until Objective 1's gate has run on real
Cluster descriptors.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import runlog
from haptic import crossmodal as X
from haptic import descriptor as D
from vision import regression as G
from vision.encoders import ATTRIBUTES, EmbeddingSet

FEATURE_SETS = {"normalised": D.NORMALISED_CORE, "raw": D.Descriptor.CORE}


def _attributes(ratings: pd.DataFrame) -> list[str]:
    attrs = [a for a in ATTRIBUTES if a in ratings.columns]
    if not attrs:
        raise ValueError(f"no attribute columns among {list(ratings.columns)}")
    return attrs


def stage1(embeddings: EmbeddingSet, ratings: pd.DataFrame, icc: pd.Series,
           zero_shot: pd.DataFrame | None = None, n_permutations: int = G.N_PERMUTATIONS,
           seed: int = 0) -> pd.DataFrame:
    """Leave one image out. Ratings are indexed by image id, which must match the
    embedding ids exactly; unmatched rated images are an error, not a silent drop."""
    missing = sorted(set(ratings.index) - set(embeddings.image_ids))
    if missing:
        raise ValueError(f"{len(missing)} rated images have no embedding, e.g. {missing[:3]}")
    attrs = _attributes(ratings)
    pos = {i: k for k, i in enumerate(embeddings.image_ids)}
    X_ = embeddings.vectors[[pos[i] for i in ratings.index]]
    table = G.evaluate([X_], ratings[attrs].to_numpy(), attrs,
                       n_permutations=n_permutations, seed=seed)
    table = G.against_ceiling(table, icc)
    if zero_shot is not None:
        table = table.merge(G.zero_shot_baseline(zero_shot, ratings[attrs]),
                            on="target", how="left")
    return table


def material_ratings(ratings: pd.DataFrame, image_materials: pd.Series) -> pd.DataFrame:
    """Mean rating per material over its images."""
    unknown = sorted(set(ratings.index) - set(image_materials.index))
    if unknown:
        raise ValueError(f"{len(unknown)} rated images have no material, e.g. {unknown[:3]}")
    return ratings.groupby(image_materials.loc[ratings.index].to_numpy()).mean()


def material_descriptors(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Recording-level descriptors to one mean row per material, in band only."""
    d = descriptors[descriptors["in_band"]] if "in_band" in descriptors else descriptors
    cols = list(D.NORMALISED_CORE) + list(D.Descriptor.CORE)
    return d.groupby("material_name")[list(dict.fromkeys(cols))].mean()


def _common(*frames) -> list:
    idx = set(frames[0].index)
    for f in frames[1:]:
        idx &= set(f.index)
    if len(idx) < 3:
        raise ValueError(f"only {len(idx)} materials in common")
    return sorted(idx)


def stage2_representability(desc: pd.DataFrame, ratings: pd.DataFrame,
                            n_permutations: int = G.N_PERMUTATIONS, seed: int = 0):
    mats = _common(desc, ratings)
    attrs = _attributes(ratings)
    return pd.concat([
        G.evaluate([desc.loc[mats, list(cols)].to_numpy()], ratings.loc[mats, attrs].to_numpy(),
                   attrs, n_permutations=n_permutations, seed=seed).assign(descriptors=name)
        for name, cols in FEATURE_SETS.items()], ignore_index=True)


def stage2_ablation(desc: pd.DataFrame, ratings: pd.DataFrame, embeddings: pd.DataFrame,
                    n_permutations: int = G.N_PERMUTATIONS, seed: int = 0):
    mats = _common(desc, ratings, embeddings)
    attrs = _attributes(ratings)
    E, P = embeddings.loc[mats].to_numpy(), ratings.loc[mats, attrs].to_numpy()
    targets = list(D.NORMALISED_CORE)
    return G.ablation({"embedding": [E], "perception": [P], "embedding+perception": [E, P]},
                      desc.loc[mats, targets].to_numpy(), targets,
                      n_permutations=n_permutations, seed=seed)


def material_embeddings(embeddings: EmbeddingSet, image_materials: pd.Series) -> pd.DataFrame:
    """Mean embedding per material. Tile ids (`image#r0c1`) map through their image."""
    base = [i.split("#")[0] for i in embeddings.image_ids]
    unknown = sorted(set(base) - set(image_materials.index))
    if unknown:
        raise ValueError(f"{len(unknown)} embedded images have no material, e.g. {unknown[:3]}")
    return pd.DataFrame(embeddings.vectors).groupby(image_materials.loc[base].to_numpy()).mean()


def stage2_mantel(desc: pd.DataFrame, confusion: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rows = []
    for name, cols in FEATURE_SETS.items():
        z = desc[list(cols)]
        z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
        dist = np.sqrt(((z.to_numpy()[:, None] - z.to_numpy()[None]) ** 2).sum(-1))
        r = X.mantel_test(confusion.to_numpy(), dist,
                          confusion_labels=list(confusion.index),
                          distance_labels=list(desc.index), seed=seed)
        rows.append({"descriptors": name, "rho": r.rho, "p": r.p_value, "n_pairs": r.n,
                     "n_materials": r.n_materials, "alternative": r.alternative})
    return pd.DataFrame(rows)


def _series(path: str, name: str) -> pd.Series:
    df = pd.read_csv(path, dtype=str)
    if df.shape[1] != 2:
        raise ValueError(f"{path}: expected two columns, image_id and {name}")
    return df.set_index(df.columns[0])[df.columns[1]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="study.stages")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("stage1")
    p1.add_argument("--embeddings", required=True, help="joined DINOv2 + CLIP .npz")
    p1.add_argument("--ratings", required=True, help="image x attribute mean ratings CSV")
    p1.add_argument("--reliability", required=True, help="study_reliability run name or path")
    p1.add_argument("--zero-shot", help="zero_shot.csv from vision.encoders")
    p2 = sub.add_parser("stage2")
    p2.add_argument("--descriptors", required=True, help="descriptors CSV from the O1 gate")
    p2.add_argument("--ratings", required=True)
    p2.add_argument("--image-materials", required=True, help="CSV: image_id, material")
    p2.add_argument("--embeddings")
    p2.add_argument("--confusion", help="material x material visual confusion CSV")
    for p in (p1, p2):
        p.add_argument("--permutations", type=int, default=G.N_PERMUTATIONS)
        p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ratings = pd.read_csv(a.ratings, index_col=0)
    config = {**vars(a)}

    if a.cmd == "stage1":
        icc = pd.Series(runlog.load_run(a.reliability).metrics["icc2k"])
        zs = pd.read_csv(a.zero_shot, index_col=0) if a.zero_shot else None
        table = stage1(EmbeddingSet.load(a.embeddings), ratings, icc, zs,
                       a.permutations, a.seed)
        print(table.round(3).to_string(index=False))
        saved = runlog.save_run("stage1_perception", config,
                                {"spearman": dict(zip(table["target"], table["spearman"])),
                                 "fraction_of_ceiling": dict(zip(table["target"],
                                                                 table["fraction_of_ceiling"]))},
                                {"scores": table}, synthetic=False)
        print(f"saved run {saved}", file=sys.stderr)
        return 0

    image_materials = _series(a.image_materials, "material")
    mratings = material_ratings(ratings, image_materials)
    desc = material_descriptors(pd.read_csv(a.descriptors))
    def save(name, metrics, table):
        print(table.round(3).to_string(index=False))
        path = runlog.save_run(name, config, metrics, {"scores": table}, synthetic=False)
        print(f"saved run {path}\n", file=sys.stderr)

    rep = stage2_representability(desc, mratings, a.permutations, a.seed)
    norm = rep[rep["descriptors"] == "normalised"]
    save("stage2_representability",
         {"r2": dict(zip(norm["target"], norm["r2"])),
          "p_permutation": dict(zip(norm["target"], norm["p_permutation"]))}, rep)
    if a.embeddings:
        emb = material_embeddings(EmbeddingSet.load(a.embeddings), image_materials)
        ab = stage2_ablation(desc, mratings, emb, a.permutations, a.seed)
        save("stage2_ablation", {"mean_r2": ab.groupby("features")["r2"].mean().to_dict()}, ab)
    if a.confusion:
        conf = pd.read_csv(a.confusion, index_col=0)
        man = stage2_mantel(desc.loc[_common(desc, conf)], conf, a.seed)
        save("stage2_mantel", man.set_index("descriptors").to_dict(orient="index"), man)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
