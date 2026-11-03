"""Stages 1 and 2, and the profile that reads them back from saved runs.

Invented materials and ratings: these check the wiring and that a planted signal
is found and a planted non-signal is not. They are not results."""
import numpy as np
import pandas as pd
import pytest

import runlog
from haptic import descriptor as D
from study import representability as P
from study import stages as S
from vision.encoders import ATTRIBUTES, EmbeddingSet

N_MATERIALS, IMAGES_PER = 12, 3


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    mats = [f"m{i}" for i in range(N_MATERIALS)]
    rough = rng.uniform(0.5, 3, N_MATERIALS)
    rows = []
    for i, m in enumerate(mats):
        for r in range(4):
            row = {"material_name": m, "in_band": r != 3}
            for c in set(D.NORMALISED_CORE) | set(D.Descriptor.CORE):
                row[c] = rng.normal(1, 0.1)
            row["roughness_index"] = rough[i] * (1 + 0.02 * rng.normal())
            row["intensity"] = rough[i] * 2
            rows.append(row)
    desc = pd.DataFrame(rows)
    images = [f"{m}_{k}.jpg" for m in mats for k in range(IMAGES_PER)]
    image_materials = pd.Series([i.split("_")[0] for i in images], index=images)
    m_of = image_materials.map(dict(zip(mats, range(N_MATERIALS)))).to_numpy()
    ratings = pd.DataFrame({a: rng.uniform(1, 7, len(images)) for a in ATTRIBUTES},
                           index=pd.Index(images, name="image_id"))
    ratings["roughness"] = 1 + 2 * rough[m_of] + rng.normal(0, 0.2, len(images))
    emb = rng.normal(size=(len(images), 20))
    emb[:, 0] = ratings["roughness"] + rng.normal(0, 0.2, len(images))
    return desc, ratings, image_materials, EmbeddingSet("dinov2:x+clip:y", tuple(images), emb)


def test_stage1_finds_the_planted_attribute_and_carries_the_ceiling(world):
    _, ratings, _, emb = world
    icc = pd.Series({a: 0.81 for a in ATTRIBUTES})
    zs = pd.DataFrame({"roughness": ratings["roughness"].rank()}, index=ratings.index)
    t = S.stage1(emb, ratings, icc, zs, n_permutations=19).set_index("target")
    assert t.loc["roughness", "spearman"] > 0.8
    assert t.loc["roughness", "ceiling"] == pytest.approx(0.9)
    assert t.loc["roughness", "spearman_zero_shot"] == pytest.approx(1.0)
    assert t.loc["warmth", "r2"] < 0.1


def test_stage1_refuses_ratings_without_embeddings(world):
    _, ratings, _, emb = world
    extra = pd.concat([ratings, ratings.iloc[:1].rename(index={ratings.index[0]: "new.jpg"})])
    with pytest.raises(ValueError, match="no embedding"):
        S.stage1(emb, extra, pd.Series(dtype=float), n_permutations=0)


def test_stage2_representability_reports_both_descriptor_forms(world):
    desc, ratings, im, _ = world
    t = S.stage2_representability(S.material_descriptors(desc),
                                  S.material_ratings(ratings, im), n_permutations=19)
    assert set(t["descriptors"]) == {"normalised", "raw"}
    norm = t[t["descriptors"] == "normalised"].set_index("target")
    assert norm.loc["roughness", "r2"] > 0.5
    assert norm.loc["roughness", "p_permutation"] == pytest.approx(0.05)
    assert norm.loc["warmth", "r2"] < 0.2
    assert (t["n"] == N_MATERIALS).all()


def test_out_of_band_recordings_do_not_reach_the_material_means(world):
    desc = world[0].copy()
    desc.loc[~desc["in_band"], "roughness_index"] = 1e6
    assert S.material_descriptors(desc)["roughness_index"].max() < 10


def test_tiles_map_to_their_material(world):
    _, _, im, emb = world
    tiled = EmbeddingSet("x", tuple(f"{i}#r0c{k}" for i in emb.image_ids for k in range(2)),
                         np.repeat(emb.vectors, 2, axis=0))
    m = S.material_embeddings(tiled, im)
    assert m.shape == (N_MATERIALS, 20)


def test_ablation_and_mantel_run_on_material_tables(world):
    desc, ratings, im, emb = world
    d, r = S.material_descriptors(desc), S.material_ratings(ratings, im)
    ab = S.stage2_ablation(d, r, S.material_embeddings(emb, im), n_permutations=0)
    assert set(ab["features"]) == {"embedding", "perception", "embedding+perception"}
    assert set(ab["target"]) == set(D.NORMALISED_CORE)
    counts = np.random.default_rng(1).integers(0, 5, (N_MATERIALS, N_MATERIALS))
    conf = pd.DataFrame(counts + np.eye(N_MATERIALS, dtype=int) * 20,
                        index=d.index, columns=d.index)
    man = S.stage2_mantel(d, conf)
    assert set(man["descriptors"]) == {"normalised", "raw"}
    assert (man["n_materials"] == N_MATERIALS).all()


# ---- the profile -------------------------------------------------------------

def _runs(stage2_r2, stage2_p, synthetic=True):
    t1 = pd.DataFrame({"target": ATTRIBUTES, "fraction_of_ceiling": [0.8, 0.7, 0.5, 0.4, 0.1]})
    t2 = pd.DataFrame({"target": ATTRIBUTES * 2, "r2": stage2_r2 + [0] * 5,
                       "p_permutation": stage2_p + [1] * 5,
                       "descriptors": ["normalised"] * 5 + ["raw"] * 5})
    t3 = pd.DataFrame({"scheme": ["A"] * 5 + ["B"] * 5, "attribute": ATTRIBUTES * 2,
                       "spearman": [0.6, 0.5, 0.2, 0.1, 0.0] * 2})
    return {"stage1": runlog.save_run("stage1_perception", {}, {}, {"scores": t1}, synthetic=synthetic),
            "stage2": runlog.save_run("stage2_representability", {}, {}, {"scores": t2}, synthetic=synthetic),
            "stage3": runlog.save_run("study_discrimination_main", {}, {},
                                      {"attribute_tracking": t3}, synthetic=synthetic)}


def test_profile_reads_every_stage_from_its_run():
    paths = _runs([0.6, 0.5, 0.3, 0.2, -0.1], [0.01, 0.01, 0.04, 0.2, 0.6])
    t = P.profile(*(runlog.load_run(paths[k]) for k in P.STAGES))
    assert t.loc["roughness", "stage1"] == 0.8
    assert t.loc["slipperiness", "stage2"] == 0.3
    assert t.loc["roughness", "stage3"] == pytest.approx(0.6)


def test_ordering_holds_when_every_part_holds():
    r2 = pd.Series([0.6, 0.5, 0.3, 0.2, -0.1], index=ATTRIBUTES)
    p = pd.Series([0.01, 0.01, 0.04, 0.2, 0.6], index=ATTRIBUTES)
    v = P.ordering_verdict(r2, p)
    assert v["ordering_held"] is True and v["control_held"] is True


def test_roughness_and_bumpiness_may_come_in_either_order():
    r2 = pd.Series([0.5, 0.6, 0.3, 0.2, -0.1], index=ATTRIBUTES)
    assert P.ordering_verdict(r2, pd.Series(1.0, index=ATTRIBUTES))["ordering_held"] is True


def test_a_failed_part_and_a_failed_control_are_reported():
    r2 = pd.Series([0.6, 0.5, 0.1, 0.2, 0.4], index=ATTRIBUTES)
    p = pd.Series([0.01, 0.01, 0.3, 0.2, 0.01], index=ATTRIBUTES)
    v = P.ordering_verdict(r2, p)
    assert v["parts"]["slipperiness_above_softness"] is False
    assert v["ordering_held"] is False and v["control_held"] is False


def test_a_dropped_attribute_makes_its_comparisons_untestable():
    r2 = pd.Series([0.6, 0.5, 0.3, 0.2, -0.1], index=ATTRIBUTES)
    v = P.ordering_verdict(r2, pd.Series(1.0, index=ATTRIBUTES), dropped=["warmth"])
    assert v["parts"]["softness_above_warmth"] is None
    assert v["control_held"] is None and v["n_comparisons_tested"] == 2


def test_cli_writes_the_figure_and_inherits_the_synthetic_flag(tmp_path):
    _runs([0.6, 0.5, 0.3, 0.2, -0.1], [0.01, 0.01, 0.04, 0.2, 0.6])
    runlog.save_run("study_reliability_main", {}, {"unreliable": []}, synthetic=True)
    out = tmp_path / "rep.png"
    assert P.main(["--out", str(out)]) == 0
    assert out.stat().st_size > 10_000
    run = runlog.load_run("representability_profile")
    assert run.info["synthetic"] is True
    assert run.metrics["ordering"]["ordering_held"] is True
    assert set(run.config["source_runs"]) == set(P.STAGES)
