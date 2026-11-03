"""Embedding sets, their alignment, and the encoders on tiny random models.

No weights are downloaded: the transformers models here are built from small
configs with random weights, so these tests check wiring, not embeddings.
"""
import numpy as np
import pytest
import torch
from PIL import Image

from vision import encoders as E


def _set(enc, ids, d=4, seed=0):
    return E.EmbeddingSet(enc, tuple(ids), np.random.default_rng(seed).normal(size=(len(ids), d)))


def test_save_and_load_round_trip(tmp_path):
    s = _set("dinov2:x", ["a.jpg", "b.jpg#r0c1"])
    back = E.EmbeddingSet.load(s.save(tmp_path / "s.npz"))
    assert back.image_ids == s.image_ids and back.encoder == s.encoder
    np.testing.assert_array_equal(back.vectors, s.vectors)


def test_concatenate_normalises_each_encoder_separately():
    a = _set("dinov2:x", "abc", d=4)
    b = E.EmbeddingSet("clip:y", tuple("abc"), _set("", "abc", d=3).vectors * 1000)
    j = E.concatenate(a, b)
    assert j.vectors.shape == (3, 7) and j.encoder == "dinov2:x+clip:y"
    np.testing.assert_allclose(np.linalg.norm(j.vectors[:, :4], axis=1), 1, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(j.vectors[:, 4:], axis=1), 1, rtol=1e-6)


def test_misaligned_sets_are_refused():
    with pytest.raises(ValueError, match="different order"):
        E.concatenate(_set("a", "abc"), _set("b", "cab"))
    with pytest.raises(ValueError, match="ids differ"):
        E.concatenate(_set("a", "abc"), _set("b", "abd"))


def test_reorder_is_the_explicit_fix_and_keeps_rows_with_their_ids():
    a, b = _set("a", "abc"), _set("b", "cab", seed=1)
    fixed = b.reorder(a.image_ids)
    np.testing.assert_array_equal(fixed.vectors[0], b.vectors[1])
    assert E.concatenate(a, fixed).image_ids == tuple("abc")
    with pytest.raises(ValueError):
        b.reorder("abd")


def test_bad_sets_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        _set("a", "aa")
    with pytest.raises(ValueError, match="vectors for"):
        E.EmbeddingSet("a", ("x",), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        E.EmbeddingSet("a", ("x",), np.array([[np.nan]]))
    with pytest.raises(ValueError, match="zero vector"):
        E.EmbeddingSet("a", ("x",), np.zeros((1, 3))).normalised()


class _Processor:
    """Stands in for a Hugging Face processor: 28x28 pixels, a toy tokeniser."""

    def __call__(self, images=None, text=None, return_tensors="pt", padding=False):
        if images is not None:
            x = [torch.tensor(np.asarray(im.resize((28, 28)), dtype=np.float32) / 255)
                 .permute(2, 0, 1) for im in images]
            return {"pixel_values": torch.stack(x)}
        ids = [[1 + (ord(c) % 90) for c in t[:12]] for t in text]
        n = max(map(len, ids))
        return {"input_ids": torch.tensor([i + [0] * (n - len(i)) for i in ids]),
                "attention_mask": torch.tensor([[1] * len(i) + [0] * (n - len(i))
                                                for i in ids])}


@pytest.fixture(scope="module")
def tiny():
    from transformers import CLIPConfig, CLIPModel, Dinov2Config, Dinov2Model
    torch.manual_seed(0)
    dino = Dinov2Model(Dinov2Config(hidden_size=16, num_hidden_layers=1,
                                    num_attention_heads=2, intermediate_size=32,
                                    image_size=28, patch_size=14))
    clip = CLIPModel(CLIPConfig(
        text_config=dict(vocab_size=100, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         max_position_embeddings=16),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=28, patch_size=14),
        projection_dim=8))
    return (E.Dinov2Encoder(dino, _Processor(), name="tiny"),
            E.ClipEncoder(clip, _Processor(), name="tiny"))


@pytest.fixture
def photos(tmp_path):
    for i, colour in enumerate([(200, 10, 10), (10, 200, 10), (10, 10, 200)]):
        sub = tmp_path / ("a" if i < 2 else "b")
        sub.mkdir(exist_ok=True)
        Image.new("RGB", (60, 90), colour).save(sub / f"{i}.png")
    return tmp_path


def test_items_are_relative_ids_and_tiles_are_named(photos):
    plain = E.image_items(photos)
    assert [i for i, _, _ in plain] == ["a/0.png", "a/1.png", "b/2.png"]
    tiled = E.image_items(photos, (2, 3))
    assert len(tiled) == 18 and tiled[5][0] == "a/0.png#r1c2"


def test_encoders_produce_aligned_sets(tiny, photos):
    dino, clip = tiny
    items = E.image_items(photos, (2, 1))
    d = E.extract(dino, items, (2, 1), batch_size=4)
    c = E.extract(clip, items, (2, 1), batch_size=4)
    assert d.vectors.shape == (6, 16) and c.vectors.shape == (6, 8)
    assert d.encoder == "dinov2:tiny" and c.encoder == "clip:tiny"
    assert E.concatenate(d, c).vectors.shape == (6, 24)


def test_zero_shot_gives_a_probability_per_attribute(tiny, photos):
    _, clip = tiny
    c = E.extract(clip, E.image_items(photos), batch_size=8)
    z = E.zero_shot(clip, c)
    assert list(z.columns) == list(E.ATTRIBUTES)
    assert ((z > 0) & (z < 1)).all().all()
    assert list(z.index) == list(c.image_ids)


def test_zero_shot_refuses_non_clip_embeddings(tiny):
    with pytest.raises(ValueError, match="CLIP"):
        E.zero_shot(tiny[1], _set("dinov2:x", "ab"))


def test_every_attribute_has_a_low_and_high_prompt():
    assert set(E.PROMPTS) == set(E.ATTRIBUTES)
    assert all(len(p) == 2 and p[0] != p[1] for p in E.PROMPTS.values())
