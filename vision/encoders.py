"""DINOv2 and CLIP image embeddings, and the CLIP zero-shot attribute baseline.

    python -m vision.encoders --images <dir> --out <dir> [--tiles 3x3] [--zero-shot]

Each encoder's vectors are cached as an EmbeddingSet (.npz): the encoder name, one
id per image, and the vectors in that order. `concatenate` L2-normalises each
encoder's vectors separately and refuses sets whose image ids differ, so a
misaligned join cannot pass silently.

The zero-shot scores are a **baseline for the perception model, not the perception
model**. Each attribute is scored as the CLIP probability of a "high" prompt
against a "low" one. The prompts were written for this project and are not
validated; they are saved with every run.

Weights download from the Hugging Face hub on first use. On a CPU this is slow
for more than a few hundred images: ask before using a Kaggle GPU for it.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DINOV2_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-base-patch16"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

ATTRIBUTES = ("roughness", "bumpiness", "slipperiness", "softness", "warmth")
PROMPTS = {     # (low, high)
    "roughness": ("a photo of a smooth fabric", "a photo of a rough fabric"),
    "bumpiness": ("a photo of a flat fabric", "a photo of a bumpy fabric"),
    "slipperiness": ("a photo of a grippy fabric", "a photo of a slippery fabric"),
    "softness": ("a photo of a stiff fabric", "a photo of a soft fabric"),
    "warmth": ("a photo of a cool, thin fabric", "a photo of a warm, thick fabric"),
}


@dataclass(frozen=True)
class EmbeddingSet:
    encoder: str
    image_ids: tuple[str, ...]
    vectors: np.ndarray

    def __post_init__(self):
        v = np.asarray(self.vectors, dtype=np.float32)
        if v.ndim != 2 or v.shape[0] != len(self.image_ids):
            raise ValueError(f"{v.shape} vectors for {len(self.image_ids)} image ids")
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("duplicate image ids")
        if not np.isfinite(v).all():
            raise ValueError("non-finite values in the vectors")
        object.__setattr__(self, "vectors", v)
        object.__setattr__(self, "image_ids", tuple(self.image_ids))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        np.savez(path, encoder=self.encoder, image_ids=np.array(self.image_ids),
                 vectors=self.vectors)
        return path if path.suffix == ".npz" else path.with_suffix(path.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingSet":
        with np.load(path, allow_pickle=False) as f:
            return cls(str(f["encoder"]), tuple(str(i) for i in f["image_ids"]),
                       f["vectors"])

    def normalised(self) -> "EmbeddingSet":
        n = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        if (n == 0).any():
            raise ValueError(f"{self.encoder}: zero vector, cannot normalise")
        return EmbeddingSet(self.encoder, self.image_ids, self.vectors / n)

    def reorder(self, image_ids) -> "EmbeddingSet":
        """The same vectors in another order. Ids must match exactly as a set."""
        image_ids = tuple(image_ids)
        if set(image_ids) != set(self.image_ids) or len(image_ids) != len(self.image_ids):
            raise ValueError(f"{self.encoder}: {_difference(self.image_ids, image_ids)}")
        pos = {i: k for k, i in enumerate(self.image_ids)}
        return EmbeddingSet(self.encoder, image_ids,
                            self.vectors[[pos[i] for i in image_ids]])


def _difference(a, b) -> str:
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    return f"ids differ: {len(only_a)} only in one set {only_a[:3]}, " \
           f"{len(only_b)} only in the other {only_b[:3]}"


def concatenate(*sets: EmbeddingSet) -> EmbeddingSet:
    """Join encoders after L2-normalising each, so neither dominates by scale.
    The ids must be identical and in the same order: call `reorder` first if the
    order differs, which is an explicit step rather than a silent fix."""
    if len(sets) < 2:
        raise ValueError("need at least two sets")
    first = sets[0].image_ids
    for s in sets[1:]:
        if s.image_ids != first:
            if set(s.image_ids) == set(first) and len(s.image_ids) == len(first):
                raise ValueError(f"{s.encoder}: same images in a different order; "
                                 "reorder it first")
            raise ValueError(f"{s.encoder}: {_difference(first, s.image_ids)}")
    return EmbeddingSet("+".join(s.encoder for s in sets), first,
                        np.hstack([s.normalised().vectors for s in sets]))


# ---- encoders --------------------------------------------------------------

class Dinov2Encoder:
    """The CLS token after the final layer norm (`pooler_output`)."""

    def __init__(self, model=None, processor=None, name: str = DINOV2_MODEL,
                 device: str = "cpu"):
        if model is None:
            from transformers import AutoImageProcessor, AutoModel
            model = AutoModel.from_pretrained(name)
            processor = AutoImageProcessor.from_pretrained(name)
        self.model, self.processor = model.eval().to(device), processor
        self.name, self.device = f"dinov2:{name}", device

    def encode(self, images) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = self.processor(images=images, return_tensors="pt")["pixel_values"]
            return self.model(pixel_values=x.to(self.device)).pooler_output.cpu().numpy()


class ClipEncoder:
    def __init__(self, model=None, processor=None, name: str = CLIP_MODEL,
                 device: str = "cpu"):
        if model is None:
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained(name)
            processor = CLIPProcessor.from_pretrained(name)
        self.model, self.processor = model.eval().to(device), processor
        self.name, self.device = f"clip:{name}", device

    def encode(self, images) -> np.ndarray:
        import torch
        with torch.no_grad():
            x = self.processor(images=images, return_tensors="pt")["pixel_values"]
            return self.model.get_image_features(pixel_values=x.to(self.device)).cpu().numpy()

    def encode_text(self, texts) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = self.processor(text=list(texts), return_tensors="pt", padding=True)
            t = {k: v.to(self.device) for k, v in t.items()}
            return self.model.get_text_features(**t).cpu().numpy()

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.detach().exp())


def zero_shot(clip: ClipEncoder, images: EmbeddingSet, prompts=PROMPTS) -> pd.DataFrame:
    """P(high prompt) per image and attribute, from CLIP image embeddings. A
    temperature-scaled softmax over the two prompts, as in CLIP zero-shot use."""
    if not images.encoder.startswith("clip:"):
        raise ValueError(f"zero-shot needs CLIP image embeddings, not {images.encoder}")
    img = images.normalised().vectors
    out = {}
    for attr, pair in prompts.items():
        txt = clip.encode_text(pair)
        txt = txt / np.linalg.norm(txt, axis=1, keepdims=True)
        logits = clip.logit_scale * img @ txt.T
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        out[attr] = p[:, 1] / p.sum(axis=1)
    return pd.DataFrame(out, index=pd.Index(images.image_ids, name="image_id"))


# ---- images ------------------------------------------------------------------

def image_items(folder: str | Path, grid: tuple[int, int] | None = None):
    """(image id, path, tile index or None). Ids are paths relative to `folder`,
    with `#r<row>c<col>` added for tiles."""
    folder = Path(folder)
    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no images under {folder}")
    items = []
    for p in paths:
        rel = p.relative_to(folder).as_posix()
        if grid is None:
            items.append((rel, p, None))
        else:
            rows, cols = grid
            items += [(f"{rel}#r{k // cols}c{k % cols}", p, k) for k in range(rows * cols)]
    return items


def load_image(path: Path, tile, grid):
    from PIL import Image

    from .embed import tiles
    with Image.open(path) as im:
        im = im.convert("RGB")
    return im if tile is None else tiles(im, *grid)[tile]


def extract(encoder, items, grid=None, batch_size: int = 32) -> EmbeddingSet:
    vecs = []
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        vecs.append(encoder.encode([load_image(p, t, grid) for _, p, t in batch]))
        print(f"  {encoder.name}: {start + len(batch)}/{len(items)}", file=sys.stderr)
    return EmbeddingSet(encoder.name, tuple(i for i, _, _ in items), np.vstack(vecs))


def _grid(text: str | None):
    if not text:
        return None
    rows, cols = (int(x) for x in text.lower().split("x"))
    return rows, cols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="vision.encoders")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True,
                    help="run name suffix, e.g. cluster_scans or product_photos")
    ap.add_argument("--tiles", help="cut each image into a ROWSxCOLS grid, e.g. 3x3")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--dinov2", default=DINOV2_MODEL)
    ap.add_argument("--clip", default=CLIP_MODEL)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args(argv)

    import torch

    import runlog

    device = "cuda" if torch.cuda.is_available() else "cpu"
    grid = _grid(a.tiles)
    items = image_items(a.images, grid)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} images on {device}", file=sys.stderr)

    dino = extract(Dinov2Encoder(name=a.dinov2, device=device), items, grid, a.batch_size)
    clip_enc = ClipEncoder(name=a.clip, device=device)
    clip = extract(clip_enc, items, grid, a.batch_size)
    joined = concatenate(dino, clip)
    files = {k: s.save(out / f"{k}.npz") for k, s in
             (("dinov2", dino), ("clip", clip), ("dinov2_clip", joined))}
    artifacts = {}
    if a.zero_shot:
        scores = zero_shot(clip_enc, clip)
        scores.to_csv(out / "zero_shot.csv")
        artifacts["zero_shot"] = scores
    saved = runlog.save_run(
        f"embeddings_{a.name}",
        {**vars(a), "device": device, "prompts": PROMPTS if a.zero_shot else None},
        {"n_images": len(items), "dims": {"dinov2": dino.vectors.shape[1],
                                          "clip": clip.vectors.shape[1]},
         "files": {k: str(v) for k, v in files.items()}},
        artifacts, synthetic=False)
    print(f"wrote {out}\nsaved run {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
