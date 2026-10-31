"""MINC-2500 and DTD: the published splits, and the images behind them.

The published split files are used rather than a fresh random split, so that
numbers are comparable with the literature. Both datasets ship several folds;
the fold in use is always recorded next to the result.

    dataset      classes   folds   train / val / test per class
    MINC-2500    23        5       2125 / 125 / 250
    DTD          47        10      40 / 40 / 40

These counts are checked against what is on disk, not assumed. A dataset copy
that does not match fails here, rather than as a quietly wrong accuracy later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    n_classes: int
    n_folds: int
    val_split: str
    per_class: dict[str, int]
    # what a line in a label file is relative to, under the dataset root
    prefix: str


MINC2500 = DatasetSpec(
    name="minc2500", n_classes=23, n_folds=5, val_split="validate",
    per_class={"train": 2125, "validate": 125, "test": 250}, prefix="",
)
DTD = DatasetSpec(
    name="dtd", n_classes=47, n_folds=10, val_split="val",
    per_class={"train": 40, "val": 40, "test": 40}, prefix="images",
)
SPECS = {s.name: s for s in (MINC2500, DTD)}


def find_root(base: str | Path, spec: DatasetSpec) -> Path:
    """Find the directory holding `images/` and `labels/`, however deeply a
    Kaggle dataset has nested it.

    The class count picks between candidates: a kernel usually has both datasets
    attached, so the first match under /kaggle/input may be the other one.
    """
    base = Path(base)
    if not base.exists():
        raise FileNotFoundError(f"dataset path not found: {base}")
    for cand in [base, *sorted(p.parent for p in base.rglob("labels"))]:
        if ((cand / "images").is_dir() and (cand / "labels").is_dir()
                and len(class_names(cand)) == spec.n_classes):
            return cand
    raise RuntimeError(
        f"no directory under {base} holds both images/ and labels/ for "
        f"{spec.name}; this copy of the dataset does not include the "
        f"published splits")


def class_names(root: Path) -> tuple[str, ...]:
    """Sorted class directories. Sorted, so an index means the same class on
    every machine and every run."""
    return tuple(sorted(p.name for p in (root / "images").iterdir() if p.is_dir()))


def read_split(root: Path, spec: DatasetSpec, split: str,
               fold: int = 1) -> list[tuple[Path, int]]:
    """Read one published split as (path, class index), and verify it."""
    if split not in spec.per_class:
        raise ValueError(f"{spec.name} has no split {split!r}; "
                         f"expected one of {sorted(spec.per_class)}")
    if not 1 <= fold <= spec.n_folds:
        raise ValueError(f"{spec.name} has folds 1..{spec.n_folds}, not {fold}")

    classes = class_names(root)
    if len(classes) != spec.n_classes:
        raise RuntimeError(f"{spec.name}: expected {spec.n_classes} classes, "
                           f"found {len(classes)} under {root / 'images'}")
    index = {c: i for i, c in enumerate(classes)}

    listing = root / "labels" / f"{split}{fold}.txt"
    if not listing.is_file():
        raise FileNotFoundError(f"split file not found: {listing}")

    items, counts = [], [0] * len(classes)
    for line in listing.read_text().split():
        path = root / spec.prefix / line
        # the class is the image's parent directory, in both datasets
        cls = Path(line).parent.name
        if cls not in index:
            raise RuntimeError(f"{listing.name}: unknown class in {line!r}")
        if not path.is_file():
            raise FileNotFoundError(f"{listing.name} lists a missing image: {path}")
        items.append((path, index[cls]))
        counts[index[cls]] += 1

    want = spec.per_class[split]
    wrong = {classes[i]: n for i, n in enumerate(counts) if n != want}
    if wrong:
        raise RuntimeError(f"{spec.name} {split}{fold}: expected {want} images "
                           f"per class, but {wrong}")
    return items


def load_splits(root: Path, spec: DatasetSpec, fold: int = 1) -> dict[str, list]:
    """All three splits for a fold, under the names train / val / test.

    Refuses overlapping splits: a test image seen in training is the one defect
    that makes every later number look better than it is.
    """
    out = {"train": read_split(root, spec, "train", fold),
           "val": read_split(root, spec, spec.val_split, fold),
           "test": read_split(root, spec, "test", fold)}
    seen: dict[Path, str] = {}
    for name, items in out.items():
        for path, _ in items:
            if path in seen:
                raise RuntimeError(f"{path.name} is in both {seen[path]} and "
                                   f"{name}: splits overlap")
            seen[path] = name
    return out


def transforms_for(train: bool, size: int = 224):
    """Standard ImageNet preprocessing, as the pretrained weights expect.

    Colour jitter is left out deliberately. Colour is part of what the haptic
    class is being inferred from, and the scan-to-photo gap should be measured
    on a model that has not been trained to ignore it by accident.
    """
    from torchvision import transforms as T
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        return T.Compose([T.RandomResizedCrop(size), T.RandomHorizontalFlip(), *norm])
    return T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), *norm])


class ImageList:
    """Map-style dataset over (path, label) pairs."""

    def __init__(self, items, transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        from PIL import Image
        path, label = self.items[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), label
