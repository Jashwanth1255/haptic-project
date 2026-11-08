import json
import re

import pytest

from vision import label_kit as K
from vision import transfer as P
from vision.textile import TEXTILE_CLASSES


def test_page_lists_only_images_and_all_classes(tmp_path):
    for n in ("b.jpg", "a.PNG", "notes.txt", "c.webp"):
        (tmp_path / n).write_bytes(b"x")
    imgs = K.images_in(tmp_path)
    assert imgs == ["a.PNG", "b.jpg", "c.webp"]
    page = K.build_page(imgs)
    got = json.loads(re.search(r"const IMAGES = (\[.*?\]),", page).group(1))
    assert got == imgs
    classes = json.loads(re.search(r"CLASSES = (\[.*?\]),", page).group(1))
    assert classes == list(TEXTILE_CLASSES)
    assert "__" not in page.replace("__proto__", "")      # every placeholder filled


def test_empty_folder_is_refused():
    with pytest.raises(ValueError):
        K.build_page([])


def test_exported_format_is_what_transfer_reads(tmp_path):
    # the page writes a header, then "image",label rows, skipping "none of the 12"
    csv = tmp_path / "labels_a.csv"
    csv.write_text('image,label\n"a.jpg",woven\n"b, x.jpg",knitted\n', encoding="utf-8")
    labels = P.read_labels(csv)
    assert labels.to_dict() == {"a.jpg": "woven", "b, x.jpg": "knitted"}
