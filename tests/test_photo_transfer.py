"""Caption parsing, selection and the bootstrap for analysis G (invented data)."""
import numpy as np
import pandas as pd

import make_photo_transfer as G


def test_caption_fabrics_both_phrasings():
    c = ("The T-shirt has short sleeves and its fabric is cotton. "
         "She wears long trousers with denim fabric and solid color patterns.")
    assert G.caption_fabrics(c) == {"cotton", "denim"}
    assert G.caption_label(c) is None
    assert G.caption_label("The shirt is with chiffon fabric. The skirt is with chiffon fabric.") == "chiffon"
    assert G.caption_label("A plain shirt with lapel neckline.") is None


def test_select_only_single_fabric_and_caps():
    rows = [{"image_id": f"i{k}", "path": f"i{k}.jpg", "product_type": "x", "gender": "W",
             "caption": f"with {fab} fabric"} for k, fab in
            enumerate(["cotton"] * 150 + ["chiffon"] * 5 + ["denim"] * 20)]
    rows.append({"image_id": "mix", "path": "mix.jpg", "product_type": "x", "gender": "W",
                 "caption": "with cotton fabric, and with leather fabric"})
    sel = G.select(pd.DataFrame(rows))
    assert sel.fabric.value_counts().to_dict() == {"cotton": 100, "chiffon": 5}
    assert "mix" not in set(sel.image_id)
    again = G.select(pd.DataFrame(rows))
    assert list(again.image_id) == list(sel.image_id)


def test_balanced_accuracy_weights_classes_equally():
    y = np.array(["a"] * 90 + ["b"] * 10)
    pred = np.array(["a"] * 100)
    assert G.balanced_accuracy(y, pred) == 0.5


def test_bootstrap_centres_on_estimate():
    y = np.array(["a"] * 50 + ["b"] * 50)
    hit = np.r_[np.ones(40), np.zeros(10), np.ones(20), np.zeros(30)]
    b = G.bootstrap(y, hit, n=2000)
    assert abs(b.mean() - 0.6) < 0.02
    assert b.std() > 0
