"""Labelling kit for the product-photo transfer set (plan item A5).

    python -m vision.label_kit --photos <folder of product photographs>

Writes `label.html` into the photo folder. Each labeller opens it in a browser on
their own machine, works alone, and clicks "Download CSV". The file is
`labels_<name>.csv` with columns image,label, exactly what
`python -m vision.cli transfer --labels-a ... --labels-b ...` reads. Photos
skipped as "none of the 12" are left out of the CSV, so they are never scored.

The page shows no definitions of the classes. It links to the DTD browse pages,
so both labellers judge against the same published examples and not against
wording written for this project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .textile import TEXTILE_CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DTD_BROWSE = "https://www.robots.ox.ac.uk/~vgg/data/dtd/"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Textile labelling</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--line:#ddd;--acc:#2a5db0;--sel:#e8f0fc}
@media (prefers-color-scheme: dark){:root{--bg:#161616;--fg:#eee;--muted:#aaa;--line:#333;--acc:#8fb4f0;--sel:#1f2b3d}}
body{background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif;margin:0;padding:16px;max-width:900px;margin-inline:auto}
img{max-width:100%;max-height:60vh;display:block;margin:8px auto;border:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:6px;margin:8px 0}
button{font:inherit;padding:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;cursor:pointer}
button.on{background:var(--sel);border-color:var(--acc)}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.muted{color:var(--muted)} a{color:var(--acc)}
</style></head><body>
<h1>Textile labelling</h1>
<p class="muted">Label every photo on your own; do not compare answers with the other
labeller. Choose the one class that best describes the garment's fabric surface. If
none fits, choose "none of the 12". Reference examples for each class are on the
<a href="__DTD__" target="_blank" rel="noopener">DTD page</a>.</p>
<div class="row"><label>Your name <input id="who" autocomplete="off"></label>
<span id="prog" class="muted"></span></div>
<div id="card"></div>
<div class="grid" id="opts"></div>
<div class="row"><button id="prev">Previous</button><button id="next">Next</button>
<button id="dl">Download CSV</button></div>
<script>
const IMAGES = __IMAGES__, CLASSES = __CLASSES__, NONE = "none of the 12";
let i = 0, labels = {};
const key = () => "labels:" + (document.getElementById("who").value.trim() || "anon");
function load(){ try { labels = JSON.parse(localStorage.getItem(key()) || "{}"); } catch(e) { labels = {}; } }
function save(){ try { localStorage.setItem(key(), JSON.stringify(labels)); } catch(e) {} }
function draw(){
  const img = IMAGES[i];
  document.getElementById("card").innerHTML = '<img alt="" src="' + encodeURI(img) + '"><div class="muted">' + img + '</div>';
  const o = document.getElementById("opts"); o.innerHTML = "";
  for (const c of [...CLASSES, NONE]) {
    const b = document.createElement("button"); b.textContent = c;
    if (labels[img] === c) b.className = "on";
    b.onclick = () => { labels[img] = c; save(); if (i < IMAGES.length - 1) i++; draw(); };
    o.appendChild(b);
  }
  const done = IMAGES.filter(x => labels[x]).length;
  document.getElementById("prog").textContent = "photo " + (i + 1) + " of " + IMAGES.length + " - " + done + " labelled";
}
document.getElementById("who").onchange = () => { load(); draw(); };
document.getElementById("prev").onclick = () => { if (i > 0) i--; draw(); };
document.getElementById("next").onclick = () => { if (i < IMAGES.length - 1) i++; draw(); };
document.getElementById("dl").onclick = () => {
  const who = document.getElementById("who").value.trim();
  if (!who) { alert("Enter your name first."); return; }
  const rows = IMAGES.filter(x => labels[x] && labels[x] !== NONE).map(x => '"' + x.replace(/"/g, '""') + '",' + labels[x]);
  const blob = new Blob(["image,label\\n" + rows.join("\\n") + "\\n"], {type: "text/csv"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "labels_" + who.replace(/[^A-Za-z0-9_-]/g, "_") + ".csv"; a.click();
};
load(); draw();
</script></body></html>
"""


def images_in(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def build_page(images: list[str]) -> str:
    if not images:
        raise ValueError("no photographs found")
    return (PAGE.replace("__IMAGES__", json.dumps(images))
            .replace("__CLASSES__", json.dumps(list(TEXTILE_CLASSES)))
            .replace("__DTD__", DTD_BROWSE))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="label_kit")
    ap.add_argument("--photos", required=True)
    a = ap.parse_args(argv)
    folder = Path(a.photos)
    imgs = images_in(folder)
    (folder / "label.html").write_text(build_page(imgs), encoding="utf-8")
    print(f"{folder / 'label.html'}: {len(imgs)} photographs, {len(TEXTILE_CLASSES)} classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
