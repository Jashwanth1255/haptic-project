"""Render the report's UML diagrams to PNG with headless Edge.

    python report/render_uml.py

Class, sequence, activity and state diagrams come from the same Mermaid files
that StarUML imports (report/uml/*.mmd). Mermaid has no use case diagram, so that
one is drawn here as SVG from the same actors, use cases and relations as
report/uml/05_use_case_script.js. Output: report/figures/uml_*.png
"""
from __future__ import annotations

import html
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parent
UML = ROOT / "uml"
OUT = ROOT / "figures"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<style>body{{margin:0;padding:24px;background:#fff;font-family:'Times New Roman',serif}}</style>{head}</head>
<body>{body}</body></html>"""


def mermaid_page(code: str) -> str:
    head = (f'<script src="{MERMAID}"></script><script>mermaid.initialize({{startOnLoad:true,'
            f'theme:"neutral",securityLevel:"loose",fontFamily:"Times New Roman",'
            f'flowchart:{{curve:"basis"}},sequence:{{mirrorActors:false}}}});</script>')
    return PAGE.format(head=head, body=f'<pre class="mermaid">{html.escape(code)}</pre>')


# ---------------------------------------------------------------- use case SVG
ACTORS = {"Shopper": (90, 240), "Study participant": (90, 650), "Researcher": (1070, 590)}
CASES = {  # name: (cx, cy)
    "Upload garment photo": (380, 100), "Classify fabric texture": (770, 100),
    "View waveform and spectrum": (380, 190), "Feel fabric (play vibration)": (380, 280),
    "Log device model and latency": (770, 280), "Compare two fabrics": (380, 380),
    "Rate image-vibration match": (380, 650),
    "Extract haptic descriptors": (770, 410), "Run separability gate": (770, 505),
    "Train image classifiers": (770, 600), "Evaluate scan-to-photo transfer": (770, 695),
    "Label product photos": (770, 790),
}
RX, RY = 125, 30
ASSOC = [("Shopper", c) for c in ("Upload garment photo", "View waveform and spectrum",
                                  "Feel fabric (play vibration)", "Compare two fabrics")]
ASSOC += [("Study participant", "Rate image-vibration match")]
ASSOC += [("Researcher", c) for c in ("Extract haptic descriptors", "Run separability gate",
                                      "Train image classifiers", "Evaluate scan-to-photo transfer",
                                      "Label product photos")]
DEPS = [("Upload garment photo", "Classify fabric texture", "include"),
        ("Feel fabric (play vibration)", "Log device model and latency", "include"),
        ("Compare two fabrics", "Feel fabric (play vibration)", "extend"),
        ("Run separability gate", "Extract haptic descriptors", "include"),
        ("Evaluate scan-to-photo transfer", "Label product photos", "include")]


def edge_point(cx, cy, tx, ty):
    """Point on the use-case ellipse boundary in the direction of (tx, ty)."""
    a = math.atan2(ty - cy, tx - cx)
    return cx + RX * math.cos(a), cy + RY * math.sin(a)


def actor_svg(name, x, y):
    return (f'<g stroke="#222" stroke-width="2" fill="none"><circle cx="{x}" cy="{y - 38}" r="12"/>'
            f'<line x1="{x}" y1="{y - 26}" x2="{x}" y2="{y + 8}"/><line x1="{x - 20}" y1="{y - 14}" x2="{x + 20}" y2="{y - 14}"/>'
            f'<line x1="{x}" y1="{y + 8}" x2="{x - 16}" y2="{y + 34}"/><line x1="{x}" y1="{y + 8}" x2="{x + 16}" y2="{y + 34}"/></g>'
            f'<text x="{x}" y="{y + 56}" text-anchor="middle" font-size="17">{html.escape(name)}</text>')


def use_case_svg() -> str:
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="860" font-family="Times New Roman">',
             '<defs><marker id="open" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="9" markerHeight="9" orient="auto">'
             '<path d="M0,0 L10,5 L0,10" fill="none" stroke="#222" stroke-width="1.5"/></marker></defs>',
             '<rect x="230" y="40" width="700" height="790" fill="none" stroke="#222" stroke-width="2"/>',
             '<text x="580" y="30" text-anchor="middle" font-size="18" font-weight="bold">Tactile Virtualization System</text>']
    for actor, case in ASSOC:
        ax, ay = ACTORS[actor]; cx, cy = CASES[case]
        sx = ax + (24 if cx > ax else -24)
        ex, ey = edge_point(cx, cy, sx, ay - 10)
        parts.append(f'<line x1="{sx}" y1="{ay - 10}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#222" stroke-width="1.5"/>')
    for a, b, kind in DEPS:
        (ax, ay), (bx, by) = CASES[a], CASES[b]
        sx, sy = edge_point(ax, ay, bx, by); ex, ey = edge_point(bx, by, ax, ay)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        parts.append(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="#222" stroke-width="1.5" '
                     f'stroke-dasharray="7,5" marker-end="url(#open)"/>')
        dx = 14 if abs(ex - sx) < 5 else 0
        parts.append(f'<text x="{mx + dx:.1f}" y="{my - 6:.1f}" text-anchor="{"start" if dx else "middle"}" font-size="15" '
                     f'font-style="italic" fill="#333">\u00ab{kind}\u00bb</text>')
    for name, (cx, cy) in CASES.items():
        parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{RX}" ry="{RY}" fill="#eef3fb" stroke="#222" stroke-width="1.8"/>'
                     f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" font-size="16">{html.escape(name)}</text>')
    for name, (x, y) in ACTORS.items():
        parts.append(actor_svg(name, x, y))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- rendering
def shoot(page: str, out: Path, width: int, height: int) -> None:
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "page.html"
        src.write_text(page, encoding="utf-8")
        raw = Path(t) / "shot.png"
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=2", f"--window-size={width},{height}",
                        "--virtual-time-budget=15000", f"--screenshot={raw}", src.as_uri()],
                       check=True, capture_output=True, timeout=120)
        im = Image.open(raw).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box is None:
        raise SystemExit(f"{out.name}: blank render")
    pad = 30
    im = im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                  min(box[2] + pad, im.width), min(box[3] + pad, im.height)))
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(out)
    print(out.name, im.size)


def main() -> None:
    jobs = [("00_architecture.mmd", "uml_architecture.png", 1500, 1600),
            ("01_class_diagram.mmd", "uml_class.png", 1700, 1500),
            ("02_sequence_feel_fabric.mmd", "uml_sequence.png", 1500, 1300),
            ("03_activity_pipeline.mmd", "uml_activity.png", 1000, 1900),
            ("04_state_session.mmd", "uml_state.png", 900, 1000)]
    for src, out, w, h in jobs:
        shoot(mermaid_page((UML / src).read_text(encoding="utf-8")), OUT / out, w, h)
    shoot(PAGE.format(head="", body=use_case_svg()), OUT / "uml_use_case.png", 1260, 940)


if __name__ == "__main__":
    main()
