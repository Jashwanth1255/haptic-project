"""Build the report's six UML diagrams as StarUML project files (.mdj).

    python report/uml/make_mdj.py            # write report/uml/staruml/*.mdj
    python report/uml/make_mdj.py --resave   # then let StarUML load and re-save each file

The diagrams are the same as the .mmd sources next to this file. --resave runs
`StarUML exec <file> -c project:save`, so the saved files carry every view
StarUML itself writes (compartments, labels, edge points).
"""
from __future__ import annotations

import argparse
import base64
import json
import random
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent / "staruml"
STARUML = Path(r"C:\Program Files\StarUML\StarUML.exe")
_rng = random.Random(15)


def new_id() -> str:
    return "AAAAAAGa" + base64.b64encode(bytes(_rng.randrange(256) for _ in range(9))).decode()


def ref(e):
    return {"$ref": e["_id"]}


def el(type_, parent, **kw):
    e = {"_type": type_, "_id": new_id()}
    if parent is not None:
        e["_parent"] = ref(parent)
    e.update(kw)
    return e


def own(parent, child, key="ownedElements"):
    parent.setdefault(key, []).append(child)
    return child


def text_width(lines, char=7.2, pad=24, minimum=80):
    return max(minimum, int(max(len(s) for s in lines) * char + pad))


class Diagram:
    def __init__(self, type_, owner, name):
        self.d = own(owner, el(type_, owner, name=name))

    def node(self, type_, model, x, y, w, h, **kw):
        v = el(type_, self.d, model=ref(model), left=x, top=y, width=w, height=h, **kw)
        return own(self.d, v, "ownedViews")

    def edge(self, type_, model, tail, head, points=None, **kw):
        v = el(type_, self.d, model=ref(model), tail=ref(tail), head=ref(head), lineStyle=1, **kw)
        # StarUML draws an edge only when it has points; it then re-routes the ends to the box borders.
        points = points or [centre(tail), centre(head)]
        v["points"] = ";".join(f"{int(x)}:{int(y)}" for x, y in points)
        return own(self.d, v, "ownedViews")


NO_COMPARTMENTS = {"suppressAttributes": True, "suppressOperations": True}


def centre(v):
    return v["left"] + v["width"] / 2, v["top"] + v["height"] / 2


# --------------------------------------------------------------------------- architecture
def architecture(project):
    m = own(project, el("UMLModel", project, name="Architecture"))
    dg = Diagram("UMLComponentDiagram", m, "System architecture")
    pk, views = {}, {}

    def package(key, name, x, y, w, h):
        p = own(m, el("UMLPackage", m, name=name))
        pk[key] = p
        dg.node("UMLPackageView", p, x, y, w, h)

    def part(key, kind, pkg, name, x, y, w=250, h=56):
        owner = pk[pkg] if pkg else m
        e = own(owner, el(kind, owner, name=name))
        views[key] = (e, dg.node(kind + "View", e, x, y, w, h, **NO_COMPARTMENTS))

    package("data", "Public data", 20, 20, 1200, 130)
    package("o1", "O1 Haptic database", 20, 190, 290, 330)
    package("o2", "O2 Visual classification", 340, 190, 560, 240)
    package("o3", "O3 Association and velocity", 340, 460, 560, 120)
    package("o4", "O4 Rendering and interface", 930, 190, 290, 240)
    package("o5", "O5 User study", 930, 460, 290, 120)
    part("CL", "UMLArtifact", "data", "Cluster Haptic Texture Dataset (118 materials, 16 cloth)", 35, 70, 400)
    part("MI", "UMLArtifact", "data", "MINC-2500 material photos", 455, 70, 220)
    part("DT", "UMLArtifact", "data", "DTD texture photos", 695, 70, 170)
    part("DF", "UMLArtifact", "data", "DeepFashion-MultiModal product photos", 885, 70, 320)
    part("LD", "UMLComponent", "o1", "Loader and steady-sliding window", 40, 240)
    part("DE", "UMLComponent", "o1", "Descriptor extraction (20-1000 Hz)", 40, 340)
    part("GT", "UMLComponent", "o1", "Separability gate and statistics", 40, 440)
    part("RM", "UMLComponent", "o2", "ResNet-50 material model", 360, 240)
    part("RT", "UMLComponent", "o2", "ResNet-50 texture model", 630, 240)
    part("TR", "UMLComponent", "o2", "Scan to photo transfer test", 495, 340)
    part("AS", "UMLComponent", "o3", "Class-pair lookup or embedding similarity; velocity analysis", 380, 505, 480)
    part("EN", "UMLComponent", "o4", "On/off encoder (10 ms tick)", 950, 240)
    part("WA", "UMLComponent", "o4", "FastAPI web app (Vibration API)", 950, 340)
    part("RI", "UMLComponent", "o5", "Rating instrument", 950, 505)
    part("RL", "UMLArtifact", None, "Run log (config, metrics, commit and source hash)", 380, 630, 400)
    for pair in ["CL LD", "LD DE", "DE GT", "MI RM", "DT RT", "DF TR", "RM TR", "RT TR", "RT AS", "DE AS",
                 "AS EN", "EN WA", "WA RI", "GT RL", "TR RL", "AS RL"]:
        a, b = pair.split()
        (sa, va), (sb, vb) = views[a], views[b]
        dep = own(sa, el("UMLDependency", sa, source=ref(sa), target=ref(sb)))
        dg.edge("UMLDependencyView", dep, va, vb)
    return m


# --------------------------------------------------------------------------- class diagram
CLASSES = {
    "Descriptor": (["material_id: int", "velocity_mm_s: float", "force_n: float", "frequency_hz: float",
                    "intensity: float", "pattern: float", "duration_ms: float"],
                   ["core_vector(): ndarray", "full_vector(): ndarray", "as_dict(): dict"]),
    "DescriptorExtractor": ([], ["to_magnitude(accel): ndarray", "bandpass(x, fs): ndarray",
                                 "spectral_centroid(f, p): float", "spectral_entropy(f, p): float",
                                 "envelope_period_ms(env, fs): float", "extract(accel, fs): Descriptor",
                                 "normalise(d): dict"]),
    "ClusterLoader": ([], ["discover(root, material_ids): list", "steady_window(position, velocity): tuple",
                           "load(stub, fs): dict", "load_names(root): dict"]),
    "SeparabilityGate": ([], ["anova_by_dimension(df): DataFrame", "holm_adjust(p_values): ndarray",
                              "pairwise_separation(df): DataFrame", "run_gate(descriptors): GateResult"]),
    "GateResult": (["passed: bool", "silhouette: float", "n_materials: int", "effective_classes: int",
                    "collapsed_pairs: list"], ["summary(): str"]),
    "RenderConfig": (["intensity_ref: float", "duration_ms: int", "tick_ms: int"], []),
    "VibrationPattern": (["scheme: str", "durations: tuple", "tick_ms: int"], ["total_ms(): int", "duty(): float"]),
    "Fidelity": (["scheme: str", "log_spectral_db: float", "wasserstein_hz: float", "level_error: float"],
                 ["renderable(): bool"]),
    "Renderer": ([], ["encode_a(d, cfg): VibrationPattern", "encode_b(d, cfg, envelope): VibrationPattern",
                      "encode_c(d, cfg): VibrationPattern", "encode_d(reference, seed): VibrationPattern",
                      "fidelity(envelope, pattern): Fidelity"]),
    "TextureClassifier": (["model: ResNet50"], ["classify(image): list"]),
    "Retrieved": (["fabric: str", "signal: ndarray", "envelope: ndarray", "descriptor: Descriptor"], []),
    "Pipeline": ([], ["load_classifier(checkpoint): TextureClassifier", "build_library(): dict",
                      "analyse(image, classify, library, cfg): dict"]),
    "WebApp": ([], ["index(): page", "analyse(image): json", "session(record): json", "latency(record): json"]),
    "SessionLog": (["dir: Path"], ["session(data): dict", "latency(data): dict", "latency_summary(): dict"]),
    "RunLog": ([], ["save_run(name, config, metrics): Path", "load_run(name): Run", "source_digest(): str"]),
    "Run": (["path: Path", "info: dict", "config: dict", "metrics: dict"], ["table(name): DataFrame"]),
}
# (first x, classes) per row; placed so that most relations run between neighbours
CLASS_ROWS = [
    (20, ["ClusterLoader", "DescriptorExtractor", "Descriptor", "SeparabilityGate", "GateResult"]),
    (20, ["WebApp", "Pipeline", "Retrieved", "RunLog", "Run"]),
    (20, ["SessionLog", "TextureClassifier", "Renderer"]),
    (520, ["RenderConfig", "VibrationPattern", "Fidelity"]),
]


def _operation(cls, sig):
    name, rest = sig.split("(", 1)
    args, ret = rest.split(")", 1)
    op = el("UMLOperation", cls, name=name)
    for a in [s.strip() for s in args.split(",") if s.strip()]:
        own(op, el("UMLParameter", op, name=a), "parameters")
    ret = ret.lstrip(": ").strip()
    if ret:
        own(op, el("UMLParameter", op, type=ret, direction="return"), "parameters")
    return op


def class_diagram(project):
    m = own(project, el("UMLModel", project, name="Design Model"))
    dg = Diagram("UMLClassDiagram", m, "Class diagram of the main modules")
    cls, views = {}, {}
    y = 20
    for x, row in CLASS_ROWS:
        tallest = 0
        for name in row:
            attrs, ops = CLASSES[name]
            c = own(m, el("UMLClass", m, name=name))
            for a in attrs:
                n, t = a.split(": ")
                own(c, el("UMLAttribute", c, name=n, type=t), "attributes")
            for o in ops:
                own(c, _operation(c, o), "operations")
            w = text_width(["+" + s for s in attrs + ops] + [name], pad=30, minimum=150)
            h = 36 + 16 * max(1, len(attrs)) + 16 * max(1, len(ops)) + 16
            cls[name], views[name] = c, dg.node("UMLClassView", c, x, y, w, h)
            x += w + 110
            tallest = max(tallest, h)
        y += tallest + 90

    def dependency(a, b, label):
        d = own(cls[a], el("UMLDependency", cls[a], name=label, source=ref(cls[a]), target=ref(cls[b])))
        dg.edge("UMLDependencyView", d, views[a], views[b])

    def association(a, b, label="", agg_a="none", nav_b="navigable"):
        asc = el("UMLAssociation", cls[a], name=label)
        asc["end1"] = el("UMLAssociationEnd", asc, reference=ref(cls[a]), aggregation=agg_a)
        asc["end2"] = el("UMLAssociationEnd", asc, reference=ref(cls[b]), navigable=nav_b)
        own(cls[a], asc)
        dg.edge("UMLAssociationView", asc, views[a], views[b])

    dependency("ClusterLoader", "DescriptorExtractor", "supplies recordings")
    dependency("DescriptorExtractor", "Descriptor", "creates")
    dependency("SeparabilityGate", "Descriptor", "analyses")
    dependency("SeparabilityGate", "GateResult", "returns")
    dependency("Renderer", "RenderConfig", "uses")
    dependency("Renderer", "VibrationPattern", "creates")
    dependency("Renderer", "Fidelity", "scores")
    association("Retrieved", "Descriptor", agg_a="composite", nav_b="unspecified")
    association("Pipeline", "TextureClassifier")
    association("Pipeline", "Retrieved", "library", agg_a="shared", nav_b="unspecified")
    dependency("Pipeline", "Renderer", "encodes with")
    association("WebApp", "Pipeline")
    association("WebApp", "SessionLog")
    dependency("RunLog", "Run", "saves and loads")
    dependency("SeparabilityGate", "RunLog", "saves results")
    return m


# --------------------------------------------------------------------------- sequence diagram
SEQ_PARTICIPANTS = ["Shopper", "Web page", "FastAPI app", "Pipeline", "DTD ResNet-50", "Renderer", "SessionLog"]
SEQ_MESSAGES = [
    (0, 1, "Choose garment photo", "synchCall"),
    (1, 2, "POST /api/analyse (image)", "synchCall"),
    (2, 3, "analyse(image)", "synchCall"),
    (3, 4, "classify(image)", "synchCall"),
    (4, 3, "ranked textile classes", "reply"),
    (3, 3, "look up descriptor for top class", "synchCall"),
    (3, 5, "encode patterns A-D and fidelity", "synchCall"),
    (5, 3, "patterns and fidelity scores", "reply"),
    (3, 2, "result with timings", "reply"),
    (2, 1, "JSON (class, waveform, spectrum, patterns)", "reply"),
    (1, 0, "show waveform, spectrum and top-3 classes", "reply"),
    (0, 1, "Tap Play", "synchCall"),
    (1, 1, "navigator.vibrate(pattern)", "synchCall"),
    (1, 0, "phone vibrates", "reply"),
    (1, 2, "POST /api/latency (timings)", "asynchCall"),
    (2, 6, "append latency record", "synchCall"),
    (1, 2, "POST /api/session (device model)", "asynchCall"),
    (2, 6, "append session record", "synchCall"),
]


def sequence_diagram(project):
    m = own(project, el("UMLModel", project, name="Interaction Model"))
    collab = own(m, el("UMLCollaboration", m, name="Feel this fabric"))
    inter = own(collab, el("UMLInteraction", collab, name="Feel this fabric"))
    dg = Diagram("UMLSequenceDiagram", inter, "Sequence diagram: feel this fabric")
    gap, x0, y0, width = 200, 40, 40, 150
    bottom = y0 + 110 + 40 * len(SEQ_MESSAGES) + 40
    lines = []
    for i, name in enumerate(SEQ_PARTICIPANTS):
        role = own(collab, el("UMLAttribute", collab, name=name), "attributes")
        life = own(inter, el("UMLLifeline", inter, name=name, represent=ref(role)), "participants")
        lv = dg.node("UMLSeqLifelineView", life, x0 + i * gap, y0, width, bottom - y0)
        part = el("UMLLinePartView", lv, model=ref(life), left=x0 + i * gap + width // 2, top=y0 + 40,
                  width=1, height=bottom - y0 - 40)
        lv["subViews"] = [part]
        lv["linePart"] = ref(part)
        lines.append((life, part, x0 + i * gap + width // 2))
    y = y0 + 90
    for src, dst, text, sort in SEQ_MESSAGES:
        (ls, ps, xs), (lt, pt, xt) = lines[src], lines[dst]
        msg = own(inter, el("UMLMessage", inter, name=text, messageSort=sort, source=ref(ls), target=ref(lt)),
                  "messages")
        pts = [(xs, y), (xt, y)] if src != dst else [(xs, y), (xs + 40, y), (xs + 40, y + 16), (xs, y + 16)]
        dg.edge("UMLSeqMessageView", msg, ps, pt, points=pts)
        y += 40 + (16 if src == dst else 0)
    return m


# --------------------------------------------------------------------------- activity diagram
def activity_diagram(project):
    m = own(project, el("UMLModel", project, name="Activity Model"))
    act = own(m, el("UMLActivity", m, name="Image to vibration"))
    dg = Diagram("UMLActivityDiagram", act, "Activity diagram: image to vibration")
    nodes = {}

    def node(key, kind, name, x, y, w, h, view="UMLControlNodeView"):
        n = own(act, el(kind, act, name=name), "nodes")
        nodes[key] = (n, dg.node(view, n, x, y, w, h))

    def action(key, name, x, y, w=330):
        node(key, "UMLAction", name, x, y, w, 44, view="UMLActionView")

    cx = 330
    node("A", "UMLInitialNode", "Start", cx - 10, 20, 20, 20)
    steps = ["Garment photo uploaded", "Classify texture with DTD ResNet-50",
             "Take top class and top-3 probabilities", "Retrieve descriptor for the class",
             "Encode on/off patterns with a 10 ms tick", "Score fidelity of each pattern",
             "Show waveform, spectrum and fidelity"]
    for i, s in enumerate(steps):
        action("BCDEFGH"[i], s, cx - 165, 70 + 74 * i)
    node("I", "UMLDecisionNode", "Can this browser vibrate?", cx - 15, 600, 30, 40)
    action("J", "Show waveform and spectrum only", 30, 700, 260)
    node("K", "UMLDecisionNode", "Compare mode on?", 575, 690, 30, 40)
    action("M", "Load second fabric and play both in turn", 660, 780, 320)
    action("L", "Play pattern with navigator.vibrate", 425, 870, 330)
    action("N", "Log device model and latency", 425, 950, 330)
    node("O", "UMLActivityFinalNode", "End", cx - 13, 1040, 26, 26)

    def flow(a, b, guard=""):
        (na, va), (nb, vb) = nodes[a], nodes[b]
        f = own(act, el("UMLControlFlow", act, source=ref(na), target=ref(nb), guard=guard), "edges")
        dg.edge("UMLControlFlowView", f, va, vb)

    for a, b in ["AB", "BC", "CD", "DE", "EF", "FG", "GH", "HI"]:
        flow(a, b)
    flow("I", "J", "No: iOS or desktop")
    flow("I", "K", "Yes: Android")
    flow("K", "L", "No")
    flow("K", "M", "Yes")
    flow("M", "L")
    flow("L", "N")
    flow("J", "O")
    flow("N", "O")
    return m


# --------------------------------------------------------------------------- state diagram
def state_diagram(project):
    m = own(project, el("UMLModel", project, name="State Model"))
    sm = own(m, el("UMLStateMachine", m, name="Rendering session"))
    region = own(sm, el("UMLRegion", sm), "regions")
    dg = Diagram("UMLStatechartDiagram", sm, "State diagram of a rendering session")
    v = {}

    def vertex(key, type_, x, y, w, h, view, **kw):
        e = own(region, el(type_, region, **kw), "vertices")
        v[key] = (e, dg.node(view, e, x, y, w, h))

    vertex("start", "UMLPseudostate", 40, 165, 20, 20, "UMLPseudostateView", kind="initial")
    for key, x, y in [("Idle", 120, 150), ("Analysing", 380, 150), ("Ready", 680, 150),
                      ("Error", 380, 360), ("Playing", 980, 40), ("Comparing", 980, 270)]:
        vertex(key, "UMLState", x, y, 130, 50, "UMLStateView", name=key)
    vertex("end", "UMLFinalState", 732, 420, 26, 26, "UMLFinalStateView")

    def transition(a, b, trigger="", bend=None):
        (ea, va), (eb, vb) = v[a], v[b]
        t = el("UMLTransition", region, source=ref(ea), target=ref(eb))
        if trigger:
            t["triggers"] = [el("UMLEvent", t, name=trigger)]
        own(region, t, "transitions")
        pts = None
        if bend:
            (ax, ay), (bx, by) = centre(va), centre(vb)
            pts = [(ax, ay), *bend, (bx, by)]
        dg.edge("UMLTransitionView", t, va, vb, points=pts)

    transition("start", "Idle")
    transition("Idle", "Analysing", "photo chosen")
    transition("Analysing", "Ready", "result received")
    transition("Ready", "Analysing", "new photo chosen", bend=[(745, 100), (445, 100)])
    transition("Analysing", "Error", "bad image or server error")
    transition("Error", "Idle", "choose another photo")
    transition("Ready", "Playing", "tap Play")
    transition("Playing", "Ready", "pattern finished", bend=[(1045, 140), (810, 140)])
    transition("Ready", "Comparing", "compare mode on")
    transition("Comparing", "Ready", "compare mode off", bend=[(1045, 240), (810, 240)])
    transition("Ready", "end", "page closed")
    return m


# --------------------------------------------------------------------------- use case diagram
def use_case_diagram(project):
    m = own(project, el("UMLModel", project, name="Use Case Model"))
    dg = Diagram("UMLUseCaseDiagram", m, "Use case diagram")
    v = {}

    def actor(key, name, x, y):
        e = own(m, el("UMLActor", m, name=name))
        v[key] = (e, dg.node("UMLActorView", e, x, y, 40, 80, **NO_COMPARTMENTS))

    def case(key, name, x, y):
        e = own(m, el("UMLUseCase", m, name=name))
        v[key] = (e, dg.node("UMLUseCaseView", e, x, y, 220, 50, **NO_COMPARTMENTS))

    actor("shopper", "Shopper", 40, 170)
    actor("participant", "Study participant", 40, 560)
    actor("researcher", "Researcher", 980, 520)
    for key, name, x, y in [("upload", "Upload garment photo", 200, 40),
                            ("classify", "Classify fabric texture", 580, 40),
                            ("view", "View waveform and spectrum", 200, 130),
                            ("feel", "Feel fabric (play vibration)", 200, 220),
                            ("log", "Log device model and latency", 580, 220),
                            ("compare", "Compare two fabrics", 200, 310),
                            ("rate", "Rate image-vibration match", 200, 580),
                            ("extract", "Extract haptic descriptors", 600, 390),
                            ("gate", "Run separability gate", 600, 470),
                            ("train", "Train image classifiers", 600, 550),
                            ("transfer", "Evaluate scan-to-photo transfer", 600, 630),
                            ("label", "Label product photos", 600, 710)]:
        case(key, name, x, y)

    def associate(a, b):
        (ea, va), (eb, vb) = v[a], v[b]
        asc = el("UMLAssociation", m)
        asc["end1"] = el("UMLAssociationEnd", asc, reference=ref(ea))
        asc["end2"] = el("UMLAssociationEnd", asc, reference=ref(eb))
        own(m, asc)
        dg.edge("UMLAssociationView", asc, va, vb)

    def relation(kind, a, b):
        (ea, va), (eb, vb) = v[a], v[b]
        r = own(ea, el(kind, ea, source=ref(ea), target=ref(eb)))
        dg.edge(kind + "View", r, va, vb)

    for u in ["upload", "view", "feel", "compare"]:
        associate("shopper", u)
    for u in ["extract", "gate", "train", "transfer", "label"]:
        associate("researcher", u)
    associate("participant", "rate")
    relation("UMLInclude", "upload", "classify")
    relation("UMLInclude", "feel", "log")
    relation("UMLExtend", "compare", "feel")
    relation("UMLInclude", "gate", "extract")
    relation("UMLInclude", "transfer", "label")
    return m


DIAGRAMS = [
    ("00_architecture", architecture),
    ("01_class_diagram", class_diagram),
    ("02_sequence_feel_fabric", sequence_diagram),
    ("03_activity_pipeline", activity_diagram),
    ("04_state_session", state_diagram),
    ("05_use_case", use_case_diagram),
]


def project(name, builders):
    p = {"_type": "Project", "_id": new_id(), "name": name, "ownedElements": []}
    first = True
    for build in builders:
        model = build(p)
        if first:  # open the first diagram when the file is opened
            stack = [model]
            while stack:
                e = stack.pop()
                if e.get("_type", "").endswith("Diagram"):
                    e["defaultDiagram"] = True
                    break
                stack.extend(reversed(e.get("ownedElements", [])))
            first = False
    return p


def resave(path: Path) -> None:
    r = subprocess.run([str(STARUML), "exec", str(path), "-c", "project:save"],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"StarUML could not re-save {path.name}: {r.stdout}{r.stderr}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resave", action="store_true", help="load and re-save each file with StarUML")
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    files = []
    for stem, build in DIAGRAMS:
        f = OUT / f"{stem}.mdj"
        f.write_text(json.dumps(project("Tactile Virtualization", [build]), indent=2), encoding="utf-8")
        files.append(f)
    allf = OUT / "Tactile_Virtualization_UML.mdj"
    allf.write_text(json.dumps(project("Tactile Virtualization", [b for _, b in DIAGRAMS]), indent=2),
                    encoding="utf-8")
    files.append(allf)
    if args.resave:
        if not STARUML.is_file():
            print(f"StarUML not found at {STARUML}; files written without re-saving.", file=sys.stderr)
        else:
            for f in files:
                resave(f)
    for f in files:
        print(f.relative_to(OUT.parent.parent.parent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
