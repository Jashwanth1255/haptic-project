// Use case diagram for StarUML. Mermaid cannot describe use case diagrams, so this
// builds one with StarUML's scripting API.
// How to run: View > Script Console (Ctrl+Alt+J), paste everything, press Enter.
// Not tested by us: if the console reports an error, use the manual steps in README.md.

const project = app.repository.select("@Project")[0];
const model = app.factory.createModel({
  id: "UMLModel", parent: project,
  modelInitializer: (m) => { m.name = "Use Case Model"; },
});
const diagram = app.factory.createDiagram({
  id: "UMLUseCaseDiagram", parent: model,
  diagramInitializer: (d) => { d.name = "Tactile Virtualization - Use Cases"; },
});

function actor(name, x, y) {
  return app.factory.createModelAndView({
    id: "UMLActor", parent: model, diagram: diagram,
    x1: x, y1: y, x2: x + 40, y2: y + 80,
    modelInitializer: (m) => { m.name = name; },
  });
}
function useCase(name, x, y) {
  return app.factory.createModelAndView({
    id: "UMLUseCase", parent: model, diagram: diagram,
    x1: x, y1: y, x2: x + 220, y2: y + 50,
    modelInitializer: (m) => { m.name = name; },
  });
}
function link(kind, from, to, owner) {
  return app.factory.createModelAndView({
    id: kind, parent: owner || model, diagram: diagram,
    tailView: from, headView: to, tailModel: from.model, headModel: to.model,
  });
}

// actors
const shopper = actor("Shopper", 40, 140);
const researcher = actor("Researcher", 760, 140);
const participant = actor("Study participant", 40, 520);

// shopper use cases
const upload = useCase("Upload garment photo", 200, 40);
const classify = useCase("Classify fabric texture", 470, 40);
const view = useCase("View waveform and spectrum", 200, 120);
const feel = useCase("Feel fabric (play vibration)", 200, 200);
const compare = useCase("Compare two fabrics", 200, 280);
const log = useCase("Log device model and latency", 470, 200);

// researcher use cases
const extract = useCase("Extract haptic descriptors", 470, 300);
const gate = useCase("Run separability gate", 470, 370);
const train = useCase("Train image classifiers", 470, 440);
const transfer = useCase("Evaluate scan-to-photo transfer", 470, 510);
const label = useCase("Label product photos", 470, 580);

// study use case
const rate = useCase("Rate image-vibration match", 200, 540);

[upload, view, feel, compare].forEach((u) => link("UMLAssociation", shopper, u));
[extract, gate, train, transfer, label].forEach((u) => link("UMLAssociation", researcher, u));
link("UMLAssociation", participant, rate);

link("UMLInclude", upload, classify, upload.model);
link("UMLInclude", feel, log, feel.model);
link("UMLExtend", compare, feel, compare.model);
link("UMLInclude", gate, extract, gate.model);
link("UMLInclude", transfer, label, transfer.model);

app.diagrams.repaint && app.diagrams.repaint();
"Use case diagram created: " + diagram.name;
