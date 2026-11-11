# UML diagrams for the Phase I report (StarUML)

Each `.mmd` file is Mermaid code that StarUML turns into a native, editable UML
diagram. All four were checked with the Mermaid 10.9.1 parser and render without
errors.

| File | Diagram | Report figure |
|---|---|---|
| `03_activity_pipeline.mmd` | Activity (flowchart) of the image-to-vibration pipeline | Fig. 3.1 |
| `05_use_case_script.js` | Use case diagram (script, see below) | Fig. 3.2 |
| `01_class_diagram.mmd` | Class diagram of the main modules | Fig. 3.3 |
| `02_sequence_feel_fabric.mmd` | Sequence diagram for "feel this fabric" | Fig. 3.4 |
| `04_state_session.mmd` | State diagram of a rendering session | Fig. 3.5 |

## Mermaid files (class, sequence, activity, state)

1. Open StarUML and create or open a project.
2. Choose **Tools > Generate Diagram by Mermaid...**
3. Delete the example code, paste the whole contents of one `.mmd` file, and press **OK**.
4. Tidy the layout if needed (drag boxes; **Format > Auto Layout**).
5. Export the image with **File > Export Diagram As > PNG** and paste it over the
   matching yellow placeholder in the report.

Repeat for each file.

Notes:
- The sequence diagram uses `participant Shopper` rather than `actor`, because
  StarUML's sequence importer does not support actors. You can change the Shopper
  lifeline to an actor afterwards in StarUML if you want the stick figure.
- In the class diagram, `DescriptorExtractor`, `ClusterLoader`, `SeparabilityGate`,
  `Renderer`, `Pipeline` and `RunLog` stand for Python modules (their functions are
  shown as operations). The other classes are real Python classes.

## Use case diagram (script)

Mermaid has no use case diagram, so `05_use_case_script.js` builds it with
StarUML's scripting API:

1. Open **View > Script Console** (Ctrl+Alt+J).
2. Paste the whole script and press Enter.
3. A diagram called "Tactile Virtualization - Use Cases" appears in the Model
   Explorer. Open it and tidy the layout.

We could not run this script ourselves. If the console shows an error, draw the
diagram by hand from the palette. It has:

- Actors: Shopper, Researcher, Study participant.
- Shopper: Upload garment photo, View waveform and spectrum, Feel fabric (play
  vibration), Compare two fabrics.
- Researcher: Extract haptic descriptors, Run separability gate, Train image
  classifiers, Evaluate scan-to-photo transfer, Label product photos.
- Study participant: Rate image-vibration match.
- Include: Upload garment photo -> Classify fabric texture; Feel fabric -> Log
  device model and latency; Run separability gate -> Extract haptic descriptors;
  Evaluate scan-to-photo transfer -> Label product photos.
- Extend: Compare two fabrics -> Feel fabric.
