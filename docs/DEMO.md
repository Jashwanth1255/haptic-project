# Showing the working system

## Before the meeting (once)

1. Install the packages: `pip install -r requirements.txt`
2. Make sure `models/dtd_fold1_seed0/best.pt` exists (about 95 MB). It is not in Git;
   copy the `models/` folder from the laptop that trained the models.
3. Do a trial run on the phone you will use (step 3 below). Vibration over the
   Wi-Fi address has not been tested on a phone yet; test it before the meeting.

## Running the demo

1. In the project folder, run `python -m webapp.demo`.
2. The page opens in the laptop's browser. The terminal also prints a phone address
   such as `http://192.168.1.73:8765`.
3. On an **Android** phone on the **same Wi-Fi**, open that address in **Chrome**.
   If Windows asks about the firewall, click *Allow* for private networks.
4. Tap a sample (weave, knit, pleat, felt) or upload a garment photo, then tap
   **Play vibration**. Tick *Compare two fabrics* to feel two in turn.
5. Press Ctrl+C in the terminal to stop.

If the phone cannot reach the laptop (some college Wi-Fi blocks devices from
seeing each other), connect the phone by USB with debugging on and run
`adb reverse tcp:8765 tcp:8765`, then open `http://localhost:8765` on the phone.
Or use the laptop's mobile hotspot for both devices.

## What to say about it

| Shown on screen | What it is |
|---|---|
| Predicted class and top-3 probabilities | Real: the DTD ResNet-50 trained in Phase I |
| Waveform, spectrum, four vibration schemes, fidelity table | Real code, applied to the retrieved signal |
| The retrieved fabric and descriptor | **Placeholder.** Each texture class maps to a made-up fabric; the page says "Synthetic data". Connecting real Cluster descriptors is Phase II work, because Phase I found that image-based retrieval does not beat the average descriptor. |
| Warning on the laptop | Expected: a laptop has no vibration motor. Feel it on the phone. |
| iPhone | Cannot vibrate from any web page (no iOS browser supports the Vibration API). Use Android. |

Known behaviour of the samples (checked 2026-11-10): weave → woven (98%),
pleat → pleated (55%), felt → matted (31%), knit → **woven** (53%). The knit
sample is misclassified; the texture model scores 80.7% on the 12 textile classes,
not 100%.

Analysis takes about 0.2–0.5 s on this laptop's CPU (classification about 180–220 ms).
