# Objective 4: rendering and visualisation interface

## The constraint

`navigator.vibrate()` takes a list of millisecond on/off durations. It has no
amplitude control and no frequency control. Practical timing granularity on Android
is 10–20 ms, so the fastest on/off switching is about 50 Hz, and 25 Hz at the 20 ms
tick used by default. Texture energy lives at 20–1000 Hz.

**The carrier frequency cannot be reproduced, and nothing here pretends to.** The
vibration the skin receives is set by the motor's own resonance, whatever the pattern
says. Three things can be encoded:

| Quantity | How | Limit |
|---|---|---|
| Intensity | share of time the motor is on | no real amplitude; one level per tick |
| Envelope modulation | timing of pulses | lost above the switching limit |
| Coarseness | a slow pulse rate standing in for the spectral centroid | a code, not the frequency |

## The four encoders (`haptic/render.py`)

| Scheme | Encoding | Carries |
|---|---|---|
| A | fixed 200 ms period, duty proportional to intensity | intensity |
| B | pulse density by first-order sigma-delta on the low-passed envelope | intensity, envelope |
| C | pulse period 40–400 ms from log centroid, 50% duty | coarseness |
| D | B's ticks shuffled, same total duration and duty | nothing: control |

C's rate is a mapping onto a renderable range. A 400 Hz texture does not become a
400 Hz vibration.

B low-passes the envelope below the switching limit first, because modulation that
cannot be played would otherwise alias into a slower rhythm the texture does not have.

## Fidelity

`fidelity()` compares intended and achievable **envelope** spectra over the playable
band, using log-spectral distance (dB) and Wasserstein distance (Hz). Raw acceleration
spectra are deliberately not compared: that gap is impossible to close by construction,
so any number from it would mislead. The page's FFT plot is for display only.

Spectra are normalised, so the two distances describe the shape of the modulation.
Level is scored separately as `level_error`. `in_band_fraction` reports how much of
the intended modulation can be played at all. Below 50% a result is flagged not
`renderable`, and the distances describe only what is left.

`make_o4_table.py` prints the schemes over the synthetic bank. **Those numbers are
scaffolding, not a result.** They show how the metric behaves and say nothing about
what a person feels.

The achievable envelope uses a first-order motor lag of 25 ms. **That value is assumed,
not measured.** LRA and ERM motors differ, and no device has been characterised.

## The interface

```bash
pip install fastapi uvicorn python-multipart
python make_o4_samples.py
python -m webapp.app
```

Then open http://127.0.0.1:8765.

- `webapp/pipeline.py`: image → class → descriptor → patterns and fidelity
- `webapp/app.py`: HTTP service
- `webapp/sessions.py`: session and latency logs, written to `logs/o4/`
- `webapp/static/`: the single page

**Retrieval is a placeholder.** `CLASS_TO_FABRIC` assigns each of the 12 textile
classes a fictional fabric by hand, with no evidence, until Objective 3 replaces it.
All descriptors, waveforms and spectra are synthetic. The classifier is the DTD fold 1
model, restricted to the 12 textile classes, and is untested on product photographs.
The sample images are generated patterns.

### Vibration support

| Device | What the page does |
|---|---|
| Android Chrome | Plays |
| iOS, any browser | Plots only, with a message. All iOS browsers run on WebKit, which has no Vibration API |
| No API elsewhere | Plots only, with a message |
| Desktop with the API | Warns: the function exists and returns true, but there is usually no motor |

Chrome refuses `vibrate()` until the person has tapped the page. Selecting a sample or
file counts. The page cannot confirm the motor ran, and cannot detect Android's own
vibration settings being off.

### Reaching it from a phone

The real device model comes from `userAgentData.getHighEntropyValues()`, because
Chrome freezes the model in its user agent string. That call needs a secure context,
so a phone must use HTTPS, or a USB connection:

```bash
adb reverse tcp:8765 tcp:8765
```

Over plain HTTP on a LAN address the session logs `device_model: unknown`, which would
break Objective 5's device analysis.

### Latency

Every playback logs the time from image selection to the `vibrate()` call. The call is
the last moment a page can observe, so motor spin-up is not included. `machine_ms`
excludes any wait for a person to press play. `GET /api/latency` returns the median
and p95. Report numbers from real phones on a real network.

## Not yet verified

- Nothing has run on a physical phone, and the target is at least 3 Android models.
- The 99-entry pattern limit is Chromium's as remembered. Check it on a device.
- The motor lag is assumed.
- Fidelity says how well an envelope is approximated, not whether anyone can feel the
  difference. That is Objective 5.
