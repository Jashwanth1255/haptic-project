"use strict";
/* Objective 4 page. See haptic/render.py for what vibrate() can and cannot carry:
   the pattern never reproduces the texture's carrier frequency. */

const SAMPLES = ["weave", "knit", "pleat", "felt"];
const SCHEME_LABEL = { A: "Duty cycle", B: "Pulse density", C: "Pulse rate", D: "Random control" };
const PAUSE_BETWEEN_MS = 500;
const PAINT_WAIT_MS = 100;

const $ = (id) => document.getElementById(id);
const slots = [];
const sessionId = (crypto.randomUUID && crypto.randomUUID()) ||
  `s-${Date.now()}-${Math.random().toString(36).slice(2)}`;

/* ---------------------------- vibration support --------------------------- */

function detectSupport() {
  const ua = navigator.userAgent;
  const data = navigator.userAgentData;
  const ios = /iPad|iPhone|iPod/.test(ua) ||
    (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1);
  const android = /Android/.test(ua);
  const mobile = data ? data.mobile : android;
  const api = typeof navigator.vibrate === "function";
  if (!api) return { state: "none", ios, android, mobile };
  // Desktop Chrome exposes the API with no motor behind it, and it returns true.
  return { state: android || mobile ? "available" : "unconfirmed", ios, android, mobile };
}

const support = detectSupport();

function showSupport() {
  const el = $("support");
  const text = {
    none: support.ios
      ? "<strong>This iPhone or iPad cannot vibrate from a web page.</strong> Every iOS browser runs on WebKit, which has not implemented the Vibration API. You can still see the waveform and spectrum. To feel the pattern, use Chrome on Android."
      : "<strong>This browser has no vibration support.</strong> You can still see the waveform and spectrum. To feel the pattern, use Chrome on Android.",
    unconfirmed: "<strong>Vibration may do nothing here.</strong> This browser has the vibration function but this looks like a desktop, which usually has no motor, and the browser cannot tell. Use an Android phone to feel the pattern.",
    available: "",
  }[support.state];
  el.innerHTML = text;
  el.classList.toggle("warn", Boolean(text));
}

/* ------------------------------- session log ------------------------------ */

function osFromUserAgent() {
  const ua = navigator.userAgent;
  const android = ua.match(/Android\s([\d.]+)/);
  if (android) return { os: "Android", os_version: android[1] };
  const ios = ua.match(/OS (\d+[_\d]*) like Mac/);
  if (ios) return { os: "iOS", os_version: ios[1].replaceAll("_", ".") };
  return { os: "unknown", os_version: "unknown" };
}

async function logSession() {
  const fallback = osFromUserAgent();
  const row = {
    session_id: sessionId, user_agent: navigator.userAgent,
    device_model: "unknown", os: fallback.os, os_version: fallback.os_version,
    browser: "unknown", mobile: support.mobile, vibration_support: support.state,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
  };
  // Chrome freezes the model in its user agent string, so the real one comes only
  // from this call, which needs a secure context (HTTPS or localhost).
  const data = navigator.userAgentData;
  if (data && data.getHighEntropyValues) {
    try {
      const h = await data.getHighEntropyValues(["model", "platform", "platformVersion", "fullVersionList"]);
      row.device_model = h.model || "not reported";
      row.os = h.platform || row.os;
      row.os_version = h.platformVersion || row.os_version;
      row.browser = (h.fullVersionList || []).map((b) => `${b.brand} ${b.version}`).join("; ");
    } catch (e) { /* keep the fallback values */ }
  }
  send("/api/session", row);
}

function send(url, body) {
  return fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body), keepalive: true,
  }).catch(() => {});
}

/* ---------------------------------- plots --------------------------------- */

function cssVar(name) { return `var(${name})`; }

function plot(el, opts) {
  const W = 520, H = 190, L = 46, R = 10, T = 8, B = 30;
  const xs = opts.series.flatMap((s) => s.x);
  const ys = opts.series.flatMap((s) => s.y);
  const xMin = opts.xMin ?? Math.min(...xs), xMax = opts.xMax ?? Math.max(...xs);
  const yMin = opts.yMin ?? Math.min(...ys), yMax = opts.yMax ?? Math.max(...ys);
  const fx = opts.logX ? (v) => Math.log10(v) : (v) => v;
  const px = (v) => L + (fx(v) - fx(xMin)) / (fx(xMax) - fx(xMin)) * (W - L - R);
  const py = (v) => H - B - (v - yMin) / ((yMax - yMin) || 1) * (H - T - B);

  let g = "";
  if (opts.band) {
    const [a, b] = opts.band;
    g += `<rect x="${px(a)}" y="${T}" width="${px(b) - px(a)}" height="${H - T - B}" fill="${cssVar("--band")}"/>`;
  }
  for (const t of opts.xTicks || []) {
    g += `<line class="grid" x1="${px(t)}" x2="${px(t)}" y1="${T}" y2="${H - B}"/>` +
         `<text x="${px(t)}" y="${H - B + 14}" text-anchor="middle">${t}</text>`;
  }
  for (const t of [yMin, (yMin + yMax) / 2, yMax]) {
    g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${py(t)}" y2="${py(t)}"/>` +
         `<text x="${L - 5}" y="${py(t) + 4}" text-anchor="end">${fmt(t)}</text>`;
  }
  g += `<text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle">${opts.xLabel}</text>`;
  for (const s of opts.series) {
    const d = s.x.map((x, i) => `${i ? "L" : "M"}${px(x).toFixed(1)} ${py(s.y[i]).toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
  }
  const legend = opts.series.length > 1
    ? `<div class="legend">${opts.series.map((s) => `<span style="--c:${s.color}">${s.label}</span>`).join("")}</div>` : "";
  el.innerHTML = `<figcaption>${opts.title} <span class="sub">${opts.sub || ""}</span></figcaption>${legend}` +
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.title}">${g}` +
    `<line class="cursor" y1="${T}" y2="${H - B}" stroke="${cssVar("--muted")}" visibility="hidden"/></svg>` +
    `<div class="readout" aria-hidden="true"></div>`;

  // Hover readout: nearest sample of each series under the pointer.
  const svg = el.querySelector("svg"), cursor = el.querySelector(".cursor"), out = el.querySelector(".readout");
  svg.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const sx = (ev.clientX - r.left) / r.width * W;
    const wanted = fx(xMin) + (sx - L) / (W - L - R) * (fx(xMax) - fx(xMin));
    const parts = opts.series.map((s) => {
      let best = 0;
      s.x.forEach((x, i) => { if (Math.abs(fx(x) - wanted) < Math.abs(fx(s.x[best]) - wanted)) best = i; });
      return { x: s.x[best], text: `${s.label}: ${fmt(s.y[best])}` };
    });
    cursor.setAttribute("x1", px(parts[0].x)); cursor.setAttribute("x2", px(parts[0].x));
    cursor.setAttribute("visibility", "visible");
    out.textContent = `${fmt(parts[0].x)} ${opts.xUnit}  ·  ${parts.map((p) => p.text).join("  ·  ")}`;
  });
  svg.addEventListener("pointerleave", () => { cursor.setAttribute("visibility", "hidden"); out.textContent = ""; });
}

function fmt(v) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

const linspace = (n, end) => Array.from({ length: n }, (_, i) => i * end / n);

function drawPlots() {
  const ready = slots.filter((s) => s.data && !s.root.hidden);
  if (!ready.length) return;
  // In comparison the two fabrics share scales, or their sizes could not be compared.
  const peak = Math.max(...ready.flatMap((s) => s.data.waveform.y.map(Math.abs)));
  const dbs = ready.flatMap((s) => s.data.spectrum.db);
  const dbMin = Math.floor(Math.min(...dbs)), dbMax = Math.ceil(Math.max(...dbs));
  const scheme = $("scheme").value;

  for (const s of ready) {
    const d = s.data, q = (sel) => s.root.querySelector(sel);
    const w = d.waveform;
    plot(q(".waveform"), {
      title: "Waveform", sub: `first ${w.duration_ms} ms of acceleration, synthetic`,
      series: [{ x: linspace(w.y.length, w.duration_ms), y: w.y, label: "acceleration", color: cssVar("--accent") }],
      xLabel: "Time (ms)", xUnit: "ms", yMin: -peak, yMax: peak, xTicks: [0, 50, 100, 150, 200],
    });
    plot(q(".spectrum"), {
      title: "FFT spectrum", sub: `shaded: the most vibrate() can switch, under ${d.limits.switching_hz} Hz`,
      series: [{ x: d.spectrum.f_hz, y: d.spectrum.db, label: "power", color: cssVar("--accent") }],
      xLabel: "Frequency (Hz, log scale)", xUnit: "Hz", logX: true, xMin: 5, xMax: 1500,
      yMin: dbMin, yMax: dbMax, band: [5, d.limits.switching_hz], xTicks: [10, 100, 1000],
    });
    const env = d.envelope, p = d.patterns[scheme];
    const peakEnv = Math.max(...env.y);
    plot(q(".envelope"), {
      title: "Envelope", sub: `intended vs modelled motor response, scheme ${scheme}`,
      series: [
        { x: linspace(env.y.length, env.duration_ms), y: env.y.map((v) => v / peakEnv), label: "intended", color: cssVar("--accent") },
        { x: linspace(p.motor.length, p.total_ms), y: p.motor, label: "motor (assumed lag)", color: cssVar("--series-2") },
      ],
      xLabel: "Time (ms)", xUnit: "ms", yMin: 0, yMax: 1, xTicks: [0, 400, 800, 1200, 1600],
    });
    renderFidelity(s, scheme);
  }
}

function renderFidelity(s, scheme) {
  const d = s.data, body = s.root.querySelector(".fidelity tbody");
  body.textContent = "";
  for (const key of "ABCD") {
    const f = d.fidelity[key], tr = body.insertRow();
    tr.classList.toggle("active", key === scheme);
    tr.insertCell().textContent = `${key}: ${SCHEME_LABEL[key]}`;
    const cells = f ? [f.log_spectral_db.toFixed(1), f.wasserstein_hz.toFixed(2), f.level_error.toFixed(3)] : ["n/a", "n/a", "n/a"];
    for (const c of cells) { const td = tr.insertCell(); td.className = "num"; td.textContent = c; }
  }
  const any = Object.values(d.fidelity).find(Boolean);
  const share = any ? Math.round(any.in_band_fraction * 100) : 0;
  s.root.querySelector(".renderable").textContent = any && any.renderable
    ? `${share}% of the intended modulation is below ${d.limits.switching_hz} Hz and can be played.`
    : `Only ${share}% of the intended modulation is below ${d.limits.switching_hz} Hz, so most of this fabric's rhythm cannot be played and these scores describe what is left.`;
}

/* -------------------------------- playback -------------------------------- */

function play(slot) {
  const q = (sel) => slot.root.querySelector(sel);
  const scheme = $("scheme").value;
  const pattern = slot.data.patterns[scheme];
  const t = slot.times;
  t.play = performance.now();
  const returned = navigator.vibrate(pattern.durations);
  t.called = performance.now();
  q(".play-status").textContent = returned
    ? `Sent ${pattern.total_ms} ms pattern, scheme ${scheme}. The page cannot confirm the motor ran.`
    : "The browser refused to vibrate. Tap the page once, then try again.";
  if (t.select !== undefined) {         // only the first play follows a selection
    const toReady = t.ready - t.select, call = t.called - t.play;
    const machine = toReady + call;
    send("/api/latency", {
      session_id: sessionId, slot: slot.name, scheme, predicted_class: slot.data.predicted.class,
      autoplay: t.auto === true, select_to_ready_ms: round(toReady),
      request_ms: round(t.requestDone - t.select), render_ms: round(t.ready - t.requestDone),
      server_ms: round(slot.data.timings_ms.classify + slot.data.timings_ms.encode),
      ready_to_play_ms: round(t.play - t.ready), vibrate_call_ms: round(call),
      machine_ms: round(machine), end_to_end_ms: round(t.called - t.select),
      vibrate_returned: returned,
    });
    q(".latency").textContent = `Latency: ${round(machine)} ms from selection to playback start, ` +
      `of which server ${round(slot.data.timings_ms.classify + slot.data.timings_ms.encode)} ms.`;
    t.select = undefined;
  }
}

const round = (v) => Math.round(v * 10) / 10;

function canPlay() { return support.state !== "none"; }

/* --------------------------------- slots ---------------------------------- */

async function analyse(slot, blob, previewUrl, select) {
  const q = (sel) => slot.root.querySelector(sel);
  const err = q(".error");
  const token = (slot.token = (slot.token || 0) + 1);
  err.hidden = true;
  const form = new FormData();
  form.append("image", blob, "image");
  let res, data, requestDone;
  try {
    res = await fetch("/api/analyse", { method: "POST", body: form });
    data = await res.json();
    requestDone = performance.now();
  } catch (e) {
    err.textContent = "Could not reach the server."; err.hidden = false; return;
  }
  if (token !== slot.token) return;     // a newer selection has replaced this one
  if (!res.ok) { err.textContent = data.detail || "Analysis failed."; err.hidden = false; return; }

  slot.data = data;
  slot.times = { select, requestDone };
  q(".preview").src = previewUrl;
  q(".predicted").textContent = data.predicted.class;
  q(".top3").textContent = data.predicted.top3
    .map((c) => `${c.class} ${Math.round(c.probability * 100)}%`).join(", ");
  q(".fabric").textContent = `Retrieved: ${data.fabric.name} (fictional), ${data.fabric.velocity_mm_s} mm/s`;
  const rows = [
    ["Frequency (centroid)", `${data.descriptor.frequency_hz.toFixed(1)} Hz`],
    ["Intensity (RMS)", data.descriptor.intensity.toFixed(3)],
    ["Pattern (entropy)", data.descriptor.pattern.toFixed(3)],
    ["Duration (envelope period)", `${data.descriptor.duration_ms.toFixed(1)} ms`],
  ];
  const body = q(".descriptor tbody");
  body.textContent = "";
  for (const [k, v] of rows) {
    const tr = body.insertRow();
    tr.insertCell().textContent = k;
    const td = tr.insertCell(); td.className = "num"; td.textContent = v;
  }
  q(".latency").textContent = "";
  q(".play-status").textContent = canPlay() ? "" : "Vibration is not available here: waveform and spectrum only.";
  q(".play").hidden = !canPlay();
  q(".result").hidden = false;
  drawPlots();
  updateBoth();

  // Wait for the plots to paint, but not forever: an unpainted page never fires this.
  await Promise.race([new Promise(requestAnimationFrame),
                      new Promise((r) => setTimeout(r, PAINT_WAIT_MS))]);
  if (token !== slot.token) return;
  slot.times.ready = performance.now();
  if (canPlay() && $("autoplay").checked && !$("compare").checked) {
    slot.times.auto = true;
    play(slot);
  } else if (!canPlay()) {
    send("/api/latency", {
      session_id: sessionId, slot: slot.name, predicted_class: data.predicted.class,
      select_to_ready_ms: round(slot.times.ready - select), vibrate_returned: null,
    });
    q(".latency").textContent = `Latency: ${round(slot.times.ready - select)} ms from selection to plots shown. No playback on this device.`;
  }
}

function buildSlot(name) {
  const root = document.importNode($("slot-template").content, true).firstElementChild;
  const slot = { name, root, data: null, times: {} };
  root.querySelector(".slot-title").textContent = `Fabric ${name}`;
  root.querySelector(".file-input").addEventListener("change", (ev) => {
    const select = performance.now();
    const file = ev.target.files[0];
    if (file) analyse(slot, file, URL.createObjectURL(file), select);
  });
  const row = root.querySelector(".samples");
  for (const s of SAMPLES) {
    const b = document.createElement("button");
    const url = `/static/samples/${s}.png`;
    const img = document.createElement("img");
    img.src = url; img.alt = "";
    b.append(img, s);
    b.addEventListener("click", async () => {
      const select = performance.now();     // before the fetch: it is part of the wait
      const blob = await (await fetch(url)).blob();
      analyse(slot, blob, url, select);
    });
    row.append(b);
  }
  root.querySelector(".play").addEventListener("click", () => play(slot));
  $("slots").append(root);
  slots.push(slot);
  return slot;
}

function updateBoth() {
  const compare = $("compare").checked;
  const both = $("play-both");
  both.hidden = !(compare && canPlay());
  both.disabled = !slots.every((s) => s.data);
}

function setCompare() {
  const on = $("compare").checked;
  slots[1].root.hidden = !on;
  $("slots").classList.toggle("two", on);
  slots[0].root.querySelector(".slot-title").hidden = !on;
  updateBoth();
  drawPlots();
}

async function playBoth() {
  play(slots[0]);
  const scheme = $("scheme").value;
  const ms = slots[0].data.patterns[scheme].total_ms + PAUSE_BETWEEN_MS;
  await new Promise((r) => setTimeout(r, ms));
  play(slots[1]);
}

function init() {
  showSupport();
  logSession();
  buildSlot("A"); buildSlot("B");
  $("compare").addEventListener("change", setCompare);
  $("scheme").addEventListener("change", drawPlots);
  $("play-both").addEventListener("click", playBoth);
  if (!canPlay()) $("autoplay").closest("label").hidden = true;
  setCompare();
}

init();
