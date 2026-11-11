"""Screenshot the Objective 4 web page after it analyses a sample image.

    python -m webapp.app            (in another terminal; serves 127.0.0.1:8765)
    python report/capture_webapp.py

Drives headless Edge over the DevTools protocol: opens the page, clicks the first
sample image, waits for the result, and saves report/figures/webapp.png.
"""
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket  # websocket-client

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
URL = "http://127.0.0.1:8765/"
OUT = Path(__file__).resolve().parent / "figures" / "webapp.png"
PORT = 9333


def main() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as prof:
        edge = subprocess.Popen([EDGE, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}",
                                 f"--user-data-dir={prof}", "--window-size=1280,1600", "about:blank"])
        try:
            for _ in range(50):
                try:
                    tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                    page = next(t for t in tabs if t["type"] == "page")
                    break
                except Exception:
                    time.sleep(0.3)
            ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
            n = [0]

            def cdp(method, **params):
                n[0] += 1
                ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
                while True:
                    msg = json.loads(ws.recv())
                    if msg.get("id") == n[0]:
                        return msg.get("result", {})

            cdp("Emulation.setDeviceMetricsOverride", width=1280, height=1600, deviceScaleFactor=1.5, mobile=False)
            cdp("Page.enable")
            cdp("Page.navigate", url=URL)
            time.sleep(3)
            clicked = cdp("Runtime.evaluate", returnByValue=True,
                          expression="(() => { const b = document.querySelector('.samples button'); if (!b) return 'none'; b.click(); return b.textContent || b.title || 'clicked'; })()")
            print("sample:", clicked.get("result", {}).get("value"))
            for _ in range(60):
                time.sleep(0.5)
                ready = cdp("Runtime.evaluate", returnByValue=True,
                            expression="!!document.querySelector('svg') && document.body.innerText.includes('Hz')")
                if ready.get("result", {}).get("value"):
                    break
            time.sleep(1.5)
            h = cdp("Runtime.evaluate", returnByValue=True,
                    expression="Math.min(document.documentElement.scrollHeight, 2600)")["result"]["value"]
            cdp("Emulation.setDeviceMetricsOverride", width=1280, height=int(h), deviceScaleFactor=1.5, mobile=False)
            time.sleep(1)
            shot = cdp("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_bytes(base64.b64decode(shot["data"]))
            print("saved", OUT, "height", h)
            ws.close()
        finally:
            edge.terminate()
            edge.wait(timeout=20)


if __name__ == "__main__":
    main()
