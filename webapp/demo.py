"""One-step demo of the Objective 4 interface.

    python -m webapp.demo

Starts the web app so that both this laptop and an Android phone on the same
Wi-Fi can open it, prints the two addresses, and opens the page in the laptop's
browser. Stop it with Ctrl+C.

Needs the trained texture model at models/dtd_fold1_seed0/best.pt (about 95 MB,
not in Git: copy the models/ folder from the laptop that trained it).
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import webbrowser

from . import pipeline as P

PORT = int(os.environ.get("O4_PORT", "8765"))


def lan_ip() -> str | None:
    """The address other devices on the local network can reach. Opening a UDP
    socket towards a public address sends nothing; it only picks the interface
    the system would route through."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


def check_model() -> bool:
    if P.DEFAULT_CHECKPOINT.is_file():
        return True
    print(f"Model not found: {P.DEFAULT_CHECKPOINT}\n"
          "Copy the models/ folder (from the laptop that trained the classifiers) into the project folder, "
          "then run this again.", file=sys.stderr)
    return False


def main() -> int:
    if not check_model():
        return 1
    import uvicorn

    from .app import create_app
    ip = lan_ip()
    print("\nFeel this fabric: demo")
    print(f"  On this laptop:          http://127.0.0.1:{PORT}")
    if ip:
        print(f"  On an Android phone:     http://{ip}:{PORT}   (same Wi-Fi; open in Chrome)")
    else:
        print("  No network address found: connect to Wi-Fi to use a phone.")
    print("  If Windows asks about the firewall, allow access on private networks.")
    print("  Press Ctrl+C to stop.\n", flush=True)
    if not os.environ.get("O4_NO_BROWSER"):
        threading.Timer(2.5, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    uvicorn.run(create_app(), host="0.0.0.0", port=PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
