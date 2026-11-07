"""Download the Cluster cloth subset (IDs 102-117) from the Hugging Face mirror.

    python fetch_cluster.py --out data/cluster            # accel, position, images
    python fetch_cluster.py --out data/cluster --with force audio
    python fetch_cluster.py --out data/cluster --all-materials --directions 0 90         --repeats 0 --no-images                      # the velocity subset, all 118

Figshare only offers the whole dataset as one 15 GB zip. The mirror holds the
same recordings one file each, so this fetches just the cloth materials. It is
pinned to one mirror revision, so a re-run gets identical files.

Sizes for the 16 cloth materials, from the mirror's listing on 2026-11-03:

    accel 555 MB · position 14 MB · scan images 4 MB · crop images 14 MB
    force 322 MB · audio 197 MB · raw_audio 352 MB      (optional)

Licence: Apache 2.0, on both figshare and the mirror. Data stays out of git: put
it under data/, which is ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from haptic.loader import CLOTH_IDS

REPO = "tamago117/cluster-haptic-texture-dataset"
REVISION = "b7c2fb70ed2d68219389478660f35c2cd49c69fb"
DEFAULT_SIGNALS = ("accel", "position")
OPTIONAL_SIGNALS = ("force", "audio", "raw_audio")
RATE_WAIT_S = 310
CROPS_PER_MATERIAL = 100
NET_RETRIES = 8
NET_WAIT_S = 15


def _transient(e) -> bool:
    """5xx and connection-level HTTP errors are worth retrying; 404 and the like are not."""
    r = getattr(e, "response", None)
    return r is None or r.status_code >= 500 or r.status_code in (408, 409)


def patterns(signals=DEFAULT_SIGNALS, ids=CLOTH_IDS, images: bool = True) -> list[str]:
    out = [f"sensor_data/{s}/{mid}/*" for s in signals for mid in ids]
    if images:
        out += [f"images/images_scan_area/img{mid}.*" for mid in ids]
        out += [f"images/crop_images/img{mid}/*" for mid in ids]
    return out + ["README.md"]


def expected_files(signals=DEFAULT_SIGNALS, ids=CLOTH_IDS, images: bool = True,
                   directions=None, repeats=None) -> list[str]:
    """Every file the selection should hold, built from the documented naming.
    Listing the mirror instead costs one request per page of its 106,000 files,
    which alone can exhaust the anonymous rate limit. `directions` and `repeats`
    narrow the selection; None means all."""
    from haptic.loader import DIRECTIONS_DEG, FORCES_N, REPEATS, VELOCITIES_MM_S
    # audio and raw_audio are FLAC on the mirror; every other signal is Parquet
    out = [f"sensor_data/{s}/{mid}/{mid}_{d}_{int(v)}_{int(fo * 1000)}_{r}"
           f"{'.flac' if s in ('audio', 'raw_audio') else '.parquet'}"
           for s in signals for mid in ids for d in (directions or DIRECTIONS_DEG)
           for v in VELOCITIES_MM_S for fo in FORCES_N for r in (repeats or REPEATS)]
    if images:
        out += [f"images/images_scan_area/img{mid}.jpg" for mid in ids]
        out += [f"images/crop_images/img{mid}/img{mid}_{k}.jpg"
                for mid in ids for k in range(1, CROPS_PER_MATERIAL + 1)]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fetch_cluster")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with", dest="extra", nargs="*", default=[], choices=OPTIONAL_SIGNALS)
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--all-materials", action="store_true",
                    help="all 118 materials, not only cloth 102-117")
    ap.add_argument("--directions", type=int, nargs="*")
    ap.add_argument("--repeats", type=int, nargs="*")
    ap.add_argument("--threads", type=int, default=1,
                    help="concurrent downloads; 4 is about 4x faster under the rate limit")
    a = ap.parse_args(argv)
    ids = tuple(range(118)) if a.all_materials else CLOTH_IDS

    import time

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    out = Path(a.out)
    allow = patterns((*DEFAULT_SIGNALS, *a.extra), ids=ids, images=not a.no_images)
    files = expected_files((*DEFAULT_SIGNALS, *a.extra), ids=ids, images=not a.no_images,
                           directions=a.directions, repeats=a.repeats)
    todo = [f for f in files if not (out / f).exists()]
    print(f"{len(files)} files selected, {len(files) - len(todo)} already here", flush=True)

    # Anonymous access allows about 1000 requests per 5 minutes. snapshot_download
    # hides a 429 by returning the partial local copy, so files are fetched one at
    # a time here, and a 429 waits out the window and retries the same file.
    # A dropped connection (seen 2026-11-07 on the CDN) is retried after a short
    # pause, up to NET_RETRIES times in a row for the same file.
    # With --threads > 1 files are fetched concurrently; the 429 budget is shared.
    import threading
    lock = threading.Lock()
    state = {"waits": 0, "done": 0}

    def fetch_one(k_f):
        k, f = k_f
        net_fails = 0
        while True:
            try:
                hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=out)
                break
            except HfHubHTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    with lock:
                        if state["waits"] >= a.retries:
                            raise
                        state["waits"] += 1
                        w = state["waits"]
                    print(f"rate limited at {k}/{len(todo)}; waiting {RATE_WAIT_S} s "
                          f"({w}/{a.retries})", flush=True)
                    time.sleep(RATE_WAIT_S)
                    continue
                if not _transient(e) or net_fails >= NET_RETRIES:
                    raise
                net_fails += 1
                print(f"HTTP error at {k}/{len(todo)} ({e}); retry {net_fails}", flush=True)
                time.sleep(NET_WAIT_S * net_fails)
            except (OSError, TimeoutError, RuntimeError) as e:   # incl. ConnectionError, xet errors
                if net_fails >= NET_RETRIES:
                    raise
                net_fails += 1
                print(f"network error at {k}/{len(todo)} ({type(e).__name__}); "
                      f"retry {net_fails}", flush=True)
                time.sleep(NET_WAIT_S * net_fails)
        with lock:
            state["done"] += 1
            if state["done"] % 500 == 0:
                print(f"  {state['done']}/{len(todo)}", flush=True)

    jobs = list(enumerate(todo, 1))
    if a.threads <= 1:
        for job in jobs:
            fetch_one(job)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(a.threads) as ex:
            list(ex.map(fetch_one, jobs))     # list() re-raises the first failure

    missing = [f for f in files if not (out / f).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} files still missing, e.g. {missing[:3]}; re-run")
    record = out / ("FETCHED_all.json" if a.all_materials else "FETCHED.json")
    record.write_text(json.dumps(
        {"repo": REPO, "revision": REVISION, "patterns": allow, "n_files": len(files),
         "directions": a.directions, "repeats": a.repeats}, indent=2))
    print(f"all {len(files)} files in {out}\nnext: python -m haptic.cli inspect --root {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
