"""Objective 4 web service.

    python -m webapp.app                    http://127.0.0.1:8765

Device model is only readable from a secure context, so a phone must reach this
over HTTPS, or over `adb reverse tcp:8765 tcp:8765` and http://localhost:8765.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

from . import pipeline as P
from .sessions import SessionLog

STATIC = Path(__file__).parent / "static"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
DEFAULT_LOG_DIR = P.ROOT / "logs" / "o4"


def create_app(classify: P.Classifier | None = None,
               log_dir: str | Path = DEFAULT_LOG_DIR) -> FastAPI:
    app = FastAPI(title="Tactile virtualization: Objective 4")
    library, cfg = P.build_library()
    classify = classify or P.load_classifier()
    log = SessionLog(log_dir)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.post("/api/analyse")
    async def analyse(image: UploadFile):
        body = await image.read()
        if not body:
            raise HTTPException(400, "empty file")
        if len(body) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "image is larger than 10 MB")
        try:
            return P.analyse(body, classify, library, cfg)
        except (UnidentifiedImageError, OSError) as e:
            raise HTTPException(400, "not a readable image") from e

    async def _record(request: Request, write):
        try:
            return {"logged": write(await request.json())}
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/api/session")
    async def session(request: Request):
        return await _record(request, log.session)

    @app.post("/api/latency")
    async def latency(request: Request):
        return await _record(request, log.latency)

    @app.get("/api/latency")
    def latency_summary():
        return log.latency_summary()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main() -> None:
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("O4_HOST", "127.0.0.1"),
                port=int(os.environ.get("O4_PORT", "8765")))


if __name__ == "__main__":
    main()
