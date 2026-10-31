"""Local server. One simulation thread, one websocket stream, one page."""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

VIEWER = Path(__file__).resolve().parent / "viewer.html"


class Command(BaseModel):
    cmd: str
    value: float | None = None
    id: int | None = None
    slot: int | None = None
    mode: str | None = None


def create_app(sim):
    @asynccontextmanager
    async def lifespan(app):
        yield
        sim.stop()

    app = FastAPI(title="Adi", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.get("/")
    def index():
        return FileResponse(VIEWER, media_type="text/html")

    @app.get("/state")
    def state():
        return JSONResponse(sim.snapshot())

    @app.post("/cmd")
    def cmd(c: Command):
        sim.command(c.model_dump(exclude_none=True))
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()

        async def receive():
            while True:
                msg = await websocket.receive_text()
                try:
                    sim.command(json.loads(msg))
                except Exception:
                    pass

        rx = asyncio.create_task(receive())
        try:
            while True:
                await websocket.send_text(
                    json.dumps(sim.snapshot(), separators=(",", ":"))
                )
                await asyncio.sleep(1.0 / sim.cfg.snap_hz)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            rx.cancel()

    return app
