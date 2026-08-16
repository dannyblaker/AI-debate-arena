"""FastAPI app: REST API + WebSocket live stream + static frontend."""
from __future__ import annotations

import asyncio
import copy
import threading
from urllib.parse import quote

from fastapi import (FastAPI, HTTPException, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import materials, models_registry, pdf_export
from .config import FAKE_LLM, FRONTEND_DIR, MODELS_DIR
from .debate import DebateConfig, run_pipeline
from .judging import CRITERIA, JUDGES

app = FastAPI(title="AI Debate Arena")


def _initial_state() -> dict:
    return {
        "phase": "idle",
        "topic": None,
        "model": None,
        "config": None,
        "download": None,
        "sources": [],
        "num_chunks": 0,
        "semantic": False,
        "transcript": [],
        "current": None,
        "judges": [{"id": j["id"], "name": j["name"], "status": "waiting",
                    "partial": {}, "ballot": None} for j in JUDGES],
        "verdict": None,
        "error": None,
        "log": [],
    }


class DebateManager:
    def __init__(self):
        self.state = _initial_state()
        self.lock = threading.Lock()
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def emit(self, type: str, **data):
        """Called from the worker thread: update state, broadcast to clients."""
        event = {"type": type, **data}
        with self.lock:
            self._apply(event)
        if self.loop and not self.loop.is_closed():
            asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    def _apply(self, ev: dict):
        s = self.state
        t = ev["type"]
        if t == "phase":
            s["phase"] = ev["phase"]
            s["log"].append(ev["message"])
        elif t == "status":
            s["log"].append(ev["message"])
        elif t == "download_progress":
            s["download"] = {k: ev[k] for k in ("filename", "done", "total", "pct")}
        elif t == "research_source":
            s["sources"].append({"title": ev["title"], "url": ev["url"],
                                 "chars": ev["chars"],
                                 "kind": ev.get("kind", "web")})
        elif t == "research_done":
            s["num_chunks"] = ev["num_chunks"]
            s["semantic"] = ev.get("semantic", False)
        elif t == "turn_start":
            s["current"] = {k: ev[k] for k in ("speaker", "phase", "round", "label")}
            s["current"]["text"] = ""
        elif t == "token":
            if s["current"]:
                s["current"]["text"] += ev["text"]
        elif t == "turn_end":
            s["transcript"].append({k: ev[k] for k in
                                    ("speaker", "phase", "round", "label", "text")})
            s["current"] = None
        elif t == "judge_start":
            for j in s["judges"]:
                if j["id"] == ev["judge_id"]:
                    j["status"] = "deliberating"
        elif t == "judge_criterion":
            for j in s["judges"]:
                if j["id"] == ev["judge_id"]:
                    j["partial"][ev["criterion"]] = ev["result"]
        elif t == "judge_result":
            for j in s["judges"]:
                if j["id"] == ev["judge_id"]:
                    j["status"] = "done"
                    j["ballot"] = ev["ballot"]
        elif t == "verdict":
            s["verdict"] = {k: v for k, v in ev.items() if k != "type"}
        elif t == "error":
            s["phase"] = "error"
            s["error"] = ev["message"]
            s["log"].append(ev["message"])

    async def _broadcast(self, event: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def start(self, cfg: DebateConfig, model: dict):
        self.stop_event = threading.Event()
        with self.lock:
            self.state = _initial_state()
            self.state["topic"] = cfg.topic
            self.state["model"] = {"id": model["id"], "name": model["name"],
                                   "repo": model["repo"]}
            self.state["config"] = {
                "rounds": cfg.rounds,
                "pro_personality": cfg.pro_personality,
                "con_personality": cfg.con_personality,
            }
        self.emit("status", message=f'New debate: "{cfg.topic}"')
        stop = self.stop_event
        self.thread = threading.Thread(
            target=run_pipeline, args=(cfg, self.emit, stop), daemon=True)
        self.thread.start()

    def snapshot(self) -> dict:
        with self.lock:
            return copy.deepcopy(self.state)


manager = DebateManager()


@app.on_event("startup")
async def _capture_loop():
    manager.loop = asyncio.get_running_loop()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


class StartRequest(BaseModel):
    topic: str = Field(min_length=8, max_length=500)
    model_id: str
    pro_personality: str = Field(default="", max_length=1000)
    con_personality: str = Field(default="", max_length=1000)
    rounds: int = Field(default=2, ge=1, le=4)
    use_web_research: bool = True


@app.post("/api/materials")
async def api_upload_material(file: UploadFile):
    data = await file.read()
    try:
        mat = materials.STORE.add(file.filename or "upload", data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": mat.id, "filename": mat.filename, "chars": len(mat.text)}


@app.get("/api/materials")
def api_list_materials():
    return {"materials": materials.STORE.summaries()}


@app.delete("/api/materials/{material_id}")
def api_delete_material(material_id: str):
    if not materials.STORE.remove(material_id):
        raise HTTPException(404, "No such material.")
    return {"ok": True}


@app.get("/api/models")
def api_models():
    return {"models": models_registry.list_models(), "fake_llm": FAKE_LLM}


@app.get("/api/system")
def api_system():
    return {
        "available_ram_gb": round(models_registry.available_ram_bytes() / 1024**3, 1),
        "criteria": [{k: c[k] for k in ("key", "label", "max")} for c in CRITERIA],
    }


@app.get("/api/state")
def api_state():
    return manager.snapshot()


@app.post("/api/debate/start")
def api_start(req: StartRequest):
    if manager.running:
        raise HTTPException(409, "A debate is already in progress.")
    model = next((m for m in models_registry.MODELS if m["id"] == req.model_id), None)
    if not model:
        raise HTTPException(400, f"Unknown model: {req.model_id}")
    cfg = DebateConfig(
        topic=req.topic.strip(),
        model_id=req.model_id,
        pro_personality=req.pro_personality.strip(),
        con_personality=req.con_personality.strip(),
        rounds=req.rounds,
        materials=materials.STORE.as_docs(),
        use_web_research=req.use_web_research,
    )
    manager.start(cfg, model)
    return {"ok": True}


@app.post("/api/debate/stop")
def api_stop():
    manager.stop_event.set()
    return {"ok": True}


@app.get("/api/export/pdf")
def api_export_pdf():
    state = manager.snapshot()
    if not state["transcript"]:
        raise HTTPException(400, "No transcript to export yet.")
    data = pdf_export.build_pdf(state)
    slug = quote((state.get("topic") or "debate")[:60].replace(" ", "-"), safe="-")
    return Response(
        content=data,
        media_type="application/pdf",
        headers={"Content-Disposition":
                 f'attachment; filename="debate-{slug}.pdf"'},
    )


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "snapshot", "state": manager.snapshot()})
    manager.clients.add(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive pings from the client
    except WebSocketDisconnect:
        pass
    finally:
        manager.clients.discard(ws)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
