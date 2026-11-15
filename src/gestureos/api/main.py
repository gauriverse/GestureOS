import asyncio
import threading
import json
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gestureos.controller import GestureController


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    controller = GestureController()
    thread = threading.Thread(target=controller.run,
                              kwargs={"show_window": False}, daemon=True)
    thread.start()
    app.state.controller = controller
    yield
    controller.stop()
    thread.join(timeout=3)


app = FastAPI(title="GestureOS", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/status")
def get_status():
    return asdict(app.state.controller.status)

@app.get("/metrics")
def get_metrics():
    return json.loads(Path("models/gesture_model_v2_metrics.json").read_text())


@app.websocket("/ws")
async def live_status(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json(asdict(app.state.controller.status))
            await asyncio.sleep(0.1)  
    except WebSocketDisconnect:
        pass