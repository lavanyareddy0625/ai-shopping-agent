"""FastAPI backend: streams agent progress and serves search history."""
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import db  # noqa: E402
from .agent import run_agent  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="AI Shopping Agent")
db.init_db()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/search")
def search(req: SearchRequest):
    """Run the agent and stream events as newline-delimited JSON."""
    search_id = db.create_search(req.query.strip())

    def stream():
        yield json.dumps({"type": "start", "search_id": search_id}) + "\n"
        seq = 0
        try:
            for event in run_agent(req.query.strip()):
                seq += 1
                db.add_step(search_id, seq, event)
                if event["type"] == "meta":
                    db.update_search(search_id, provider=event["provider"], model=event["model"])
                elif event["type"] == "model_switch":
                    db.update_search(search_id, model=event["to"])
                elif event["type"] == "requirements":
                    db.update_search(search_id, requirements=event["requirements"])
                elif event["type"] == "final":
                    db.finish_search(search_id, event["result"], event.get("comparison"))
                yield json.dumps(event, default=str) + "\n"
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            db.fail_search(search_id, msg)
            yield json.dumps({"type": "error", "message": msg}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/history")
def history(limit: int = 50):
    return db.list_searches(limit)


@app.get("/api/history/{search_id}")
def history_detail(search_id: int):
    item = db.get_search(search_id)
    if not item:
        raise HTTPException(404, "Search not found")
    return item


@app.delete("/api/history/{search_id}")
def history_delete(search_id: int):
    if not db.delete_search(search_id):
        raise HTTPException(404, "Search not found")
    return {"deleted": search_id}
