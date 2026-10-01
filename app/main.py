"""FastAPI backend: streams agent progress and serves search history."""
import json
import os
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from . import db  # noqa: E402
from .agent import run_agent  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

RATE_LIMIT = int(os.getenv("RATE_LIMIT_SEARCHES", "6"))             # new searches per IP per window
RATE_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "600"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_SEARCHES", "3"))     # agent runs at the same time
RESULT_CACHE_TTL = int(os.getenv("RESULT_CACHE_TTL", "21600"))       # reuse an identical finished search for 6 h

app = FastAPI(title="AI Shopping Agent")
db.init_db()

_slots = threading.BoundedSemaphore(MAX_CONCURRENT)
_recent: dict[str, deque] = defaultdict(deque)
_recent_lock = threading.Lock()


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    refresh: bool = False  # skip the cache and search again


@app.middleware("http")
async def anonymous_client_id(request: Request, call_next):
    """Give each browser a random id so people only see their own search history."""
    sid = request.cookies.get("sid", "")
    fresh = not re.fullmatch(r"[0-9a-f]{32}", sid)
    if fresh:
        sid = uuid.uuid4().hex
    request.state.sid = sid
    response = await call_next(request)
    if fresh:
        response.set_cookie("sid", sid, max_age=365 * 24 * 3600, httponly=True, samesite="lax")
    return response


def _rate_limited(ip: str) -> int:
    """0 if the request may proceed, else seconds until the client may search again."""
    now = time.time()
    with _recent_lock:
        hits = _recent[ip]
        while hits and now - hits[0] > RATE_WINDOW:
            hits.popleft()
        if len(hits) >= RATE_LIMIT:
            return max(1, int(RATE_WINDOW - (now - hits[0])))
        hits.append(now)
    return 0


def _replay(item: dict):
    """Stream a stored search back as if it had just run (instant, no LLM or web calls)."""
    yield json.dumps({"type": "start", "search_id": item["id"], "cached": True}) + "\n"
    if item.get("requirements"):
        yield json.dumps({"type": "requirements", "requirements": item["requirements"]}) + "\n"
    yield json.dumps({
        "type": "final", "cached": True, "cached_at": item["finished_at"],
        "result": {"summary": item["summary"], "recommendations": item["recommendations"]},
        "comparison": item["comparison"],
    }, default=str) + "\n"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/search")
def search(req: SearchRequest, request: Request):
    """Run the agent and stream events as newline-delimited JSON."""
    query = " ".join(req.query.split())
    if not re.search(r"\w{2,}", query):
        raise HTTPException(422, "Describe what you want to buy, e.g. 'black cotton t-shirt, size L'.")
    sid = request.state.sid

    if not req.refresh:
        cached_id = db.find_recent_done(query, sid, RESULT_CACHE_TTL)
        if cached_id and (item := db.get_search(cached_id, sid)):
            return StreamingResponse(_replay(item), media_type="application/x-ndjson")

    ip = request.client.host if request.client else "unknown"
    if wait := _rate_limited(ip):
        return JSONResponse({"detail": f"Too many searches. Try again in {wait // 60 + 1} min."}, status_code=429,
                            headers={"Retry-After": str(wait)})
    if not _slots.acquire(blocking=False):
        return JSONResponse({"detail": "The agent is busy with other searches. Try again in a minute."},
                            status_code=503, headers={"Retry-After": "30"})

    search_id = db.create_search(query, sid)

    def stream():
        try:
            yield json.dumps({"type": "start", "search_id": search_id}) + "\n"
            seq = 0
            try:
                for event in run_agent(query):
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
        finally:
            _slots.release()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/api/history")
def history(request: Request, limit: int = 50):
    return db.list_searches(max(1, min(limit, 200)), request.state.sid)


@app.get("/api/history/{search_id}")
def history_detail(search_id: int, request: Request):
    item = db.get_search(search_id, request.state.sid)
    if not item:
        raise HTTPException(404, "Search not found")
    return item


@app.delete("/api/history/{search_id}")
def history_delete(search_id: int, request: Request):
    if not db.delete_search(search_id, request.state.sid):
        raise HTTPException(404, "Search not found")
    return {"deleted": search_id}
