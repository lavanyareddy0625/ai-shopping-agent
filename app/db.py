"""SQLite persistence for searches, agent steps and recommendations."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "shopping.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS searches (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'running',   -- running | done | error
    provider     TEXT,
    model        TEXT,
    requirements TEXT,                               -- JSON
    summary      TEXT,
    comparison   TEXT,                               -- JSON from compare_products
    error        TEXT,
    created_at   TEXT NOT NULL,
    finished_at  TEXT
);
CREATE TABLE IF NOT EXISTS recommendations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id   INTEGER NOT NULL REFERENCES searches(id) ON DELETE CASCADE,
    rank        INTEGER,
    name        TEXT,
    price       REAL,
    currency    TEXT,
    key_details TEXT,                                -- JSON list
    reason      TEXT,
    source_url  TEXT
);
CREATE TABLE IF NOT EXISTS agent_steps (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id INTEGER NOT NULL REFERENCES searches(id) ON DELETE CASCADE,
    seq       INTEGER,
    type      TEXT,
    name      TEXT,
    data      TEXT,                                  -- JSON
    created_at TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)


def create_search(query: str) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO searches (query, created_at) VALUES (?, ?)", (query, _now()))
        return cur.lastrowid


def add_step(search_id: int, seq: int, event: dict) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO agent_steps (search_id, seq, type, name, data, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (search_id, seq, event.get("type"), event.get("name"), json.dumps(event, default=str), _now()),
        )


def update_search(search_id: int, **fields) -> None:
    for key in ("requirements", "comparison"):
        if key in fields and not isinstance(fields[key], (str, type(None))):
            fields[key] = json.dumps(fields[key], default=str)
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE searches SET {cols} WHERE id = ?", (*fields.values(), search_id))


def finish_search(search_id: int, final: dict, comparison: dict | None) -> None:
    with connect() as c:
        c.execute(
            "UPDATE searches SET status='done', summary=?, comparison=?, finished_at=? WHERE id=?",
            (final.get("summary"), json.dumps(comparison) if comparison else None, _now(), search_id),
        )
        c.execute("DELETE FROM recommendations WHERE search_id = ?", (search_id,))
        for i, r in enumerate(final.get("recommendations") or [], 1):
            c.execute(
                """INSERT INTO recommendations
                   (search_id, rank, name, price, currency, key_details, reason, source_url)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (search_id, i, r.get("name"), r.get("price"), r.get("currency"),
                 json.dumps(r.get("key_details") or []), r.get("reason"), r.get("source_url")),
            )


def fail_search(search_id: int, error: str) -> None:
    update_search(search_id, status="error", error=error, finished_at=_now())


def list_searches(limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(
            """SELECT s.id, s.query, s.status, s.created_at, s.provider, s.model,
                      (SELECT COUNT(*) FROM recommendations r WHERE r.search_id = s.id) AS n_results
               FROM searches s ORDER BY s.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_search(search_id: int) -> dict | None:
    with connect() as c:
        s = c.execute("SELECT * FROM searches WHERE id = ?", (search_id,)).fetchone()
        if not s:
            return None
        recs = c.execute("SELECT * FROM recommendations WHERE search_id = ? ORDER BY rank", (search_id,)).fetchall()
        steps = c.execute("SELECT data FROM agent_steps WHERE search_id = ? ORDER BY seq", (search_id,)).fetchall()
    out = dict(s)
    for key in ("requirements", "comparison"):
        out[key] = json.loads(out[key]) if out.get(key) else None
    out["recommendations"] = [{**dict(r), "key_details": json.loads(r["key_details"] or "[]")} for r in recs]
    out["steps"] = [json.loads(r["data"]) for r in steps]
    return out


def delete_search(search_id: int) -> bool:
    with connect() as c:
        return c.execute("DELETE FROM searches WHERE id = ?", (search_id,)).rowcount > 0
