"""SQLite persistence for searches, agent steps and recommendations."""
import json
import os
import sqlite3
import time
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
    finished_at  TEXT,
    client_id    TEXT                                -- anonymous browser id (cookie); NULL = legacy row
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
    source_url  TEXT,
    extra       TEXT                                 -- JSON: pack_size, unit_price, notes, unverified
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
CREATE TABLE IF NOT EXISTS cache (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,                        -- JSON
    created_at REAL NOT NULL                         -- unix time
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


def _add_column(c, table: str, column: str, ddl: str) -> None:
    if column not in {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db() -> None:
    with connect() as c:
        c.executescript(SCHEMA)
        # Databases created by earlier versions lack these columns.
        _add_column(c, "searches", "client_id", "TEXT")
        _add_column(c, "recommendations", "extra", "TEXT")


# --- tool-result cache ---------------------------------------------------------------------
def cache_get(key: str, ttl_seconds: float) -> dict | None:
    with connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)")
        row = c.execute("SELECT value, created_at FROM cache WHERE key = ?", (key,)).fetchone()
    if row and time.time() - row["created_at"] <= ttl_seconds:
        return json.loads(row["value"])
    return None


def cache_set(key: str, value: dict) -> None:
    with connect() as c:
        c.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)")
        c.execute("INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
                  (key, json.dumps(value, default=str), time.time()))


# --- searches ---------------------------------------------------------------------------------
def _visible(client_id: str | None) -> tuple[str, tuple]:
    """SQL filter: a browser sees its own searches plus legacy rows that have no owner."""
    if client_id is None:
        return "1=1", ()
    return "(client_id = ? OR client_id IS NULL)", (client_id,)


def create_search(query: str, client_id: str | None = None) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO searches (query, created_at, client_id) VALUES (?, ?, ?)",
                        (query, _now(), client_id))
        return cur.lastrowid


def find_recent_done(query: str, client_id: str | None, max_age_seconds: float) -> int | None:
    """Id of a finished, non-degraded search with the same (normalised) query run recently."""
    norm = " ".join(query.lower().split())
    where, params = _visible(client_id)
    with connect() as c:
        rows = c.execute(
            f"""SELECT id, query, finished_at FROM searches
                WHERE status = 'done' AND {where}
                  AND EXISTS (SELECT 1 FROM recommendations r WHERE r.search_id = searches.id)
                ORDER BY id DESC LIMIT 30""", params).fetchall()
    cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
    for r in rows:
        if " ".join(r["query"].lower().split()) == norm and r["finished_at"] \
                and datetime.fromisoformat(r["finished_at"]).timestamp() >= cutoff:
            return r["id"]
    return None


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
            "UPDATE searches SET status=?, summary=?, comparison=?, finished_at=? WHERE id=?",
            ("partial" if final.get("degraded") else "done", final.get("summary"),
             json.dumps(comparison) if comparison else None, _now(), search_id),
        )
        c.execute("DELETE FROM recommendations WHERE search_id = ?", (search_id,))
        for i, r in enumerate(final.get("recommendations") or [], 1):
            c.execute(
                """INSERT INTO recommendations
                   (search_id, rank, name, price, currency, key_details, reason, source_url, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (search_id, i, r.get("name"), r.get("price"), r.get("currency"),
                 json.dumps(r.get("key_details") or []), r.get("reason"), r.get("source_url"),
                 json.dumps({k: r[k] for k in EXTRA_FIELDS if k in r})),
            )


EXTRA_FIELDS = ("pack_size", "unit_price", "notes", "unverified")


def fail_search(search_id: int, error: str) -> None:
    update_search(search_id, status="error", error=error, finished_at=_now())


def list_searches(limit: int = 50, client_id: str | None = None) -> list[dict]:
    where, params = _visible(client_id)
    with connect() as c:
        rows = c.execute(
            f"""SELECT searches.id, searches.query, searches.status, searches.created_at,
                       searches.provider, searches.model,
                       (SELECT COUNT(*) FROM recommendations r WHERE r.search_id = searches.id) AS n_results
                FROM searches WHERE {where} ORDER BY searches.id DESC LIMIT ?""",
            (*params, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_search(search_id: int, client_id: str | None = None) -> dict | None:
    where, params = _visible(client_id)
    with connect() as c:
        s = c.execute(f"SELECT * FROM searches WHERE id = ? AND {where}", (search_id, *params)).fetchone()
        if not s:
            return None
        recs = c.execute("SELECT * FROM recommendations WHERE search_id = ? ORDER BY rank", (search_id,)).fetchall()
        steps = c.execute("SELECT data FROM agent_steps WHERE search_id = ? ORDER BY seq", (search_id,)).fetchall()
    out = dict(s)
    for key in ("requirements", "comparison"):
        out[key] = json.loads(out[key]) if out.get(key) else None
    out["recommendations"] = [
        {**{k: v for k, v in dict(r).items() if k != "extra"}, "key_details": json.loads(r["key_details"] or "[]"),
         **json.loads(r["extra"] or "{}")}
        for r in recs
    ]
    out["steps"] = [json.loads(r["data"]) for r in steps]
    return out


def delete_search(search_id: int, client_id: str | None = None) -> bool:
    where, params = _visible(client_id)
    with connect() as c:
        return c.execute(f"DELETE FROM searches WHERE id = ? AND {where}", (search_id, *params)).rowcount > 0
