"""Edge cases: currency, region, packs, size/colour, caching, quota failures, grounding, API limits."""
import json
import sqlite3
from types import SimpleNamespace

import httpx
import openai
import pytest

from app import agent, db, llm, tools
from .test_app import PRODUCT_HTML, _tool_msg


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "t.db"))
    db.init_db()
    return tmp_path / "t.db"


# ------------------------------------------------------------------------------ currency
@pytest.mark.parametrize("url,expected", [
    ("https://www.amazon.com/dp/B0", "USD"),
    ("https://www.amazon.in/dp/B0", "INR"),
    ("https://www.flipkart.com/p/x", "INR"),
    ("https://www.myntra.com/tshirts/x", "INR"),
    ("https://www.smartprix.com/mobiles/x", "INR"),
    ("https://www.amazon.co.uk/dp/B0", "GBP"),
    ("https://shop.example.org/x", None),
    ("", None),
    (None, None),
])
def test_infer_currency(url, expected):
    assert tools.infer_currency(url) == expected


def test_is_local_store_follows_region(monkeypatch):
    monkeypatch.setattr(tools, "SEARCH_REGION", "in-en")
    assert tools.is_local_store("https://www.flipkart.com/x") and tools.is_local_store("https://shop.co.in/x")
    assert not tools.is_local_store("https://www.walmart.com/x")
    monkeypatch.setattr(tools, "SEARCH_REGION", "wt-wt")  # no profile: nothing is "local"
    assert not tools.is_local_store("https://www.flipkart.com/x")


# ------------------------------------------------------------------------------ search
def _hit(url, title="t"):
    return {"title": title, "href": url, "body": "snippet"}


def test_search_prefers_local_stores_and_adds_store_query(monkeypatch, tmp_db):
    monkeypatch.setattr(tools, "SEARCH_REGION", "in-en")
    queries = []

    def fake_ddg(query, n):
        queries.append(query)
        if "site:" in query:
            return [_hit("https://www.amazon.in/a"), _hit("https://www.flipkart.com/b")], None
        return [_hit("https://www.walmart.com/w"), _hit("https://www.target.com/t")], None

    monkeypatch.setattr(tools, "_ddg", fake_ddg)
    out = tools.web_search("black t-shirt")
    urls = [r["url"] for r in out["results"]]
    assert len(queries) == 2 and "site:amazon.in" in queries[1]
    assert urls[:2] == ["https://www.amazon.in/a", "https://www.flipkart.com/b"]  # local first
    assert out["results"][0]["currency_hint"] == "INR" and out["results"][-1]["currency_hint"] == "USD"


def test_search_is_cached_and_failures_are_not(monkeypatch, tmp_db):
    monkeypatch.setattr(tools, "SEARCH_REGION", "wt-wt")
    calls = []

    def fake_ddg(query, n):
        calls.append(query)
        return ([_hit("https://a.example/x")], None) if len(calls) > 1 else (None, RuntimeError("rate limited"))

    monkeypatch.setattr(tools, "_ddg", fake_ddg)
    first = tools.web_search("running shoes")
    assert first["results"] == [] and "Search failed" in first["error"]
    second = tools.web_search("running shoes")           # not served from cache: failures aren't stored
    assert len(second["results"]) == 1 and "cached" not in second
    third = tools.web_search("  Running   SHOES ")        # same normalised query -> cache hit
    assert third["cached"] is True and len(calls) == 2


def test_cache_failure_never_breaks_search(monkeypatch):
    monkeypatch.setattr(tools, "SEARCH_REGION", "wt-wt")
    monkeypatch.setattr(tools, "_ddg", lambda q, n: ([_hit("https://a.example/x")], None))
    monkeypatch.setattr(db, "DB_PATH", "/nonexistent-dir/\0/x.db")
    assert len(tools.web_search("anything")["results"]) == 1


def test_extract_product_is_cached(monkeypatch, tmp_db):
    hits = []

    def fake_get(url, **kw):
        hits.append(url)
        return SimpleNamespace(status_code=200, text=PRODUCT_HTML, url=url, headers={"content-type": "text/html"})

    monkeypatch.setattr(httpx, "get", fake_get)
    assert tools.extract_product("https://shop.example/tuf")["price"] == 64990
    assert tools.extract_product("https://shop.example/tuf")["cached"] is True
    assert len(hits) == 1


def test_extract_product_errors(monkeypatch, tmp_db):
    monkeypatch.setattr(httpx, "get", lambda *a, **k: SimpleNamespace(
        status_code=403, text="", url="u", headers={"content-type": "text/html"}))
    assert "403" in tools.extract_product("https://blocked.example/x")["error"]
    monkeypatch.setattr(httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectTimeout("slow")))
    assert "ConnectTimeout" in tools.extract_product("https://slow.example/x")["error"]


# ------------------------------------------------------------------------------ price parsing
def test_price_guess_uses_product_line_and_currency():
    html = """<html><head><title>Hanes Beefy-T Black Tee</title></head><body>
      <p>Free shipping over $35</p><p>Hanes Beefy-T Black Tee now $9.99</p></body></html>"""
    info = tools.parse_product_html(html, "https://www.walmart.com/ip/1")
    assert info["price_guess"] == 9.99 and info["currency"] == "USD"


def test_currency_falls_back_to_domain_when_page_has_none():
    html = "<html><head><title>Plain Tee</title></head><body><p>Plain Tee ₹499</p></body></html>"
    assert tools.parse_product_html(html, "https://shop.example/x")["currency"] == "INR"
    html = "<html><head><title>Tee</title></head><body><p>Nothing here</p></body></html>"
    assert "currency" not in tools.parse_product_html(html, "https://www.amazon.com/x")  # no price, no currency


def test_variant_lines_are_extracted():
    html = "<html><head><title>Tee</title></head><body><p>Available sizes: S, M, L, XL</p><p>Colour: Black</p></body></html>"
    assert any("XL" in l for l in tools.parse_product_html(html, "https://a.example")["variant_lines"])


# ------------------------------------------------------------------------------ packs & variants
@pytest.mark.parametrize("name,n", [
    ("Fruit of the Loom Tee (6-Pack, Black, XL)", 6), ("Hanes Beefy-T (3-Pack)", 3), ("Pack of 4 socks", 4),
    ("Set of 2 mugs", 2), ("Plain black tee", 1), ("Galaxy 128GB", 1), ("Pack of 500 clips", 1),
])
def test_parse_pack_size(name, n):
    assert tools.parse_pack_size(name) == n


def test_pack_prices_are_compared_per_item():
    out = tools.compare_products([
        {"name": "Tee single", "price": 8, "url": "a", "fit_score": 8},
        {"name": "Tee (6-Pack)", "price": 22, "url": "b", "fit_score": 8},
    ])
    single = next(r for r in out["ranked"] if r["name"] == "Tee single")
    pack = next(r for r in out["ranked"] if r["pack_size"] == 6)
    assert pack["unit_price"] == 3.67 and pack["total_cost"] == 22
    assert any("only need 1" in n for n in pack["notes"])
    assert out["ranked"][0] is single   # buying 1 shirt: the single beats an oversized pack


def test_pack_is_fine_when_quantity_matches():
    out = tools.compare_products([
        {"name": "Tee single", "price": 8, "url": "a", "fit_score": 8},
        {"name": "Tee (6-Pack)", "price": 22, "url": "b", "fit_score": 8},
    ], quantity=6)
    assert out["ranked"][0]["pack_size"] == 6          # 22 total beats 6 x 8 = 48
    assert not any("only need" in n for n in out["ranked"][0]["notes"])


def test_budget_applies_to_what_you_pay():
    out = tools.compare_products([{"name": "Tee (6-Pack)", "price": 22, "url": "b", "fit_score": 9}], budget=10)
    assert out["ranked"] == [] and out["over_budget_excluded"][0]["price"] == 22


def test_size_and_colour_handling():
    out = tools.compare_products([
        {"name": "Tee Black XL", "price": 9, "url": "a", "fit_score": 8},
        {"name": "Tee Black XXL", "price": 9, "url": "b", "fit_score": 8},            # XXL is not XL
        {"name": "Tee Black", "price": 9, "url": "c", "fit_score": 8, "size_available": False},
        {"name": "Tee Navy XL", "price": 9, "url": "d", "fit_score": 8, "color_available": False},
        {"name": "Tee Black XL sold out", "price": 9, "url": "e", "fit_score": 8, "in_stock": False},
        {"name": "Tee Black plain", "price": 9, "url": "f", "fit_score": 8, "size_available": True},
    ], size="XL", color="Black")
    ranked = {r["url"]: r for r in out["ranked"]}
    assert set(ranked) == {"a", "b", "f"}
    assert ranked["a"]["size_status"] == "confirmed" and ranked["b"]["size_status"] == "unconfirmed"
    assert ranked["a"]["score"] > ranked["b"]["score"]
    assert any("size XL not confirmed" in n for n in ranked["b"]["notes"])
    assert {u["reason"] for u in out["unavailable_excluded"]} == {
        "size XL unavailable", "colour Black unavailable", "out of stock"}


def test_compare_handles_bad_input():
    out = tools.compare_products([{"name": "X", "price": "n/a", "fit_score": "high", "pack_size": "lots"},
                                  {"name": "", "price": 5}, {"price": 3}], quantity="two", budget="cheap")
    assert [r["name"] for r in out["ranked"]] == ["X"] and out["ranked"][0]["price"] is None
    assert out["quantity"] == 1 and out["budget"] is None
    assert tools.compare_products([])["ranked"] == []


# ------------------------------------------------------------------------------ agent
class Script:
    """LLM stand-in: replays responses; an Exception in the script is raised instead."""

    def __init__(self, *items):
        self.items, self.n = list(items), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        item = self.items[self.n]
        self.n += 1
        if isinstance(item, Exception):
            raise item
        return item


def _quota_error():
    req = httpx.Request("POST", "https://llm.example")
    return openai.RateLimitError("GenerateRequestsPerDayPerProjectPerModel", response=httpx.Response(429, request=req), body=None)


COMPARE = ("compare_products", {"products": [
    {"name": "Plain Tee (3-Pack)", "price": 15, "url": "https://shop.example/tee", "fit_score": 8, "pack_size": 3}]})
SEARCH = ("web_search", {"query": "tee"})


@pytest.fixture
def fake_web(monkeypatch):
    monkeypatch.setitem(agent.TOOL_FUNCS, "web_search", lambda query, max_results=8: {
        "query": query, "results": [{"title": "t", "url": "https://shop.example/tee", "snippet": "s"}]})


def test_agent_degrades_gracefully_when_llm_quota_runs_out(fake_web):
    llm_ = Script(_tool_msg(SEARCH, COMPARE), _quota_error())
    events = list(agent.run_agent("plain tee", client=llm_, model="only-model"))
    final = events[-1]
    assert final["type"] == "final" and final["result"]["degraded"] is True
    rec = final["result"]["recommendations"][0]
    assert rec["name"] == "Plain Tee (3-Pack)" and rec["pack_size"] == 3 and rec["unit_price"] == 5.0
    assert rec["unverified"] is False and "unreviewed" in final["result"]["summary"]


def test_agent_fails_loudly_with_nothing_to_fall_back_on():
    llm_ = Script(_quota_error())
    with pytest.raises(RuntimeError, match="No model left"):
        list(agent.run_agent("plain tee", client=llm_, model="only-model"))


def test_agent_flags_invented_links_and_fills_currency(fake_web):
    llm_ = Script(
        _tool_msg(SEARCH),
        _tool_msg(("submit_recommendations", {"summary": "ok", "recommendations": [
            {"name": "Real", "price": 5, "key_details": [], "reason": "r", "source_url": "https://shop.example/tee/"},
            {"name": "Made up", "price": 5, "currency": "USD", "key_details": [], "reason": "r",
             "source_url": "https://totally-invented.example/p"}]})),
    )
    recs = list(agent.run_agent("tee", client=llm_, model="m"))[-1]["result"]["recommendations"]
    assert recs[0]["unverified"] is False and recs[0]["currency"] is None   # .example: unknown, not guessed
    assert recs[1]["unverified"] is True


def test_agent_accepts_clarifying_answer_for_vague_requests():
    llm_ = Script(_tool_msg(("submit_recommendations", {"summary": "What would you like to buy?", "recommendations": []})))
    final = list(agent.run_agent("hello", client=llm_, model="m"))[-1]
    assert final["type"] == "final" and final["result"]["recommendations"] == [] and "elapsed_s" in final


def test_agent_survives_tool_crash_and_bad_arguments(monkeypatch):
    monkeypatch.setitem(agent.TOOL_FUNCS, "web_search", lambda **kw: 1 / 0)
    llm_ = Script(
        _tool_msg(("web_search", {"query": "x"}), ("extract_product", {"nope": 1})),
        _tool_msg(("submit_recommendations", {"summary": "nothing found", "recommendations": []})),
    )
    events = list(agent.run_agent("tee", client=llm_, model="m"))
    results = [e["result"] for e in events if e["type"] == "tool_result"]
    assert "ZeroDivisionError" in results[0]["error"]            # the prefetch crashed: agent carries on
    assert "ZeroDivisionError" in results[1]["error"] and "Bad arguments" in results[2]["error"]
    assert events[-1]["type"] == "final"


def test_system_prompt_mentions_region_and_variants():
    prompt = agent.SYSTEM_PROMPT.format(today="2026-10-01", region_note=agent._region_note())
    assert "untrusted" in prompt and "pack_size" in prompt and "clarifying question" in prompt


# ------------------------------------------------------------------------------ llm
def test_exhausted_models_are_skipped_then_retried(monkeypatch):
    monkeypatch.setattr(llm, "_EXHAUSTED", {})
    llm.mark_exhausted("a")
    assert llm.live_models(["a", "b"]) == ["b"]
    llm.mark_exhausted("b")
    assert llm.live_models(["a", "b"]) == ["a", "b"]          # all dead: try everything again
    llm._EXHAUSTED["a"] -= llm.EXHAUSTED_SKIP_SECONDS + 1
    assert llm.live_models(["a", "b"]) == ["a"]


# ------------------------------------------------------------------------------ database
def test_old_database_is_migrated(monkeypatch, tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE searches (id INTEGER PRIMARY KEY AUTOINCREMENT, query TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'running', provider TEXT, model TEXT, requirements TEXT, summary TEXT,
            comparison TEXT, error TEXT, created_at TEXT NOT NULL, finished_at TEXT);
        CREATE TABLE recommendations (id INTEGER PRIMARY KEY AUTOINCREMENT, search_id INTEGER, rank INTEGER,
            name TEXT, price REAL, currency TEXT, key_details TEXT, reason TEXT, source_url TEXT);
        INSERT INTO searches (query, status, created_at) VALUES ('old query', 'done', '2026-09-30T00:00:00+00:00');
    """)
    conn.commit(); conn.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))
    db.init_db(); db.init_db()                                   # idempotent
    assert db.list_searches(client_id="anyone")[0]["query"] == "old query"   # legacy rows stay visible
    sid = db.create_search("new", "me")
    db.finish_search(sid, {"summary": "s", "recommendations": [
        {"name": "n", "key_details": [], "reason": "r", "source_url": "u", "pack_size": 3, "unit_price": 5.0}]}, None)
    assert db.get_search(sid, "me")["recommendations"][0]["pack_size"] == 3


def test_history_is_private_per_client(tmp_db):
    mine, theirs = db.create_search("my phone", "me"), db.create_search("their phone", "them")
    assert [s["query"] for s in db.list_searches(client_id="me")] == ["my phone"]
    assert db.get_search(theirs, "me") is None and db.delete_search(theirs, "me") is False
    assert db.delete_search(mine, "me") is True


def test_degraded_results_are_not_replayed(tmp_db):
    sid = db.create_search("tee", "me")
    db.finish_search(sid, {"summary": "s", "degraded": True,
                           "recommendations": [{"name": "n", "key_details": [], "reason": "r", "source_url": "u"}]}, None)
    assert db.get_search(sid, "me")["status"] == "partial"
    assert db.find_recent_done("tee", "me", 3600) is None


# ------------------------------------------------------------------------------ API
@pytest.fixture
def api(monkeypatch, tmp_db):
    from fastapi.testclient import TestClient
    from app import main
    main._recent.clear()
    runs = []

    def factory():
        runs.append(1)
        return (Script(_tool_msg(("parse_requirements", {"product_type": "tee", "use_case": "daily",
                                                         "search_queries": ["tee"]})),
                       _tool_msg(("submit_recommendations", {"summary": "done", "recommendations": [
                           {"name": "Tee", "price": 9, "currency": "USD", "key_details": [], "reason": "r",
                            "source_url": "https://shop.example/tee"}]}))), ["m"], "groq")

    monkeypatch.setattr("app.agent.get_client", factory)
    return SimpleNamespace(main=main, client=TestClient(main.app), new_client=lambda: TestClient(main.app), runs=runs)


def _events(resp):
    return [json.loads(l) for l in resp.text.splitlines() if l]


def test_identical_search_is_replayed_instantly(api):
    first = _events(api.client.post("/api/search", json={"query": "black plain t-shirt"}))
    assert first[-1]["type"] == "final" and len(api.runs) == 1
    again = _events(api.client.post("/api/search", json={"query": "  Black   PLAIN t-shirt "}))
    assert len(api.runs) == 1 and again[0]["cached"] is True
    assert again[-1]["cached"] is True and again[-1]["result"]["recommendations"][0]["name"] == "Tee"
    fresh = _events(api.client.post("/api/search", json={"query": "black plain t-shirt", "refresh": True}))
    assert len(api.runs) == 2 and "cached" not in fresh[0]


def test_other_people_do_not_get_my_cached_search_or_history(api):
    api.client.post("/api/search", json={"query": "black plain t-shirt"})
    other = api.new_client()
    assert other.get("/api/history").json() == []
    _events(other.post("/api/search", json={"query": "black plain t-shirt"}))
    assert len(api.runs) == 2
    first_id = api.client.get("/api/history").json()[0]["id"]
    assert other.get(f"/api/history/{first_id}").status_code == 404
    assert other.delete(f"/api/history/{first_id}").status_code == 404


def test_rate_limit(api, monkeypatch):
    monkeypatch.setattr(api.main, "RATE_LIMIT", 2)
    codes = [api.client.post("/api/search", json={"query": f"tee number {i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    r = api.client.post("/api/search", json={"query": "tee number 9"})
    assert "Retry-After" in r.headers and "Too many searches" in r.json()["detail"]


def test_busy_server_returns_503(api, monkeypatch):
    import threading
    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(api.main, "_slots", sem)
    assert api.client.post("/api/search", json={"query": "blue jeans"}).status_code == 503


def test_slot_is_released_after_a_failed_run(api, monkeypatch):
    import threading
    sem = threading.BoundedSemaphore(1)
    monkeypatch.setattr(api.main, "_slots", sem)
    monkeypatch.setattr("app.agent.get_client", lambda: (_ for _ in ()).throw(RuntimeError("No LLM key configured")))
    events = _events(api.client.post("/api/search", json={"query": "blue jeans"}))
    assert events[-1]["type"] == "error" and "No LLM key" in events[-1]["message"]
    assert sem.acquire(blocking=False)                      # slot was given back
    assert api.client.get("/api/history").json()[0]["status"] == "error"


@pytest.mark.parametrize("query", ["??", "   ", "!!!!!!", "ab"])
def test_junk_queries_are_rejected(api, query):
    assert api.client.post("/api/search", json={"query": query}).status_code == 422
    assert api.runs == []


# ------------------------------------------------------------------------------ prefetch
def test_first_search_is_prefetched_so_the_llm_can_skip_straight_to_extraction(monkeypatch):
    searched = []
    monkeypatch.setitem(agent.TOOL_FUNCS, "web_search", lambda query, max_results=8: (
        searched.append(query), {"query": query, "results": [{"title": "t", "url": "https://shop.example/tee", "snippet": "s"}]})[1])
    llm_ = Script(_tool_msg(("submit_recommendations", {"summary": "s", "recommendations": [
        {"name": "Tee", "price": 5, "key_details": [], "reason": "r", "source_url": "https://shop.example/tee"}]})))
    seen_messages = []
    orig = llm_._create
    llm_.chat.completions.create = lambda **kw: (seen_messages.append(list(kw["messages"])), orig(**kw))[1]
    events = list(agent.run_agent("black tee xl", client=llm_, model="m"))
    assert searched == ["black tee xl"]
    assert events[1]["type"] == "tool_call" and events[1].get("prefetch") is True
    assert "https://shop.example/tee" in seen_messages[0][1]["content"]
    assert events[-1]["result"]["recommendations"][0]["unverified"] is False  # prefetched URLs count as seen


def test_failed_prefetch_does_not_stop_the_agent(monkeypatch):
    monkeypatch.setitem(agent.TOOL_FUNCS, "web_search", lambda query, max_results=8: {
        "query": query, "results": [], "error": "Search failed: throttled"})
    llm_ = Script(_tool_msg(("submit_recommendations", {"summary": "Nothing found", "recommendations": []})))
    assert list(agent.run_agent("black tee", client=llm_, model="m"))[-1]["type"] == "final"
