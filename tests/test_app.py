"""Offline tests: tool logic, and a full agent + API + DB run driven by a scripted fake LLM."""
import json
from types import SimpleNamespace

import pytest

from app import tools

PRODUCT_HTML = """
<html><head><title>ASUS TUF A15 | Shop</title>
<script type="application/ld+json">
{"@context":"https://schema.org","@graph":[{"@type":"WebPage"},
 {"@type":"Product","name":"ASUS TUF Gaming A15 (Ryzen 7 7435HS, RTX 3050 6GB)","brand":{"@type":"Brand","name":"ASUS"},
  "offers":{"@type":"Offer","price":"64,990","priceCurrency":"INR","availability":"https://schema.org/InStock"},
  "aggregateRating":{"ratingValue":"4.3","reviewCount":"1200"}}]}
</script></head>
<body><table><tr><th>Processor</th><td>AMD Ryzen 7 7435HS</td></tr>
<tr><th>RAM</th><td>16 GB DDR5</td></tr><tr><th>Colour</th><td>Grey</td></tr></table>
<p>Special price ₹64,990 today only</p></body></html>
"""


def test_parse_product_html_reads_jsonld_and_specs():
    info = tools.parse_product_html(PRODUCT_HTML, "https://shop.example/tuf")
    assert info["name"].startswith("ASUS TUF Gaming A15")
    assert info["price"] == 64990
    assert info["currency"] == "INR"
    assert info["brand"] == "ASUS"
    assert info["availability"] == "InStock"
    assert "Processor: AMD Ryzen 7 7435HS" in info["specs"]
    assert not any("Colour" in s for s in info["specs"])
    assert any("64,990" in line for line in info["price_mentions"])


def test_parse_specs():
    s = tools.parse_specs("Core i7-13620H, 16GB DDR5 RAM, 1TB NVMe SSD, RTX 4060 8GB GDDR6")
    assert s == {"ram_gb": 16, "storage_gb": 1024, "gpu": "RTX 4060", "dedicated_gpu": True,
                 "vram_gb": 8, "cpu": "Core i7-13620H"}


def test_compare_products_budget_dedupe_and_ranking():
    result = tools.compare_products(
        [
            {"name": "Laptop A RTX 4050", "price": 69000, "url": "a", "fit_score": 9,
             "specs": {"ram": "16GB DDR5", "gpu": "RTX 4050 6GB GDDR6"}},
            {"name": "Laptop B", "price": 45000, "url": "b", "fit_score": 4, "specs": {"ram": "8GB DDR4"}},
            {"name": "Laptop A RTX 4050", "price": 68000, "url": "dup", "fit_score": 9},
            {"name": "Laptop C", "price": 90000, "url": "c", "fit_score": 10},
            {"name": "Laptop D", "price": 72000, "url": "d", "fit_score": 7},
        ],
        budget=70000,
    )
    names = [r["name"] for r in result["ranked"]]
    assert names[0] == "Laptop A RTX 4050"
    assert "Laptop C" not in names and result["over_budget_excluded"][0]["name"] == "Laptop C"
    assert len(names) == 3  # duplicate removed
    assert "slightly over budget" in next(r for r in result["ranked"] if r["name"] == "Laptop D")["notes"]


# ----------------------------------------------------------------------------- fake LLM
def _tool_msg(*calls):
    tcs = [
        SimpleNamespace(id=f"call_{i}", type="function",
                        function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for i, (name, args) in enumerate(calls)
    ]
    msg = SimpleNamespace(content=None, tool_calls=tcs)
    msg.model_dump = lambda exclude_none=True: {
        "role": "assistant",
        "tool_calls": [{"id": t.id, "type": "function",
                        "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in tcs],
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeLLM:
    """Plays back a fixed agent trajectory and checks tool results are fed back."""

    def __init__(self):
        self.script = [
            _tool_msg(("parse_requirements", {"product_type": "laptop", "budget_max": 70000, "currency": "INR",
                                              "use_case": "AI development", "must_have": ["RTX GPU", "16GB RAM"],
                                              "search_queries": ["best laptop AI development under 70000 India"]})),
            _tool_msg(("web_search", {"query": "best laptop AI development under 70000 India 2026"})),
            _tool_msg(("extract_product", {"url": "https://shop.example/tuf"})),
            _tool_msg(("compare_products", {"budget": 70000, "products": [
                {"name": "ASUS TUF Gaming A15", "price": 64990, "currency": "INR", "url": "https://shop.example/tuf",
                 "specs": {"gpu": "RTX 3050 6GB GDDR6", "ram": "16GB DDR5"}, "fit_score": 8},
                {"name": "Too Expensive Pro", "price": 99000, "url": "https://shop.example/pro", "fit_score": 10},
            ]})),
            _tool_msg(("submit_recommendations", {"summary": "The TUF A15 is the best fit.", "recommendations": [
                {"name": "ASUS TUF Gaming A15", "price": 64990, "currency": "INR",
                 "key_details": ["RTX 3050 6GB", "16GB DDR5"], "reason": "Dedicated NVIDIA GPU within budget",
                 "source_url": "https://shop.example/tuf"}]})),
        ]
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.script[len(self.calls) - 1]


@pytest.fixture
def offline_tools(monkeypatch):
    monkeypatch.setattr(tools, "web_search", lambda query, max_results=8: {
        "query": query, "results": [{"title": "TUF A15", "url": "https://shop.example/tuf", "snippet": "₹64,990"}]})
    monkeypatch.setattr(tools, "extract_product", lambda url: tools.parse_product_html(PRODUCT_HTML, url))
    from app import agent
    monkeypatch.setitem(agent.TOOL_FUNCS, "web_search", tools.web_search)
    monkeypatch.setitem(agent.TOOL_FUNCS, "extract_product", tools.extract_product)


def test_agent_loop(offline_tools):
    from app.agent import run_agent
    fake = FakeLLM()
    events = list(run_agent("best laptop for AI dev under 70,000", client=fake, model="fake"))
    types = [e["type"] for e in events]
    assert types[0] == "meta" and types[-1] == "final"
    assert "requirements" in types
    final = events[-1]
    assert final["result"]["recommendations"][0]["name"] == "ASUS TUF Gaming A15"
    assert [r["name"] for r in final["comparison"]["ranked"]] == ["ASUS TUF Gaming A15"]
    # The extract_product result was passed back to the LLM as a tool message.
    tool_msgs = [m["content"] for m in fake.calls[-1]["messages"] if m["role"] == "tool"]
    assert any('"price": 64990' in c and "Ryzen 7 7435HS" in c for c in tool_msgs)


def test_api_streams_and_persists(offline_tools, monkeypatch, tmp_path):
    from app import db
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "test.db"))
    from app import main
    db.init_db()
    fake = FakeLLM()
    monkeypatch.setattr("app.agent.get_client", lambda: (fake, ["fake-model"], "groq"))

    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    resp = client.post("/api/search", json={"query": "best laptop for AI dev under 70,000"})
    events = [json.loads(line) for line in resp.text.splitlines() if line]
    assert events[0]["type"] == "start" and events[-1]["type"] == "final"

    history = client.get("/api/history").json()
    assert history[0]["status"] == "done" and history[0]["n_results"] == 1
    detail = client.get(f"/api/history/{history[0]['id']}").json()
    assert detail["provider"] == "groq"
    assert detail["requirements"]["budget_max"] == 70000
    assert detail["recommendations"][0]["source_url"] == "https://shop.example/tuf"
    assert detail["comparison"]["ranked"][0]["price"] == 64990
    assert len(detail["steps"]) == len(events) - 1

    assert client.delete(f"/api/history/{history[0]['id']}").status_code == 200
    assert client.get("/api/history").json() == []


def test_daily_quota_falls_back_to_next_model(offline_tools):
    import httpx
    import openai
    from app.agent import _completion

    req = httpx.Request("POST", "https://llm.example")
    daily = openai.RateLimitError("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                                  response=httpx.Response(429, request=req), body=None)
    seen = []

    def create(**kw):
        seen.append(kw["model"])
        if kw["model"] == "model-a":
            raise daily
        return "response"

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    models = ["model-a", "model-b"]
    gen = _completion(client, models, [])
    event = next(gen)
    assert event == {"type": "model_switch", "from": "model-a", "to": "model-b", "reason": "daily quota used up"}
    with pytest.raises(StopIteration) as done:
        next(gen)
    assert done.value.value == "response" and seen == ["model-a", "model-b"] and models == ["model-b"]
