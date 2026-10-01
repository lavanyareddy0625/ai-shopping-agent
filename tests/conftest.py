"""Keep every test offline: tools that would hit the network are stubbed unless a test overrides them."""
import pytest

from app import agent


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setitem(agent.TOOL_FUNCS, "web_search", lambda query, max_results=8: {
        "query": query, "results": [{"title": "t", "url": "https://shop.example/tee", "snippet": "s"}]})
    monkeypatch.setitem(agent.TOOL_FUNCS, "extract_product", lambda url: {"url": url, "error": "offline test"})
