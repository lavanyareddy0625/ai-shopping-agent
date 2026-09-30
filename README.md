# AI Shopping Agent (MVP)

Type a request such as **"find the best laptop for AI development under 70,000"**. An LLM agent
works out what you need, searches the public web, pulls product details from real pages,
compares the candidates, and recommends the best options with prices, key specs and source links.
Every search, agent step and result is saved in SQLite.

## Stack

| Layer     | Choice |
|-----------|--------|
| LLM       | **Groq** (`llama-3.3-70b-versatile`) or **Gemini** (`gemini-3.8-flash`), both called through their OpenAI-compatible APIs with function calling |
| Backend   | FastAPI (Python 3.12), streams agent progress as NDJSON |
| Search    | DuckDuckGo via `ddgs` (no API key needed) |
| Scraping  | httpx + BeautifulSoup (schema.org JSON-LD, meta tags, spec tables, price text) |
| Database  | SQLite (`data/shopping.db`) |
| Frontend  | A single static HTML/JS page |

## Run it

```bash
cd ai-shopping-agent
cp .env.example .env          # then paste a GROQ_API_KEY or GEMINI_API_KEY
uv sync                       # or: pip install -r requirements.txt  (Python 3.12+)
uv run uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

Tests (offline; a scripted fake LLM drives the full agent → API → DB path):

```bash
uv run pytest -q
```

## Agentic workflow

```
user request
   │
   ▼
LLM ──► parse_requirements   product type, budget, currency, use case, must-haves, search queries
   │
   ├──► web_search (×N, parallel)     DuckDuckGo → titles, URLs, snippets
   │
   ├──► extract_product (×N, parallel) fetch page → JSON-LD Product / meta price / spec rows / ₹ mentions
   │
   ├──► compare_products              dedupe, budget filter, score & rank (deterministic code)
   │
   └──► submit_recommendations        top 3–5 with price, key details, reason, source URL
```

The LLM decides which tool to call next. It can re-search, open more pages, or skip pages that
block bots and use the search snippets instead. The loop stops at 12 rounds; if the agent runs
out of rounds, it is forced to submit its recommendations.

**Comparison scoring** (`app/tools.py::compare_products`):

```
score = 0.60 × fit_score (LLM-judged 0–10 fit to requirements)
      + 0.25 × price efficiency (1 − 0.5 × price/budget)
      + 0.15 × hardware bonus (RAM, dedicated GPU, VRAM, SSD parsed from specs)
```

Products more than 5% over budget are excluded and listed separately.

**Grounding:** the system prompt forbids inventing prices or URLs. Every `source_url` must come
from a tool result, and the UI shows the raw result of every tool call so you can check them.

## Project layout

```
app/
  main.py    FastAPI routes: POST /api/search (stream), GET/DELETE /api/history[/id], GET /
  agent.py   system prompt, tool schemas, tool-calling loop (generator of events)
  tools.py   web_search, extract_product, compare_products
  llm.py     Groq / Gemini client selection
  db.py      SQLite schema + queries (searches, agent_steps, recommendations)
static/index.html   UI: query box, live agent trace, recommendation cards, comparison table, history
tests/test_app.py   offline tests
```

## API

| Method | Path | Description |
|--------|------|-------------|
| POST   | `/api/search` `{"query": "..."}` | Runs the agent. Streams events: `start`, `meta`, `requirements`, `thought`, `tool_call`, `tool_result`, `final`, `error` |
| GET    | `/api/history` | Past searches |
| GET    | `/api/history/{id}` | One search: requirements, recommendations, comparison, full agent trace |
| DELETE | `/api/history/{id}` | Delete a search |

## Limitations

- Amazon and Flipkart often block scripted requests. When that happens the agent uses search
  snippets and price-comparison or review pages (Smartprix, 91mobiles, MySmartPrice…) instead.
- Prices come from public pages at search time and may be stale.
- Groq's free tier has tight tokens-per-minute limits. The client retries on HTTP 429. If you hit
  daily limits, switch to `LLM_MODEL=llama-3.1-8b-instant` or use Gemini.
- Search results default to India (`SEARCH_REGION=in-en`). Change it for other markets.
