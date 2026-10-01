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

## What it handles

| Situation | Behaviour |
|-----------|-----------|
| Currency / region | Searches favour stores that serve `SEARCH_REGION` (India: Amazon.in, Flipkart, Myntra, Ajio, Croma). Every price carries a currency, inferred from the store's domain when the page doesn't state one. |
| Size, colour, quantity | Parsed from the request. Listings that are out of stock or lack the requested size/colour are excluded (and listed); unconfirmed variants are ranked lower and flagged. |
| Multi-packs | `6-Pack` / `pack of 3` are detected; items are compared per unit and the budget applies to what you actually pay. Buying a bigger pack than needed is penalised. |
| Invented links | Any recommendation whose URL never appeared in a tool result is flagged "unverified" in the UI. |
| LLM quota exhausted | Models out of daily quota are skipped for 6 h. If the model fails after candidates were ranked, the top candidates are returned as a *partial* result instead of an error. |
| Vague / non-shopping requests | The agent asks one clarifying question instead of guessing. |
| Prompt injection | Web pages and snippets are treated as untrusted data. |
| Repeat searches | Search and page results are cached (6 h). An identical finished search is replayed instantly; "Search again" forces fresh prices. |
| Several users | History is private per browser (cookie). Per-IP rate limit and a cap on concurrent agent runs protect the free-tier quota. |

## Agentic workflow

```
user request
   │
   ▼
(code) web_search on the raw request, run immediately so the LLM doesn't spend a round on it
   │
   ▼
LLM ──► parse_requirements   product type, budget, currency, size, colour, quantity, use case, must-haves
   │
   ├──► web_search (×N, parallel)     DuckDuckGo → titles, URLs, snippets
   │
   ├──► extract_product (×N, parallel) fetch page → JSON-LD Product / meta price / spec rows / ₹ mentions
   │
   ├──► compare_products              dedupe, budget filter, score & rank (deterministic code)
   │
   └──► submit_recommendations        top 3–5 with price, key details, reason, source URL
```

A typical run takes 3 LLM rounds (about 1-2 minutes on free tiers; mostly LLM latency and rate limits, so a
Groq key is much faster). The LLM decides which tool to call next. It can re-search, open more pages, or skip pages that
block bots and use the search snippets instead. The loop stops at 8 rounds; if the agent runs
out of rounds, it is forced to submit its recommendations.

**Comparison scoring** (`app/tools.py::compare_products`):

```
score = 0.60 × fit_score (LLM-judged 0–10 fit to requirements)
      + 0.25 × price efficiency (1 − 0.5 × cost/budget, or unit price vs the other candidates)
      + 0.15 × hardware bonus (RAM, dedicated GPU, VRAM, SSD parsed from specs)
      − 0.10 oversized pack − 0.05 per unconfirmed size/colour
```

Products more than 5% over budget, out of stock, or missing the requested size/colour are excluded
and listed separately.

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
| POST   | `/api/search` `{"query": "...", "refresh": false}` | Runs the agent (or replays a recent identical search). Streams events: `start`, `meta`, `requirements`, `thought`, `tool_call`, `tool_result`, `final`, `error`. Returns 422 for junk input, 429 when rate-limited, 503 when busy |
| GET    | `/api/history` | Past searches |
| GET    | `/api/history/{id}` | One search: requirements, recommendations, comparison, full agent trace |
| DELETE | `/api/history/{id}` | Delete a search |

## Limitations

- Amazon and Flipkart often block scripted requests. When that happens the agent uses search
  snippets and price-comparison or review pages (Smartprix, 91mobiles, MySmartPrice…) instead.
- Prices come from public pages at search time and may be stale.
- Groq's free tier has tight tokens-per-minute limits. The client retries on HTTP 429. If you hit
  daily limits, switch to `LLM_MODEL=llama-3.1-8b-instant` or use Gemini.
- Size/colour/stock are only verified when a page states them; otherwise the result is marked "not confirmed".
- Results come from DuckDuckGo plus scraping, so they are only as good as what public pages expose.
- Search results default to India (`SEARCH_REGION=in-en`). Change it for other markets.
