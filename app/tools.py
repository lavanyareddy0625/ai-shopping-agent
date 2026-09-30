"""Agent tools: web search, product extraction and product comparison.

Each tool is a plain function returning a JSON-serialisable dict so the agent
loop can hand results straight back to the LLM.
"""
import json
import os
import re
import threading
import time
from typing import Any, Iterator, Optional

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

SEARCH_REGION = os.getenv("SEARCH_REGION", "in-en")
_search_lock = threading.Lock()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

SPEC_KEYWORDS = (
    "processor", "cpu", "chip", "ram", "memory", "graphics", "gpu", "vram",
    "storage", "ssd", "display", "screen", "resolution", "refresh", "battery",
    "weight", "operating system", "cores", "camera", "warranty",
)

PRICE_RE = re.compile(r"(?:₹|Rs\.?|INR|\$|USD)\s?(\d{1,3}(?:,\d{2,3})+|\d{3,7})(?:\.\d{1,2})?", re.I)


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #
def web_search(query: str, max_results: int = 8) -> dict:
    """Search the public web (DuckDuckGo) and return titles, URLs and snippets."""
    max_results = max(1, min(int(max_results or 8), 10))
    hits, error = None, None
    # DuckDuckGo throttles bursts, so searches run one at a time with a retry.
    with _search_lock:
        for attempt in range(2):
            try:
                hits = DDGS().text(query, region=SEARCH_REGION, safesearch="moderate", max_results=max_results)
                break
            except Exception as e:  # network errors, throttling, "No results found"
                error = e
                time.sleep(2 + attempt * 2)
        time.sleep(1)
    if hits is None:
        return {"query": query, "results": [], "error": f"Search failed: {error}. Try a simpler query."}
    results = [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": (h.get("body") or "")[:300],
        }
        for h in hits or []
        if h.get("href")
    ]
    return {"query": query, "results": results}


# --------------------------------------------------------------------------- #
# extract_product
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(value))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _iter_jsonld_products(data: Any) -> Iterator[dict]:
    if isinstance(data, dict):
        types = data.get("@type")
        types = types if isinstance(types, list) else [types]
        if any(t in ("Product", "ProductGroup") for t in types):
            yield data
        for v in data.values():
            if isinstance(v, (dict, list)):
                yield from _iter_jsonld_products(v)
    elif isinstance(data, list):
        for item in data:
            yield from _iter_jsonld_products(item)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_product_html(html: str, url: str) -> dict:
    """Pull structured product data out of a page (JSON-LD, meta tags, spec tables, text)."""
    soup = BeautifulSoup(html, "html.parser")
    info: dict = {"url": url}

    # 1. schema.org JSON-LD -- the most reliable source when present.
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        product = next(_iter_jsonld_products(data), None)
        if not product:
            continue
        info["name"] = _clean(product.get("name", ""))
        brand = product.get("brand")
        info["brand"] = brand.get("name") if isinstance(brand, dict) else brand
        offers = product.get("offers")
        if isinstance(offers, list) and offers:
            offers = offers[0]
        if isinstance(offers, dict):
            info["price"] = _to_float(offers.get("price") or offers.get("lowPrice"))
            info["currency"] = offers.get("priceCurrency")
            availability = offers.get("availability")
            if availability:
                info["availability"] = str(availability).rsplit("/", 1)[-1]
        rating = product.get("aggregateRating")
        if isinstance(rating, dict):
            info["rating"] = rating.get("ratingValue")
            info["review_count"] = rating.get("reviewCount") or rating.get("ratingCount")
        if product.get("description"):
            info["description"] = _clean(product["description"])[:400]
        break

    # 2. Meta tags as fallback.
    def meta(*names: str) -> Optional[str]:
        for n in names:
            tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n}) \
                or soup.find("meta", attrs={"itemprop": n})
            if tag and tag.get("content"):
                return tag["content"]
        return None

    if not info.get("name"):
        title = meta("og:title") or (soup.title.string if soup.title else "")
        info["name"] = _clean(title)[:200]
    if info.get("price") is None:
        info["price"] = _to_float(meta("product:price:amount", "og:price:amount", "price"))
        info["currency"] = info.get("currency") or meta("product:price:currency", "og:price:currency", "priceCurrency")
    if not info.get("description"):
        desc = meta("og:description", "description")
        if desc:
            info["description"] = _clean(desc)[:400]

    # 3. Spec tables (key: value rows).
    spec_lines: list[str] = []
    for row in soup.find_all("tr"):
        cells = [_clean(c.get_text(" ")) for c in row.find_all(["th", "td"])]
        cells = [c for c in cells if c]
        if len(cells) >= 2:
            line = f"{cells[0]}: {' '.join(cells[1:])}"[:200]
            if any(k in line.lower() for k in SPEC_KEYWORDS):
                spec_lines.append(line)

    # 4. Visible text: spec-looking lines and lines that mention a price.
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        tag.decompose()
    lines = [_clean(line) for line in soup.get_text("\n").splitlines()]
    lines = [line for line in lines if 4 <= len(line) <= 220]

    for line in lines:
        if len(spec_lines) >= 25:
            break
        low = line.lower()
        if any(k in low for k in SPEC_KEYWORDS) and line not in spec_lines:
            spec_lines.append(line)

    price_lines: list[str] = []
    for line in lines:
        if PRICE_RE.search(line) and line not in price_lines:
            price_lines.append(line)
        if len(price_lines) >= 15:
            break

    info["specs"] = spec_lines[:25]
    # Listing / review pages ("best laptops under X") put many models + prices in text.
    info["price_mentions"] = price_lines
    if info.get("price") is None and price_lines:
        info["price_guess"] = _to_float(PRICE_RE.search(price_lines[0]).group(1))
    return {k: v for k, v in info.items() if v not in (None, "", [])}


def extract_product(url: str) -> dict:
    """Fetch a URL and extract product name, price, rating and key specs."""
    try:
        resp = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=15)
    except httpx.HTTPError as e:
        return {"url": url, "error": f"Fetch failed: {type(e).__name__}. Use the search snippet instead."}
    if resp.status_code >= 400:
        return {"url": url, "error": f"HTTP {resp.status_code} (site likely blocks bots). Use the search snippet instead."}
    if "html" not in resp.headers.get("content-type", "html"):
        return {"url": url, "error": "Not an HTML page."}
    return parse_product_html(resp.text[:1_500_000], str(resp.url))


# --------------------------------------------------------------------------- #
# compare_products
# --------------------------------------------------------------------------- #
def parse_specs(text: str) -> dict:
    """Best-effort extraction of comparable hardware specs from free text."""
    specs: dict = {}
    t = text or ""
    if m := re.search(r"(\d{1,3})\s?GB\s*(?:of\s*)?(?:LP)?(?:DDR\d\w*|RAM|unified|memory)", t, re.I):
        specs["ram_gb"] = int(m.group(1))
    if m := re.search(r"(\d+(?:\.\d)?)\s?(TB|GB)\s*(?:PCIe\s*)?(?:NVMe\s*)?(?:M\.2\s*)?SSD", t, re.I):
        size = float(m.group(1))
        specs["storage_gb"] = int(size * 1024 if m.group(2).upper() == "TB" else size)
    if m := re.search(r"(RTX\s?(?:A)?\d{3,4}\w*(?:\s?Ti)?|GTX\s?\d{3,4}\w*|Radeon\s?RX\s?\d{3,4}\w*|Arc\s?A\d{3}\w*)", t, re.I):
        specs["gpu"] = _clean(m.group(1))
        specs["dedicated_gpu"] = True
    if m := re.search(r"(\d{1,2})\s?GB\s*(?:GDDR\d\w*|VRAM|graphics)", t, re.I):
        specs["vram_gb"] = int(m.group(1))
    if m := re.search(
        r"(Core\s?Ultra\s?\d\s?\d{3}\w*|Core\s?i\d[\s-]?\d{4,5}\w*|Ryzen\s?(?:AI\s?)?\d\s?(?:\w+\s)?\d{3,4}\w*|Apple\s?M\d(?:\s?(?:Pro|Max))?|Snapdragon\s?X\s?\w+)",
        t, re.I,
    ):
        specs["cpu"] = _clean(m.group(1))
    return specs


def _hardware_bonus(specs: dict) -> float:
    """0..1 bonus from objectively comparable specs (zero when none are known)."""
    score = 0.0
    if ram := specs.get("ram_gb"):
        score += min(ram, 32) / 32 * 0.4
    if specs.get("dedicated_gpu"):
        score += 0.3
    if vram := specs.get("vram_gb"):
        score += min(vram, 8) / 8 * 0.15
    if storage := specs.get("storage_gb"):
        score += min(storage, 1024) / 1024 * 0.15
    return round(min(score, 1.0), 3)


def compare_products(products: list[dict], budget: Optional[float] = None, priorities: Optional[list[str]] = None) -> dict:
    """Deduplicate, budget-filter and rank candidate products.

    score = 0.60 * requirement fit (LLM-judged, 0-10)
          + 0.25 * price efficiency (cheaper relative to budget is better)
          + 0.15 * hardware bonus (parsed RAM / GPU / VRAM / SSD)
    """
    budget = _to_float(budget)
    ranked, over_budget, seen = [], [], set()

    for p in products or []:
        name = _clean(str(p.get("name", "")))
        key = re.sub(r"[^a-z0-9]", "", name.lower())[:50]
        if not name or key in seen:
            continue
        seen.add(key)

        price = _to_float(p.get("price"))
        raw_specs = p.get("specs") or {}
        spec_text = " ".join(f"{k} {v}" for k, v in raw_specs.items()) if isinstance(raw_specs, dict) else str(raw_specs)
        parsed = parse_specs(f"{name} {spec_text}")

        try:
            fit = max(0.0, min(float(p.get("fit_score", 5)), 10.0))
        except (TypeError, ValueError):
            fit = 5.0

        notes = []
        if price is None:
            price_score = 0.5
            notes.append("price unknown")
        elif budget:
            if price > budget * 1.05:
                over_budget.append({"name": name, "price": price, "url": p.get("url")})
                continue
            if price > budget:
                notes.append("slightly over budget")
            price_score = max(0.0, 1 - 0.5 * price / budget)
        else:
            price_score = 0.5

        hw = _hardware_bonus(parsed)
        score = round(0.60 * fit / 10 + 0.25 * price_score + 0.15 * hw, 4)
        ranked.append({
            "name": name,
            "price": price,
            "currency": p.get("currency"),
            "url": p.get("url"),
            "fit_score": fit,
            "price_score": round(price_score, 3),
            "hardware_bonus": hw,
            "score": score,
            "parsed_specs": parsed,
            "specs": raw_specs,
            "notes": notes,
        })

    ranked.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return {
        "budget": budget,
        "priorities": priorities or [],
        "ranked": ranked,
        "over_budget_excluded": over_budget,
        "method": "score = 0.60*fit + 0.25*price_efficiency + 0.15*hardware_bonus; >5% over budget excluded",
    }
