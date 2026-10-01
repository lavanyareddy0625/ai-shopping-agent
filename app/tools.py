"""Agent tools: web search, product extraction and product comparison.

Each tool is a plain function returning a JSON-serialisable dict so the agent
loop can hand results straight back to the LLM.
"""
import json
import math
import os
import re
import threading
import time
from urllib.parse import urlparse
from typing import Any, Iterator, Optional

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

from . import db

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

_TLD_CURRENCY = {"in": "INR", "uk": "GBP", "de": "EUR", "fr": "EUR", "it": "EUR", "es": "EUR",
                 "ca": "CAD", "au": "AUD", "jp": "JPY", "com": "USD"}
# Indian stores / price sites that use a plain .com domain.
_INR_COM_HOSTS = ("flipkart.com", "myntra.com", "ajio.com", "croma.com", "tatacliq.com", "nykaa.com",
                  "smartprix.com", "91mobiles.com", "gadgets360.com", "mysmartprice.com", "findprix.com")

# Stores to favour for each DuckDuckGo region, so a generic query returns shops that sell locally.
REGION_PROFILES = {
    "in-en": {"country": "India", "currency": "INR", "tlds": ("in",),
              "stores": ("amazon.in", "flipkart.com", "myntra.com", "ajio.com", "croma.com")},
    "us-en": {"country": "the United States", "currency": "USD", "tlds": (),
              "stores": ("amazon.com", "walmart.com", "target.com", "bestbuy.com")},
    "uk-en": {"country": "the United Kingdom", "currency": "GBP", "tlds": ("uk",),
              "stores": ("amazon.co.uk", "argos.co.uk", "currys.co.uk")},
}
SEARCH_TTL = int(os.getenv("SEARCH_CACHE_TTL", "21600"))    # 6 h: identical searches are reused
EXTRACT_TTL = int(os.getenv("EXTRACT_CACHE_TTL", "21600"))


def region_profile() -> Optional[dict]:
    return REGION_PROFILES.get(SEARCH_REGION)


def _host(url: Optional[str]) -> str:
    return (urlparse(url or "").hostname or "").lower()


def infer_currency(url: Optional[str]) -> Optional[str]:
    """Guess a price currency from a store URL's domain when the page didn't state one."""
    host = _host(url)
    if not host:
        return None
    if any(host == h or host.endswith("." + h) for h in _INR_COM_HOSTS):
        return "INR"
    return _TLD_CURRENCY.get(host.rsplit(".", 1)[-1])


def is_local_store(url: Optional[str]) -> bool:
    """True when the URL belongs to a store that serves the configured search region."""
    profile = region_profile()
    host = _host(url)
    if not profile or not host:
        return False
    return host.rsplit(".", 1)[-1] in profile["tlds"] or any(
        host == s or host.endswith("." + s) for s in profile["stores"])


_SYMBOL_CURRENCY = {"₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "$": "USD", "usd": "USD"}
PRICE_RE = re.compile(
    r"(₹|Rs\.?|INR|\$|USD)\s?(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)", re.I)


# --------------------------------------------------------------------------- #
# cache (SQLite) -- a cache failure must never break a search
# --------------------------------------------------------------------------- #
def _cache_get(key: str, ttl: int) -> Optional[dict]:
    try:
        return db.cache_get(key, ttl)
    except Exception:
        return None


def _cache_set(key: str, value: dict) -> None:
    try:
        db.cache_set(key, value)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# web_search
# --------------------------------------------------------------------------- #
def _ddg(query: str, max_results: int) -> tuple[Optional[list], Optional[Exception]]:
    """One DuckDuckGo query. Searches run one at a time (it throttles bursts), with a retry."""
    error = None
    with _search_lock:
        for attempt in range(2):
            try:
                hits = DDGS().text(query, region=SEARCH_REGION, safesearch="moderate", max_results=max_results)
                time.sleep(1)
                return hits, None
            except Exception as e:  # network errors, throttling, "No results found"
                error = e
                time.sleep(2 + attempt * 2)
        time.sleep(1)
    return None, error


def _to_results(hits: list) -> list[dict]:
    return [
        {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": (h.get("body") or "")[:300]}
        for h in hits or [] if h.get("href")
    ]


def _search_uncached(query: str, max_results: int) -> dict:
    hits, error = _ddg(query, max_results)
    if hits is None:
        return {"query": query, "results": [], "error": f"Search failed: {error}. Try a simpler query."}
    results = _to_results(hits)

    # Generic queries mostly return foreign shops. When too few results are from stores that serve
    # the user's region, run one extra query restricted to the region's main stores.
    profile = region_profile()
    if profile and "site:" not in query.lower() and sum(is_local_store(r["url"]) for r in results) < 3:
        sites = " OR ".join(f"site:{s}" for s in profile["stores"][:4])
        extra, _ = _ddg(f"({sites}) {query}", max_results)
        known = {r["url"] for r in results}
        results += [r for r in _to_results(extra or []) if r["url"] not in known]

    for r in results:
        r["local_store"] = is_local_store(r["url"])
        r["currency_hint"] = infer_currency(r["url"])
    results.sort(key=lambda r: not r["local_store"])  # stable: local stores first
    return {"query": query, "region": SEARCH_REGION, "results": results[:max_results + 2]}


def web_search(query: str, max_results: int = 8) -> dict:
    """Search the public web (DuckDuckGo), favouring stores that serve the user's region."""
    max_results = max(1, min(int(max_results or 8), 10))
    key = f"search:{SEARCH_REGION}:{max_results}:{' '.join(query.lower().split())}"
    if (hit := _cache_get(key, SEARCH_TTL)) is not None:
        return {**hit, "cached": True}
    out = _search_uncached(query, max_results)
    if out.get("results"):
        _cache_set(key, out)
    return out


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
    # Size / colour lines help check that the requested variant exists.
    info["variant_lines"] = [l for l in lines if re.search(r"\b(sizes?|colou?rs?)\b", l, re.I) and len(l) <= 120][:6]
    # Listing / review pages ("best laptops under X") put many models + prices in text.
    info["price_mentions"] = price_lines
    guess_currency = None
    if info.get("price") is None and price_lines:
        # Prefer a price line that mentions the product itself over an unrelated "₹499 shipping" line.
        tokens = [w for w in re.findall(r"[a-z0-9]+", str(info.get("name", "")).lower()) if len(w) > 3]
        line = next((l for l in price_lines if any(w in l.lower() for w in tokens)), price_lines[0])
        m = PRICE_RE.search(line)
        info["price_guess"] = _to_float(m.group(2))
        guess_currency = _SYMBOL_CURRENCY.get(m.group(1).lower())
    if (info.get("price") is not None or info.get("price_guess") is not None) and not info.get("currency"):
        info["currency"] = guess_currency or infer_currency(url)
    return {k: v for k, v in info.items() if v not in (None, "", [])}


def extract_product(url: str) -> dict:
    """Fetch a URL and extract product name, price, rating and key specs (cached for a few hours)."""
    key = f"extract:{url}"
    if (hit := _cache_get(key, EXTRACT_TTL)) is not None:
        return {**hit, "cached": True}
    out = _extract_uncached(url)
    if "error" not in out:
        _cache_set(key, out)
    return out


def _extract_uncached(url: str) -> dict:
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


_PACK_RE = re.compile(
    r"\b(\d{1,3})\s*[- ]?(?:pack|pk|pcs|pieces|count|ct)\b|\b(?:pack|set|lot|bundle)\s+of\s+(\d{1,3})\b", re.I)


def parse_pack_size(text: str) -> int:
    """Items per listing: '(6-Pack, Black, XL)' -> 6, 'Pack of 3' -> 3, anything else -> 1."""
    m = _PACK_RE.search(text or "")
    n = int(next(g for g in m.groups() if g)) if m else 1
    return n if 1 <= n <= 100 else 1


def _variant_status(requested: Optional[str], text: str, flag: Any) -> Optional[str]:
    """confirmed | unconfirmed | unavailable for a requested size / colour (None if not requested)."""
    if not requested or not str(requested).strip():
        return None
    if flag is False:
        return "unavailable"
    if flag is True:
        return "confirmed"
    pattern = r"(?<![a-z0-9])" + re.escape(str(requested).strip().lower()) + r"(?![a-z0-9])"
    return "confirmed" if re.search(pattern, text.lower()) else "unconfirmed"


def compare_products(products: list[dict], budget: Optional[float] = None, priorities: Optional[list[str]] = None,
                     size: Optional[str] = None, color: Optional[str] = None, quantity: Optional[int] = 1) -> dict:
    """Deduplicate, filter and rank candidate products.

    score = 0.60 * requirement fit (LLM-judged, 0-10)
          + 0.25 * price efficiency (total cost vs budget, or unit price vs the other candidates)
          + 0.15 * hardware bonus (parsed RAM / GPU / VRAM / SSD)
          - 0.10 buying a bigger pack than needed, - 0.05 per unconfirmed size / colour

    Prices are compared per item: a 6-pack at 22 is 3.67 each, not 22. Out-of-stock listings and
    listings the page says lack the requested size / colour are excluded, never silently ranked.
    """
    budget = _to_float(budget)
    try:
        qty = max(1, int(quantity or 1))
    except (TypeError, ValueError):
        qty = 1
    ranked, over_budget, unavailable, seen = [], [], [], set()

    for p in products or []:
        name = _clean(str(p.get("name", "")))
        key = re.sub(r"[^a-z0-9]", "", name.lower())[:50]
        if not name or key in seen:
            continue
        seen.add(key)

        price = _to_float(p.get("price"))
        raw_specs = p.get("specs") or {}
        spec_text = " ".join(f"{k} {v}" for k, v in raw_specs.items()) if isinstance(raw_specs, dict) else str(raw_specs)
        text = f"{name} {spec_text}"
        parsed = parse_specs(text)

        try:
            fit = max(0.0, min(float(p.get("fit_score", 5)), 10.0))
        except (TypeError, ValueError):
            fit = 5.0

        notes: list[str] = []
        penalty = 0.0

        # --- availability: out of stock / requested variant missing -> excluded with the reason
        size_status = _variant_status(size, text, p.get("size_available"))
        color_status = _variant_status(color, text, p.get("color_available"))
        reason = ("out of stock" if p.get("in_stock") is False
                  else f"size {size} unavailable" if size_status == "unavailable"
                  else f"colour {color} unavailable" if color_status == "unavailable" else None)
        if reason:
            unavailable.append({"name": name, "price": price, "url": p.get("url"), "reason": reason})
            continue
        for label, status, wanted in (("size", size_status, size), ("colour", color_status, color)):
            if status == "unconfirmed":
                notes.append(f"{label} {wanted} not confirmed on the page")
                penalty += 0.05

        # --- pack size -> per-item price and what it costs to get `qty` items
        try:
            pack = int(p.get("pack_size") or 0) or parse_pack_size(name)
        except (TypeError, ValueError):
            pack = parse_pack_size(name)
        pack = max(pack, 1)
        unit_price = round(price / pack, 2) if price is not None else None
        total_cost = round(math.ceil(qty / pack) * price, 2) if price is not None else None
        if pack > 1:
            notes.append(f"{pack}-pack" + (f" ({unit_price:g} each)" if unit_price is not None else ""))
            if pack > qty:
                notes.append(f"you only need {qty}, so you'd pay for {pack}")
                penalty += 0.10

        if price is None:
            notes.append("price unknown")
        elif budget:
            if total_cost > budget * 1.05:
                over_budget.append({"name": name, "price": total_cost, "url": p.get("url")})
                continue
            if total_cost > budget:
                notes.append("slightly over budget")

        ranked.append({
            "name": name, "price": price, "currency": p.get("currency") or infer_currency(p.get("url")),
            "pack_size": pack, "unit_price": unit_price, "total_cost": total_cost,
            "url": p.get("url"), "fit_score": fit, "hardware_bonus": _hardware_bonus(parsed),
            "size_status": size_status, "color_status": color_status, "_penalty": penalty,
            "parsed_specs": parsed, "specs": raw_specs, "notes": notes,
        })

    # --- price efficiency (needs all candidates when there is no budget)
    max_unit = max((r["unit_price"] for r in ranked if r["unit_price"]), default=None)
    for r in ranked:
        if r["total_cost"] is None:
            price_score = 0.5
        elif budget:
            price_score = max(0.0, 1 - 0.5 * r["total_cost"] / budget)
        elif max_unit:
            price_score = 1 - 0.5 * r["unit_price"] / max_unit
        else:
            price_score = 0.5
        r["price_score"] = round(price_score, 3)
        r["score"] = round(max(0.0, 0.60 * r["fit_score"] / 10 + 0.25 * price_score
                               + 0.15 * r["hardware_bonus"] - r.pop("_penalty")), 4)

    ranked.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return {
        "budget": budget,
        "quantity": qty,
        "size": size,
        "color": color,
        "priorities": priorities or [],
        "ranked": ranked,
        "over_budget_excluded": over_budget,
        "unavailable_excluded": unavailable,
        "method": ("score = 0.60*fit + 0.25*price_efficiency (per item) + 0.15*hardware_bonus, minus penalties "
                   "for oversized packs and unconfirmed size/colour; >5% over budget, out-of-stock and "
                   "missing-variant listings excluded"),
    }
