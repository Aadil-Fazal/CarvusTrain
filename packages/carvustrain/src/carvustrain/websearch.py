"""Web search learning — fetch and ingest knowledge from the open web.

Uses only the Python standard library (urllib) so CarvusTrain stays
zero-dependency by default. Optional BeautifulSoup-like parsing is
implemented with simple HTML tag stripping.
"""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

USER_AGENT = (
    "CarvusTrain/1.2 (+https://github.com/Aadil-Fazal/CarvusTrain; educational learning bot)"
)
DEFAULT_TIMEOUT = 15
MAX_SNIPPETS = 8
MAX_CONTENT_CHARS = 12000


class WebSearchError(Exception):
    """Raised when a web search or page fetch fails."""


def _http_get(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """GET a URL and return decoded text body."""
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json,*/*"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        raise WebSearchError(f"HTTP {e.code} for {url}") from e
    except urllib.error.URLError as e:
        raise WebSearchError(f"Network error fetching {url}: {e.reason}") from e
    except Exception as e:
        raise WebSearchError(f"Failed to fetch {url}: {e}") from e


def _strip_html(raw: str) -> str:
    """Remove script/style tags and HTML markup; collapse whitespace."""
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _sentences_from_text(text: str, max_sentences: int = 40) -> List[str]:
    """Split cleaned text into informative sentence-like facts."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    facts: List[str] = []
    for part in parts:
        clean = part.strip()
        if len(clean) < 40:
            continue
        if len(clean) > 500:
            clean = clean[:497] + "..."
        # Skip nav/boilerplate-ish lines
        low = clean.lower()
        if any(x in low for x in ("cookie", "sign in", "subscribe", "javascript enabled", "privacy policy")):
            continue
        facts.append(clean)
        if len(facts) >= max_sentences:
            break
    return facts


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo Instant Answer API (no API key required).

    Returns a list of dicts with keys: title, url, snippet.
    """
    params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1, "skip_disambig": 1})
    url = f"https://api.duckduckgo.com/?{params}"
    raw = _http_get(url)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise WebSearchError(f"Invalid JSON from DuckDuckGo: {e}") from e

    results: List[Dict[str, str]] = []

    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    heading = (data.get("Heading") or query).strip()
    if abstract:
        results.append({"title": heading, "url": abstract_url, "snippet": abstract})

    for topic in data.get("RelatedTopics") or []:
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and "Topics" in topic:
            for sub in topic.get("Topics") or []:
                if len(results) >= max_results:
                    break
                text = (sub.get("Text") or "").strip()
                first_url = (sub.get("FirstURL") or "").strip()
                if text:
                    results.append({"title": text[:80], "url": first_url, "snippet": text})
        elif isinstance(topic, dict):
            text = (topic.get("Text") or "").strip()
            first_url = (topic.get("FirstURL") or "").strip()
            if text:
                results.append({"title": text[:80], "url": first_url, "snippet": text})

    # Fallback: Wikipedia OpenSearch if DDG returned nothing useful
    if not results:
        results.extend(_wikipedia_opensearch(query, max_results=max_results))

    return results[:max_results]


def _wikipedia_opensearch(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fallback search via Wikipedia OpenSearch API."""
    params = urllib.parse.urlencode(
        {"action": "opensearch", "search": query, "limit": max_results, "namespace": 0, "format": "json"}
    )
    url = f"https://en.wikipedia.org/w/api.php?{params}"
    raw = _http_get(url)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list) or len(data) < 4:
        return []
    titles, descriptions, urls = data[1], data[2], data[3]
    out: List[Dict[str, str]] = []
    for title, desc, link in zip(titles, descriptions, urls):
        out.append({"title": title, "url": link, "snippet": desc or title})
    return out


def fetch_page_facts(url: str, max_facts: int = 20) -> List[str]:
    """Download a page and extract factual sentence snippets."""
    if not url:
        return []
    body = _http_get(url)
    if len(body) > MAX_CONTENT_CHARS * 3:
        body = body[: MAX_CONTENT_CHARS * 3]
    text = _strip_html(body)
    if len(text) > MAX_CONTENT_CHARS:
        text = text[:MAX_CONTENT_CHARS]
    return _sentences_from_text(text, max_sentences=max_facts)


def learn_from_search(
    query: str,
    *,
    max_results: int = 5,
    fetch_pages: bool = True,
    max_facts_per_page: int = 12,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Search the web for ``query`` and return (facts, source_metadata).

    Facts are ready to ingest into a KnowledgeBase via ``add_facts``.
    """
    results = search_duckduckgo(query, max_results=max_results)
    facts: List[str] = []
    sources: List[Dict[str, Any]] = []

    for item in results:
        snippet = (item.get("snippet") or "").strip()
        title = (item.get("title") or "").strip()
        link = (item.get("url") or "").strip()
        if snippet:
            fact = snippet if snippet.endswith((".", "!", "?")) else snippet + "."
            facts.append(f"{title}: {fact}" if title and title not in fact else fact)
            sources.append({"category": "web", "source": link or "duckduckgo", "query": query, "title": title})

        if fetch_pages and link and link.startswith("http"):
            try:
                page_facts = fetch_page_facts(link, max_facts=max_facts_per_page)
                for pf in page_facts:
                    facts.append(pf)
                    sources.append({"category": "web", "source": link, "query": query, "title": title})
            except WebSearchError:
                continue

    # Deduplicate while preserving order
    seen = set()
    unique_facts: List[str] = []
    unique_sources: List[Dict[str, Any]] = []
    for fact, meta in zip(facts, sources):
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_facts.append(fact)
        unique_sources.append(meta)

    return unique_facts[: MAX_SNIPPETS * max_facts_per_page], unique_sources


class WebLearner:
    """High-level helper that searches the web and feeds a KnowledgeBase."""

    def __init__(self, knowledge_base: Any) -> None:
        self.knowledge_base = knowledge_base
        self.last_sources: List[Dict[str, Any]] = []

    def learn(
        self,
        query: str,
        *,
        max_results: int = 5,
        fetch_pages: bool = True,
    ) -> int:
        """Search and ingest facts. Returns number of facts added."""
        facts, sources = learn_from_search(query, max_results=max_results, fetch_pages=fetch_pages)
        before = len(self.knowledge_base)
        for fact, meta in zip(facts, sources):
            self.knowledge_base.add_fact(fact, meta=meta)
        self.last_sources = sources
        return len(self.knowledge_base) - before

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Return raw search results without ingesting."""
        return search_duckduckgo(query, max_results=max_results)
