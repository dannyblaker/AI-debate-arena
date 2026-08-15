"""Topic research: Wikipedia articles + web pages found via DuckDuckGo.

Every source is fetched best-effort — any individual failure (or a completely
offline machine) degrades gracefully to fewer/zero sources, in which case the
debaters fall back on the model's own knowledge.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Callable

import requests
import trafilatura

from .config import MAX_WEB_SOURCES, MAX_WIKI_SOURCES
from .models_registry import StopRequested

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
}
MAX_DOC_CHARS = 16_000

# Search results that are ads or click-tracking redirects, not articles.
AD_URL_MARKERS = ("bing.com/aclick", "duckduckgo.com/y.js", "/aclk?",
                  "googleadservices", "doubleclick.net", "syndicatedsearch")


def _clean_title(raw_title: str, html: str) -> str:
    """Prefer the page's own <title>; otherwise strip the search-engine
    breadcrumb junk some results carry ('www.site.com › news › 2026Actual
    headline...')."""
    try:
        meta = trafilatura.extract_metadata(html)
        if meta and meta.title:
            return meta.title.strip()[:150]
    except Exception:
        pass
    t = re.sub(r"^(?:[\w.-]+\s*›\s*)+", "", raw_title or "").strip()
    t = re.sub(r"^\d{4}(?:/\d{2}/\d{2})?\s*", "", t)
    return (t or raw_title or "")[:150]


@dataclass
class Doc:
    title: str
    url: str
    text: str


def _wikipedia_docs(topic: str, emit_status: Callable[[str], None]) -> list[Doc]:
    docs = []
    api = "https://en.wikipedia.org/w/api.php"
    r = requests.get(api, params={
        "action": "query", "list": "search", "srsearch": topic,
        "srlimit": MAX_WIKI_SOURCES, "format": "json",
    }, headers=HEADERS, timeout=15)
    r.raise_for_status()
    for hit in r.json().get("query", {}).get("search", []):
        title = hit["title"]
        try:
            r2 = requests.get(api, params={
                "action": "query", "prop": "extracts", "explaintext": 1,
                "titles": title, "format": "json",
            }, headers=HEADERS, timeout=15)
            r2.raise_for_status()
            pages = r2.json().get("query", {}).get("pages", {})
            for page in pages.values():
                text = (page.get("extract") or "").strip()
                if len(text) > 400:
                    url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
                    docs.append(Doc(f"Wikipedia: {title}", url, text[:MAX_DOC_CHARS]))
        except Exception as e:
            emit_status(f"Skipped Wikipedia article '{title}': {e}")
    return docs


def _web_docs(topic: str, emit_status: Callable[[str], None],
              stop: threading.Event) -> list[Doc]:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(topic, max_results=MAX_WEB_SOURCES))
    except Exception as e:
        emit_status(f"Web search unavailable: {e}")
        return []
    docs = []
    for res in results:
        if stop.is_set():
            raise StopRequested()
        url = res.get("href") or res.get("url") or ""
        title = res.get("title") or url
        if not url:
            continue
        if any(marker in url for marker in AD_URL_MARKERS):
            emit_status(f"Skipped (ad/redirect): {title[:80]}")
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            r.raise_for_status()
            text = trafilatura.extract(r.text, url=url) or ""
            if len(text) > 500:
                docs.append(Doc(_clean_title(title, r.text), url,
                                text.strip()[:MAX_DOC_CHARS]))
            else:
                emit_status(f"Skipped (too little text): {title}")
        except Exception:
            emit_status(f"Skipped (fetch failed): {title}")
    return docs


def gather(topic: str, emit: Callable[..., None], stop: threading.Event) -> list[Doc]:
    """Collect research documents. `emit(type, **data)` receives
    'research_source' and 'status' events as sources come in."""
    docs: list[Doc] = []

    emit("status", message="Searching Wikipedia…")
    try:
        wiki = _wikipedia_docs(topic, lambda msg: emit("status", message=msg))
    except Exception as e:
        wiki = []
        emit("status", message=f"Wikipedia unavailable: {e}")
    for d in wiki:
        docs.append(d)
        emit("research_source", title=d.title, url=d.url, chars=len(d.text))

    if stop.is_set():
        raise StopRequested()

    emit("status", message="Searching the web…")
    for d in _web_docs(topic, lambda msg: emit("status", message=msg), stop):
        docs.append(d)
        emit("research_source", title=d.title, url=d.url, chars=len(d.text))

    return docs
