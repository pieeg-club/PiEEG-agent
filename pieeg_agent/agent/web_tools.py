"""Web tools — curated access to BCI research and PiEEG resources.

This toolset gives the copilot read access to trusted scientific sources and
PiEEG ecosystem web resources. Designed for exploration and learning, not
unrestricted web access. All fetches are confined to an allowlist of:

* Wikipedia (neuroscience, BCI, EEG topics)
* Trusted scientific publishers and preprint servers
* PiEEG official websites and documentation
* PiEEG social media profiles

Unlike general-purpose web search, this is **curated**: the agent can only
fetch from pre-approved domains known to contain reliable BCI/neuroscience
content or official PiEEG information.

Tools:
* ``search_web`` — Search Wikipedia and PubMed for BCI/neuroscience topics
* ``fetch_url`` — Fetch content from a trusted domain (domain must be allowlisted)
* ``list_trusted_sources`` — Show all currently allowlisted domains

This is intentionally read-only and rate-limited to prevent abuse.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import urllib.request
from typing import Any

from .tools import Tool, _spec

logger = logging.getLogger("pieeg.agent.web")

# ── Trusted source allowlist ────────────────────────────────────────────

WIKIPEDIA_DOMAINS = [
    "en.wikipedia.org",
    "en.m.wikipedia.org",
]

# Major scientific publishers and archives (peer-reviewed content)
SCIENTIFIC_SOURCES = [
    # Preprint servers
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
    "psyarxiv.com",
    
    # Open access publishers
    "plos.org",
    "plosone.org",
    "frontiersin.org",
    "mdpi.com",
    "elifesciences.org",
    "nature.com",
    "sciencedirect.com",
    
    # Databases and aggregators
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "scholar.google.com",
    "semanticscholar.org",
    
    # Neuroscience-specific
    "jneurosci.org",
    "neuron.org",
    "journals.physiology.org",
    "eneuro.org",
    
    # BCI/neuroimaging-specific
    "journals.sagepub.com",  # Journal of Cognitive Neuroscience, etc.
    "iopscience.iop.org",    # Journal of Neural Engineering
    "springer.com",
    "sciencedirect.com",     # Various BCI journals
    
    # Standards and protocols
    "biosemi.com",
    "brainproducts.com",
    "cognionics.com",
    "openbci.com",
]

# PiEEG ecosystem websites
PIEEG_DOMAINS = [
    # Official sites
    "pieeg.com",
    "www.pieeg.com",
    
    # Documentation
    "docs.pieeg.com",
    "github.com/pieeg-club",
    "raw.githubusercontent.com/pieeg-club",
]

# Combine all trusted domains
TRUSTED_DOMAINS = set(WIKIPEDIA_DOMAINS + SCIENTIFIC_SOURCES + PIEEG_DOMAINS)

# Rate limiting (prevent abuse)
_last_fetch_time = 0.0
MIN_FETCH_INTERVAL = 1.0  # seconds between fetches


class WebTools:
    """Curated web access toolset for BCI research and PiEEG resources."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._register_all()

    # ── registry surface (mirrors NeuralTools / ActuatorTools) ──────────
    def specs(self):
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, name: str, arguments: dict | None = None) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}", "available": self.names()}
        try:
            return tool.handler(arguments or {})
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── registration ────────────────────────────────────────────────────
    def _add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def _register_all(self) -> None:
        self._add(Tool(
            _spec(
                "search_web",
                "Search Wikipedia or PubMed for BCI, neuroscience, and EEG topics. "
                "Returns titles and snippets with URLs. Use this to find general "
                "knowledge (Wikipedia) or research papers (PubMed). For PiEEG-specific "
                "questions, use search_docs instead.",
                {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'P300 event-related potential', "
                                     "'brain-computer interface')",
                    },
                    "source": {
                        "type": "string",
                        "enum": ["wikipedia", "pubmed"],
                        "description": "Which source to search (default: wikipedia)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (1-10, default: 5)",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                required=["query"],
            ),
            self._search_web,
        ))

        self._add(Tool(
            _spec(
                "fetch_url",
                "Fetch text content from a trusted URL. The domain must be on the "
                "allowlist (Wikipedia, scientific publishers, PiEEG sites). Returns "
                "the page text (HTML stripped). Use list_trusted_sources to see "
                "which domains are allowed.",
                {
                    "url": {
                        "type": "string",
                        "description": "Full URL to fetch (must be from a trusted domain)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max characters to return (default: 4000)",
                        "minimum": 100,
                        "maximum": 10000,
                    },
                },
                required=["url"],
            ),
            self._fetch_url,
        ))

        self._add(Tool(
            _spec(
                "list_trusted_sources",
                "List all domains the agent can fetch from. Shows Wikipedia, "
                "scientific publishers, and PiEEG ecosystem sites. Use this to "
                "see what sources are available for fetch_url.",
                {
                    "category": {
                        "type": "string",
                        "enum": ["all", "wikipedia", "scientific", "pieeg"],
                        "description": "Filter by category (default: all)",
                    },
                },
            ),
            self._list_trusted_sources,
        ))

    # ── handlers ────────────────────────────────────────────────────────
    def _search_web(self, args: dict) -> dict:
        """Search Wikipedia or PubMed for BCI/neuroscience topics."""
        query = args.get("query", "").strip()
        if not query:
            return {"error": "query cannot be empty"}

        source = args.get("source", "wikipedia").lower()
        limit = min(max(args.get("limit", 5), 1), 10)

        if source == "wikipedia":
            return self._search_wikipedia(query, limit)
        elif source == "pubmed":
            return self._search_pubmed(query, limit)
        else:
            return {"error": f"unknown source {source!r}, use 'wikipedia' or 'pubmed'"}

    def _search_wikipedia(self, query: str, limit: int) -> dict:
        """Search Wikipedia via the MediaWiki API."""
        try:
            _rate_limit()
            
            # Use Wikipedia's opensearch API
            encoded = urllib.parse.quote(query)
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded}&limit={limit}&format=json"
            
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PiEEG-Agent/1.0 (Educational BCI Assistant)"}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()
            
            import json
            results = json.loads(data)
            
            # Wikipedia opensearch returns: [query, [titles], [snippets], [urls]]
            if len(results) < 4:
                return {"results": [], "count": 0, "source": "wikipedia"}
            
            titles = results[1]
            snippets = results[2]
            urls = results[3]
            
            items = []
            for i in range(min(len(titles), limit)):
                items.append({
                    "title": titles[i],
                    "snippet": snippets[i] if i < len(snippets) else "",
                    "url": urls[i] if i < len(urls) else "",
                })
            
            return {
                "results": items,
                "count": len(items),
                "source": "wikipedia",
                "query": query,
            }
            
        except Exception as exc:
            logger.exception("Wikipedia search failed")
            return {"error": f"Wikipedia search failed: {exc}", "results": []}

    def _search_pubmed(self, query: str, limit: int) -> dict:
        """Search PubMed via the E-utilities API."""
        try:
            _rate_limit()
            
            # Use NCBI E-utilities esearch
            encoded = urllib.parse.quote(query)
            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
                f"db=pubmed&term={encoded}&retmax={limit}&retmode=json"
            )
            
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PiEEG-Agent/1.0 (Educational BCI Assistant)"}
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()
            
            import json
            results = json.loads(data)
            
            id_list = results.get("esearchresult", {}).get("idlist", [])
            
            if not id_list:
                return {"results": [], "count": 0, "source": "pubmed"}
            
            # Fetch summaries for the IDs
            ids_str = ",".join(id_list[:limit])
            summary_url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
                f"db=pubmed&id={ids_str}&retmode=json"
            )
            
            time.sleep(0.4)  # NCBI rate limit: max 3 requests/sec
            
            with urllib.request.urlopen(summary_url, timeout=10) as response:
                summary_data = response.read()
            
            summaries = json.loads(summary_data)
            
            items = []
            for pmid in id_list[:limit]:
                doc = summaries.get("result", {}).get(pmid, {})
                if not doc:
                    continue
                
                title = doc.get("title", "")
                authors = doc.get("authors", [])
                author_str = ", ".join([a.get("name", "") for a in authors[:3]])
                if len(authors) > 3:
                    author_str += " et al."
                
                journal = doc.get("source", "")
                pub_date = doc.get("pubdate", "")
                
                snippet = f"{author_str}. {journal}. {pub_date}."
                
                items.append({
                    "title": title,
                    "snippet": snippet,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "pmid": pmid,
                })
            
            return {
                "results": items,
                "count": len(items),
                "source": "pubmed",
                "query": query,
            }
            
        except Exception as exc:
            logger.exception("PubMed search failed")
            return {"error": f"PubMed search failed: {exc}", "results": []}

    def _fetch_url(self, args: dict) -> dict:
        """Fetch text content from a trusted URL."""
        url = args.get("url", "").strip()
        if not url:
            return {"error": "url cannot be empty"}
        
        max_chars = min(max(args.get("max_chars", 4000), 100), 10000)
        
        # Check if domain is trusted
        try:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            # Strip www. for comparison
            domain_stripped = domain.replace("www.", "")
            
            is_trusted = False
            for trusted in TRUSTED_DOMAINS:
                trusted_stripped = trusted.replace("www.", "")
                if domain_stripped == trusted_stripped or domain_stripped.endswith("." + trusted_stripped):
                    is_trusted = True
                    break
            
            if not is_trusted:
                return {
                    "error": f"Domain {domain} is not on the trusted allowlist. "
                            "Use list_trusted_sources to see allowed domains.",
                }
            
            _rate_limit()
            
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "PiEEG-Agent/1.0 (Educational BCI Assistant)"}
            )
            
            with urllib.request.urlopen(req, timeout=15) as response:
                content_type = response.headers.get("Content-Type", "")
                raw = response.read()
            
            # Decode content
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
                text = raw.decode(charset, errors="replace")
            else:
                text = raw.decode("utf-8", errors="replace")
            
            # Strip HTML if present
            if "text/html" in content_type or "<html" in text[:500].lower():
                text = _strip_html(text)
            
            # Truncate
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[... truncated at {max_chars} chars]"
            
            return {
                "url": url,
                "content": text,
                "chars": len(text),
                "domain": domain,
            }
            
        except Exception as exc:
            logger.exception("URL fetch failed")
            return {"error": f"Failed to fetch {url}: {exc}"}

    def _list_trusted_sources(self, args: dict) -> dict:
        """List all trusted domains the agent can fetch from."""
        category = args.get("category", "all").lower()
        
        if category == "all":
            sources = {
                "wikipedia": sorted(WIKIPEDIA_DOMAINS),
                "scientific": sorted(SCIENTIFIC_SOURCES),
                "pieeg": sorted(PIEEG_DOMAINS),
            }
            total = len(TRUSTED_DOMAINS)
        elif category == "wikipedia":
            sources = {"wikipedia": sorted(WIKIPEDIA_DOMAINS)}
            total = len(WIKIPEDIA_DOMAINS)
        elif category == "scientific":
            sources = {"scientific": sorted(SCIENTIFIC_SOURCES)}
            total = len(SCIENTIFIC_SOURCES)
        elif category == "pieeg":
            sources = {"pieeg": sorted(PIEEG_DOMAINS)}
            total = len(PIEEG_DOMAINS)
        else:
            return {"error": f"unknown category {category!r}, use 'all', 'wikipedia', 'scientific', or 'pieeg'"}
        
        return {
            "category": category,
            "sources": sources,
            "total": total,
        }


# ── helpers ─────────────────────────────────────────────────────────────

def _rate_limit() -> None:
    """Simple rate limiter: ensure MIN_FETCH_INTERVAL between requests."""
    global _last_fetch_time
    now = time.monotonic()
    elapsed = now - _last_fetch_time
    if elapsed < MIN_FETCH_INTERVAL:
        time.sleep(MIN_FETCH_INTERVAL - elapsed)
    _last_fetch_time = time.monotonic()


def _strip_html(html: str) -> str:
    """Rudimentary HTML → plain text converter.
    
    Removes tags, scripts, styles, and excessive whitespace. Good enough for
    basic content extraction. For production, consider using BeautifulSoup or lxml.
    """
    # Remove script and style elements
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", text)
    
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    
    return text.strip()
