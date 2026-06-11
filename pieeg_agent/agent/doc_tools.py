"""Documentation tools — the agent's access to PiEEG ecosystem knowledge.

This toolset gives the copilot read access to the broader PiEEG ecosystem docs:
hardware specs, server configuration, electrode placement, software architecture,
and troubleshooting guides. Docs are fetched from the PiEEG-docs GitHub repository
(https://github.com/pieeg-club/PiEEG-docs) and cached locally for subsequent queries.

Unlike the agent's own docstrings/README (which describe the *copilot* itself),
this is **product knowledge**: how to set up the headset, configure pieeg-server,
understand the data pipeline, etc.

The single tool ``search_docs`` accepts a natural-language query and returns
relevant documentation chunks with source attribution. Search is keyword-based
(fast, no dependencies) and can be upgraded to embeddings-based RAG later.

This is intentionally read-only and side-effect-free, just like NeuralTools.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .tools import Tool, _spec

# Documentation files to fetch from GitHub repository
DOCS_BASE_URL = "https://raw.githubusercontent.com/pieeg-club/PiEEG-docs/main/pages"
DOCS_FILES = [
    # Hardware docs
    "hardware/index.mdx",
    "hardware/signal-quality/noise-elimination.mdx",
    "hardware/accessories/electrodes.mdx",
    "hardware/accessories/cap.mdx",
    "hardware/accessories/power-bank.mdx",
    "hardware/safety/index.mdx",
    # Software - Getting Started
    "software/getting-started/installation.mdx",
    "software/getting-started/quick-start.mdx",
    "software/getting-started/configuration.mdx",
    # Software - Features
    "software/features/server.mdx",
    "software/features/dashboard.mdx",
    "software/features/recording.mdx",
    "software/features/detectors.mdx",
    # Software - API
    "software/api/websocket.mdx",
    "software/api/data-format.mdx",
    # Software - Integrations
    "software/integrations/lsl.mdx",
    "software/integrations/notebooks.mdx",
    "software/integrations/webhooks.mdx",
    "software/integrations/vrchat-osc.mdx",
    # Software - Reference
    "software/reference/architecture.mdx",
    "software/reference/troubleshooting.mdx",
    "software/reference/authentication.mdx",
    "software/reference/security.mdx",
]

# Cache directory for fetched docs
CACHE_DIR = Path.home() / ".pieeg-agent" / "docs-cache"


class DocumentationTools:
    """Read-only documentation toolset for PiEEG ecosystem knowledge."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._doc_cache: list[dict] | None = None  # lazy-loaded
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
                "search_docs",
                "Search PiEEG ecosystem documentation from the official GitHub "
                "repository (hardware setup, server configuration, electrode "
                "placement, software architecture, troubleshooting). Returns "
                "relevant doc sections with source attribution. Use when the user "
                "asks about device setup, headset specs, pieeg-server config, LSL "
                "streaming, or component architecture.",
                {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g. "
                        "'electrode placement', 'LSL not streaming', 'sampling rate')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max doc sections to return (default 3).",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                required=["query"],
            ),
            self._search_docs,
        ))

    # ── handler ──────────────────────────────────────────────────────────
    def _search_docs(self, args: dict) -> dict:
        """Search the GitHub documentation via keyword matching."""
        query = args.get("query", "").strip()
        if not query:
            return {"error": "query cannot be empty"}

        max_results = args.get("max_results", 3)
        max_results = max(1, min(10, max_results))

        # Lazy-load doc cache on first search
        if self._doc_cache is None:
            try:
                self._doc_cache = self._index_docs()
            except Exception as e:
                return {
                    "error": "failed to fetch documentation",
                    "detail": f"Could not access GitHub repository: {e}",
                    "suggestion": "Check internet connection or try again later"
                }

        if not self._doc_cache:
            return {
                "error": "no documentation found",
                "detail": "Failed to fetch any files from PiEEG-docs repository",
            }

        # Simple keyword search with scoring
        results = self._keyword_search(self._doc_cache, query, max_results)

        return {
            "query": query,
            "total_found": len(results),
            "results": results,
            "source": "github.com/pieeg-club/PiEEG-docs",
        }

    # ── indexing / search logic ──────────────────────────────────────────
    def _index_docs(self) -> list[dict]:
        """Fetch and chunk all documentation files from GitHub."""
        chunks: list[dict] = []
        
        for file_path in DOCS_FILES:
            url = f"{DOCS_BASE_URL}/{file_path}"
            try:
                content = self._fetch_page(url)
                if content:
                    file_chunks = self._chunk_mdx_content(content, file_path)
                    chunks.extend(file_chunks)
            except Exception:
                # Skip files that fail to fetch
                continue
        
        return chunks

    def _fetch_page(self, url: str) -> str:
        """Fetch a documentation file from GitHub, with caching."""
        # Try cache first
        cache_file = self._get_cache_path(url)
        if cache_file.exists():
            try:
                return cache_file.read_text(encoding="utf-8")
            except Exception:
                pass
        
        # Fetch from GitHub raw URL
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8")
            
            # Cache for next time
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(content, encoding="utf-8")
            
            return content
        except Exception:
            return ""

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        # Convert URL to safe filename
        safe_name = url.replace(DOCS_BASE_URL + "/", "").replace("/", "_")
        return CACHE_DIR / safe_name

    def _chunk_mdx_content(self, mdx: str, source_path: str) -> list[dict]:
        """Split MDX file by headers into searchable chunks."""
        # Remove JSX components and extract markdown content
        # Simple approach: remove import statements and JSX tags
        text = re.sub(r'^import\s+.*$', '', mdx, flags=re.MULTILINE)
        text = re.sub(r'<[^>]+>', '', text)  # Remove JSX tags
        
        chunks: list[dict] = []
        current_section = source_path.replace(".mdx", "").replace("/", " > ")
        current_content: list[str] = []

        for line in text.split("\n"):
            if line.startswith("#"):
                # Save previous section
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if len(content_text) > 50:  # Only include substantial chunks
                        chunks.append({
                            "source": source_path,
                            "section": current_section,
                            "content": content_text,
                        })
                # Start new section
                current_section = line.lstrip("#").strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            content_text = "\n".join(current_content).strip()
            if len(content_text) > 50:
                chunks.append({
                    "source": source_path,
                    "section": current_section,
                    "content": content_text,
                })

        return chunks

    def _keyword_search(
        self,
        docs: list[dict],
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Simple keyword-based search with term-frequency scoring."""
        query_terms = set(query.lower().split())
        scored: list[tuple[int, dict]] = []

        for chunk in docs:
            text = (chunk["section"] + " " + chunk["content"]).lower()
            # Score: count how many query terms appear
            score = sum(1 for term in query_terms if term in text)
            if score > 0:
                scored.append((score, chunk))

        # Sort by relevance (highest score first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top results with score included
        return [
            {
                "source": chunk["source"],
                "section": chunk["section"],
                "content": chunk["content"],
                "relevance_score": score,
            }
            for score, chunk in scored[:max_results]
        ]
