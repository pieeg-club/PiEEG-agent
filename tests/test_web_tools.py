"""Web tools tests — curated web access for BCI research.

Basic functionality tests for Wikipedia/PubMed search and trusted URL fetching.
These tests verify the tool interface and allowlist logic without requiring
actual network access (mocked where appropriate).
"""

from pieeg_agent.agent.web_tools import TRUSTED_DOMAINS, WebTools


def test_specs_advertise_web_tools():
    """Web tools spec should list search_web, fetch_url, list_trusted_sources."""
    tools = WebTools()
    names = {s.name for s in tools.specs()}
    assert names == {
        "search_web",
        "fetch_url",
        "list_trusted_sources",
    }
    # Every spec advertises an object schema
    for spec in tools.specs():
        assert spec.input_schema["type"] == "object"


def test_list_trusted_sources_returns_all_categories():
    """list_trusted_sources should return Wikipedia, scientific, and PiEEG domains."""
    tools = WebTools()
    
    # All categories
    out = tools.call("list_trusted_sources", {"category": "all"})
    assert "sources" in out
    assert "wikipedia" in out["sources"]
    assert "scientific" in out["sources"]
    assert "pieeg" in out["sources"]
    assert out["total"] == len(TRUSTED_DOMAINS)
    
    # Wikipedia only
    wiki_out = tools.call("list_trusted_sources", {"category": "wikipedia"})
    assert "en.wikipedia.org" in wiki_out["sources"]["wikipedia"]
    
    # Scientific only
    sci_out = tools.call("list_trusted_sources", {"category": "scientific"})
    assert "pubmed.ncbi.nlm.nih.gov" in sci_out["sources"]["scientific"]
    assert "arxiv.org" in sci_out["sources"]["scientific"]
    
    # PiEEG only
    pieeg_out = tools.call("list_trusted_sources", {"category": "pieeg"})
    assert "pieeg.com" in pieeg_out["sources"]["pieeg"]


def test_search_web_requires_query():
    """search_web should return error if query is empty."""
    tools = WebTools()
    out = tools.call("search_web", {"query": ""})
    assert "error" in out


def test_search_web_rejects_unknown_source():
    """search_web should only accept 'wikipedia' or 'pubmed' as source."""
    tools = WebTools()
    out = tools.call("search_web", {"query": "test", "source": "google"})
    assert "error" in out
    assert "unknown source" in out["error"].lower()


def test_fetch_url_requires_trusted_domain():
    """fetch_url should reject URLs from non-allowlisted domains."""
    tools = WebTools()
    
    # Untrusted domain should be rejected
    out = tools.call("fetch_url", {"url": "https://example.com/test"})
    assert "error" in out
    assert "not on the trusted allowlist" in out["error"]
    
    # Empty URL should be rejected
    out_empty = tools.call("fetch_url", {"url": ""})
    assert "error" in out_empty


def test_trusted_domains_include_key_sources():
    """Verify the allowlist includes essential BCI/neuroscience sources."""
    # Wikipedia
    assert "en.wikipedia.org" in TRUSTED_DOMAINS
    
    # Major scientific sources
    assert "pubmed.ncbi.nlm.nih.gov" in TRUSTED_DOMAINS
    assert "arxiv.org" in TRUSTED_DOMAINS
    assert "biorxiv.org" in TRUSTED_DOMAINS
    assert "nature.com" in TRUSTED_DOMAINS
    assert "frontiersin.org" in TRUSTED_DOMAINS
    
    # PiEEG ecosystem
    assert "pieeg.com" in TRUSTED_DOMAINS
    assert "github.com/pieeg-club" in TRUSTED_DOMAINS


def test_unknown_tool_returns_error():
    """Calling a nonexistent tool should return error dict, not raise."""
    tools = WebTools()
    out = tools.call("nonexistent_tool")
    assert "error" in out
    assert "available" in out
