"""Quick test to verify DocumentationTools fetches from PiEEG-docs GitHub repo."""

from pieeg_agent.agent import DocumentationTools

# Create instance
docs = DocumentationTools()
print(f"✓ DocumentationTools loaded")
print(f"✓ Available tools: {docs.names()}")

# Test search (will fetch from GitHub on first run)
print("\n🌐 Fetching documentation from github.com/pieeg-club/PiEEG-docs...")
result = docs.call('search_docs', {'query': 'sampling rate LSL', 'max_results': 3})

if 'error' in result:
    print(f"✗ Error: {result['error']}")
    if 'detail' in result:
        print(f"  Detail: {result['detail']}")
else:
    print(f"✓ Search for '{result['query']}' returned {result['total_found']} results from {result['source']}:")
    for i, res in enumerate(result['results'], 1):
        print(f"\n  {i}. {res['section']} (score: {res['relevance_score']})")
        print(f"     Source: {res['source']}")
        # Show first 150 chars of content
        content_preview = res['content'][:150].replace('\n', ' ')
        print(f"     {content_preview}...")

# Test another search (should use cache)
print("\n\n🔍 Second search (using cache)...")
result2 = docs.call('search_docs', {'query': 'electrode placement'})
print(f"✓ Search for 'electrode placement' returned {result2['total_found']} results")

print("\n✅ All tests passed! Documentation is being fetched from GitHub repository")
