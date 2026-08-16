"""Single-purpose tools and schemas for the LLM runtime."""

from ddgs import DDGS


def web_search(query: str, max_results: int = 3) -> dict:
    """Search DuckDuckGo and return a compact text summary plus structured sources."""
    if not query or not query.strip():
        return {
            "content": "No search query provided.",
            "sources": [],
            "query": "",
        }

    try:
        results = list(DDGS().text(query.strip(), max_results=max_results))
        if not results:
            return {
                "content": "No web results found.",
                "sources": [],
                "query": query.strip(),
            }

        snippets = []
        sources = []
        for i, item in enumerate(results, 1):
            title = item.get("title", "")
            body = item.get("body", "")
            url = item.get("href") or item.get("url") or ""
            source_label = f" (Source: {url})" if url else ""
            snippets.append(f"[{i}] {title}{source_label}: {body}")
            if url:
                sources.append({"title": title, "url": url})

        return {
            "content": "\n\n".join(snippets),
            "sources": sources,
            "query": query.strip(),
        }
    except Exception as error:
        return {
            "content": f"Web search failed: {error}",
            "sources": [],
            "query": query.strip() if query else "",
        }


WEB_SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the live web for recent news, events, or current factual information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific query string to search for.",
                }
            },
            "required": ["query"],
        },
    },
}

# Explicit dictionary dispatch to guarantee security without eval
AVAILABLE_TOOLS = {
    "web_search": web_search,
}
