"""Single-purpose tools and schemas for the LLM runtime."""

from ddgs import DDGS

from harshu_ai_os.rag.chroma_store import get_notes_collection, query_notes
from harshu_ai_os.rag.embedding_client import get_embedding_client


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


def rag_lookup(query: str, max_results: int = 3) -> dict:
    """Search Harshu AI OS local Chroma knowledge base and return compact text chunks."""
    if not query or not query.strip():
        return {
            "content": "No search query provided.",
            "sources": [],
            "query": "",
        }

    try:
        collection = get_notes_collection()
        client = get_embedding_client()
        retrieval = query_notes(
            collection=collection,
            client=client,
            question=query.strip(),
            top_k=max_results,
        )

        texts = retrieval.get("texts", [])[:max_results]
        metadatas = retrieval.get("metadatas", [])[:max_results]

        if not texts:
            return {
                "content": "No local knowledge base results found.",
                "sources": [],
                "query": query.strip(),
            }

        snippets = []
        sources = []
        for i, (text, meta) in enumerate(zip(texts, metadatas), 1):
            source_name = (
                meta.get("source", "knowledge_base")
                if isinstance(meta, dict)
                else "knowledge_base"
            )
            snippets.append(f"[{i}] {source_name}: {text}")
            sources.append({"title": source_name, "url": ""})

        return {
            "content": "\n\n".join(snippets),
            "sources": sources,
            "query": query.strip(),
        }
    except Exception as error:
        return {
            "content": f"RAG lookup failed: {error}",
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

RAG_LOOKUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "rag_lookup",
        "description": (
            "Search Harshu AI OS's internal/local indexed knowledge base for project "
            "documents, stored notes, architecture, and other private indexed information. "
            "Do not use this for current web information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific query string to search for in the local knowledge base.",
                }
            },
            "required": ["query"],
        },
    },
}

# Explicit dictionary dispatch to guarantee security without eval
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "rag_lookup": rag_lookup,
}

