"""Model Context Protocol (MCP) server adapter conforming to the 2026-07-28 specification.

Uses the official Model Context Protocol Python SDK v2 (MCPServer).
Exposes only safe, bounded, read-only capabilities:
- rag_lookup: Knowledge base search with bounded query and results.
- system_status: Truthful runtime capability and component health inspection.

Strictly avoids arbitrary shell execution, write operations, and secret exposure.
"""

import sys
from typing import Any

from mcp.server import MCPServer
import mcp.types as types

from harshu_ai_os.llm.tools import rag_lookup as core_rag_lookup
from harshu_ai_os.rag.chroma_store import get_notes_collection

# Current specification version supported by this adapter
LATEST_PROTOCOL_VERSION = types.LATEST_PROTOCOL_VERSION  # "2026-07-28"
DEFAULT_SERVER_NAME = "harshu-ai-os-mcp"


def check_system_status() -> dict[str, Any]:
    """Inspect and return truthful capability and component health status."""
    status_report: dict[str, Any] = {
        "version": "0.1.0",
        "protocol_version": LATEST_PROTOCOL_VERSION,
        "python_version": sys.version.split()[0],
        "workflows": ["direct", "agent", "strict_rag"],
        "tools_exposed": ["rag_lookup", "system_status"],
        "read_only": True,
    }
    try:
        _ = get_notes_collection()
        status_report["vector_store"] = "healthy"
    except Exception as exc:
        status_report["vector_store"] = f"degraded: {type(exc).__name__}"

    return status_report


def create_mcp_server(name: str = DEFAULT_SERVER_NAME) -> MCPServer:
    """Create and configure the official MCPServer with safe read-only tools."""
    server = MCPServer(name=name)

    @server.tool(
        name="rag_lookup",
        description="Search Harshu AI OS internal indexed knowledge base (read-only).",
    )
    def rag_lookup(query: str, max_results: int = 3) -> str:
        """Search Harshu AI OS internal indexed knowledge base (read-only)."""
        # 1. Enforce query type and non-empty bounds
        if not isinstance(query, str):
            return "Error: Query must be a valid string."

        clean_query = query.strip()
        if not clean_query:
            return "Error: Search query cannot be empty."

        if len(clean_query) > 1000:
            clean_query = clean_query[:1000]

        # 2. Enforce max_results range [1, 10]
        try:
            val_results = int(max_results)
        except (ValueError, TypeError):
            val_results = 3

        bounded_max_results = max(1, min(val_results, 10))

        # 3. Execute safe read-only retrieval
        res = core_rag_lookup(clean_query, max_results=bounded_max_results)
        if isinstance(res, dict):
            return res.get("content", str(res))
        return str(res)

    @server.tool(
        name="system_status",
        description="Inspect Harshu AI OS truthful runtime capabilities and component health.",
    )
    def system_status() -> str:
        """Inspect Harshu AI OS truthful runtime capabilities and component health."""
        status = check_system_status()
        lines = [
            "Harshu AI OS Runtime Status:",
            f"- Version: {status['version']}",
            f"- Protocol Version: {status['protocol_version']}",
            f"- Python: {status['python_version']}",
            f"- Workflows: {', '.join(status['workflows'])}",
            f"- Vector Store: {status['vector_store']}",
            f"- Read-Only: {status['read_only']}",
        ]
        return "\n".join(lines)

    return server
