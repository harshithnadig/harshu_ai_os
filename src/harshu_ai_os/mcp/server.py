"""Model Context Protocol (MCP) v1 safe read-only server adapter."""

from typing import Any

from harshu_ai_os.llm.tools import rag_lookup


class MCPServer:
    """Minimal, safe, read-only Model Context Protocol (MCP) JSON-RPC 2.0 server.

    Adheres strictly to the official MCP specification:
    - Protocol version 2024-11-05
    - Capabilities: read-only tool listing and execution
    - No arbitrary shell or filesystem access
    """

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, name: str = "harshu-ai-os-mcp", version: str = "0.1.0") -> None:
        self.name = name
        self.version = version
        self._tools: dict[str, dict[str, Any]] = {
            "rag_lookup": {
                "name": "rag_lookup",
                "description": "Query Harshu AI OS internal indexed knowledge base (read-only).",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Knowledge search query"},
                        "max_results": {"type": "integer", "description": "Max results to return (1-5)", "default": 3},
                    },
                    "required": ["query"],
                },
                "handler": lambda query, max_results=3: rag_lookup(query, max_results=max_results),
            },
            "system_status": {
                "name": "system_status",
                "description": "Inspect Harshu AI OS runtime status and available capabilities.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                },
                "handler": lambda: {
                    "content": "Harshu AI OS v1 runtime operational. Workflows: Direct, Agent, Strict RAG.",
                    "sources": [],
                },
            },
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool descriptors."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in self._tools.values()
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an allowlisted tool conforming to the MCP tool calling contract."""
        if name not in self._tools:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: Tool '{name}' not found in MCP tool allowlist."}],
            }

        handler = self._tools[name]["handler"]
        try:
            raw_result = handler(**arguments)
            if isinstance(raw_result, dict):
                text_content = raw_result.get("content", str(raw_result))
            else:
                text_content = str(raw_result)

            return {
                "isError": False,
                "content": [{"type": "text", "text": text_content}],
            }
        except Exception as exc:
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"MCP tool execution failed: {type(exc).__name__}"}],
            }

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """Handle a single MCP JSON-RPC 2.0 request."""
        req_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        if message.get("jsonrpc") != "2.0":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: JSON-RPC 2.0 required."},
            }

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "serverInfo": {"name": self.name, "version": self.version},
                    "capabilities": {"tools": {"listChanged": False}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.list_tools()},
            }

        if method == "tools/call":
            tool_name = params.get("name", "")
            args = params.get("arguments", {})
            result = self.call_tool(tool_name, args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found."},
        }
