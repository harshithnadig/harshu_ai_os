"""Tests for MCP v1 (Model Context Protocol) integration."""

from harshu_ai_os.mcp.server import MCPServer


def test_mcp_initialize_handshake():
    server = MCPServer()
    msg = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    resp = server.handle_message(msg)

    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "harshu-ai-os-mcp"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_tools_list():
    server = MCPServer()
    msg = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    resp = server.handle_message(msg)

    assert resp["id"] == 2
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "rag_lookup" in tool_names
    assert "system_status" in tool_names


def test_mcp_call_system_status():
    server = MCPServer()
    msg = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "system_status", "arguments": {}},
    }
    resp = server.handle_message(msg)

    assert resp["id"] == 3
    result = resp["result"]
    assert result["isError"] is False
    assert len(result["content"]) == 1
    assert "Harshu AI OS v1 runtime operational" in result["content"][0]["text"]


def test_mcp_call_rag_lookup_mocked(monkeypatch):
    server = MCPServer()
    monkeypatch.setattr(
        "harshu_ai_os.mcp.server.rag_lookup",
        lambda query, max_results=3: {"content": f"Retrieved mock info for {query}", "sources": []},
    )

    msg = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {"name": "rag_lookup", "arguments": {"query": "fastapi"}},
    }
    resp = server.handle_message(msg)

    assert resp["id"] == 4
    result = resp["result"]
    assert result["isError"] is False
    assert "Retrieved mock info for fastapi" in result["content"][0]["text"]


def test_mcp_rejects_unauthorized_tool():
    server = MCPServer()
    msg = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {"name": "bash_exec", "arguments": {"cmd": "ls"}},
    }
    resp = server.handle_message(msg)

    assert resp["id"] == 5
    result = resp["result"]
    assert result["isError"] is True
    assert "not found in MCP tool allowlist" in result["content"][0]["text"]


def test_mcp_unknown_method_and_invalid_rpc():
    server = MCPServer()

    # Unknown method
    resp_unknown = server.handle_message({"jsonrpc": "2.0", "id": 6, "method": "nonexistent"})
    assert resp_unknown["error"]["code"] == -32601

    # Invalid RPC version
    resp_bad = server.handle_message({"jsonrpc": "1.0", "id": 7, "method": "initialize"})
    assert resp_bad["error"]["code"] == -32600
