"""Tests for Model Context Protocol (MCP) conforming to the 2026-07-28 specification.

Uses the official MCP Python SDK v2 client (ClientSession, InMemoryTransport).
Validates:
1. Modern 2026-07-28 discovery via server/discover
2. Tool discovery and schema definitions
3. Safe read-only execution of rag_lookup
4. Input bounds and type validation
5. Truthful system_status runtime inspection
6. Rejection of unlisted / unauthorized tools
7. Legacy client backward-compatibility via initialize handshake
"""

from unittest.mock import MagicMock
import pytest
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession

from harshu_ai_os.mcp.server import (
    LATEST_PROTOCOL_VERSION,
    check_system_status,
    create_mcp_server,
)


@pytest.mark.anyio
async def test_mcp_modern_protocol_discover():
    """Verify modern 2026-07-28 discovery path via server/discover."""
    server = create_mcp_server()
    transport = InMemoryTransport(server)

    async with transport as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            discover_result = await session.discover()

            assert LATEST_PROTOCOL_VERSION in discover_result.supported_versions
            assert "2026-07-28" in discover_result.supported_versions
            assert discover_result.capabilities.tools is not None


@pytest.mark.anyio
async def test_mcp_tool_discovery_and_allowlist():
    """Verify only allowlisted read-only tools are exposed."""
    server = create_mcp_server()
    transport = InMemoryTransport(server)

    async with transport as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            disc = await session.discover()
            session.adopt(disc)

            tools_result = await session.list_tools()
            tool_names = {tool.name for tool in tools_result.tools}

            # Must contain our safe read-only tools
            assert "rag_lookup" in tool_names
            assert "system_status" in tool_names

            # Strictly must NOT expose shell or filesystem execution
            assert "bash" not in tool_names
            assert "sh" not in tool_names
            assert "exec" not in tool_names
            assert "file_write" not in tool_names


@pytest.mark.anyio
async def test_mcp_call_rag_lookup_success(monkeypatch):
    """Verify safe read-only execution of rag_lookup with mocked store."""
    server = create_mcp_server()
    transport = InMemoryTransport(server)

    monkeypatch.setattr(
        "harshu_ai_os.mcp.server.core_rag_lookup",
        lambda query, max_results=3: {
            "content": f"Mock knowledge for: {query} (top_k={max_results})",
            "sources": [{"title": "test_doc", "url": ""}],
        },
    )

    async with transport as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            disc = await session.discover()
            session.adopt(disc)

            result = await session.call_tool("rag_lookup", {"query": "architecture", "max_results": 2})

            assert result.is_error is False
            assert len(result.content) == 1
            assert "Mock knowledge for: architecture (top_k=2)" in result.content[0].text


@pytest.mark.anyio
async def test_mcp_rag_lookup_input_validation_and_bounds():
    """Verify input arguments are strictly validated and bounded."""
    server = create_mcp_server()
    transport = InMemoryTransport(server)

    async with transport as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            disc = await session.discover()
            session.adopt(disc)

            # Empty query returns validation error
            res_empty = await session.call_tool("rag_lookup", {"query": "   "})
            assert "Search query cannot be empty" in res_empty.content[0].text

            # Excessive string length is bounded
            long_query = "x" * 2000
            res_long = await session.call_tool("rag_lookup", {"query": long_query, "max_results": 100})
            assert res_long.is_error is False


@pytest.mark.anyio
async def test_mcp_truthful_system_status_runtime_check(monkeypatch):
    """Verify system_status inspects genuine component health without static assumptions."""
    # Healthy vector store check
    mock_collection = MagicMock()
    monkeypatch.setattr("harshu_ai_os.mcp.server.get_notes_collection", lambda: mock_collection)

    status = check_system_status()
    assert status["vector_store"] == "healthy"
    assert status["protocol_version"] == "2026-07-28"
    assert status["read_only"] is True

    # Degraded vector store check
    def failing_store():
        raise ConnectionError("ChromaDB socket unavailable")

    monkeypatch.setattr("harshu_ai_os.mcp.server.get_notes_collection", failing_store)

    degraded_status = check_system_status()
    assert "degraded: ConnectionError" in degraded_status["vector_store"]


@pytest.mark.anyio
async def test_mcp_rejects_unlisted_tools():
    """Verify unlisted / unauthorized tool calls fail safely."""
    server = create_mcp_server()
    transport = InMemoryTransport(server)

    async with transport as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            disc = await session.discover()
            session.adopt(disc)

            result = await session.call_tool("unauthorized_shell_exec", {"command": "ls"})

            assert result.is_error is True
            assert "Unknown tool: unauthorized_shell_exec" in result.content[0].text


@pytest.mark.anyio
async def test_mcp_legacy_client_backward_compatibility():
    """Verify legacy clients using initialize handshake are supported."""
    server = create_mcp_server()
    transport = InMemoryTransport(server)

    async with transport as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            init_res = await session.initialize()

            assert init_res.protocol_version is not None
            assert init_res.server_info.name == "harshu-ai-os-mcp"

            tools = await session.list_tools()
            tool_names = [t.name for t in tools.tools]
            assert "rag_lookup" in tool_names
