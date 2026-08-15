"""Isolated memory capability tests for OmniRoute subsystem."""

import sys
from pathlib import Path
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from omiroute.client.gateway_client import OmniRouteClient, MemoryItem


class LocalMemorySimulator:
    """Isolated in-memory store mirroring OmniRoute's memory subsystem mechanics."""
    def __init__(self):
        self.records: dict[str, dict] = {}

    def store(self, key: str, content: str, memory_type: str = "factual", metadata: dict = None) -> dict:
        mem_id = f"mem_{len(self.records) + 1}"
        record = {
            "id": mem_id,
            "key": key,
            "content": content,
            "type": memory_type,
            "metadata": metadata or {},
        }
        self.records[mem_id] = record
        return record

    def retrieve(self, query: str, limit: int = 5) -> list[MemoryItem]:
        # Keyword & token overlap scoring mirroring FTS5 / semantic match
        query_tokens = set(query.lower().replace("?", "").split())
        scored = []
        for rec in self.records.values():
            content_tokens = set(rec["content"].lower().split())
            overlap = len(query_tokens.intersection(content_tokens))
            if overlap > 0:
                scored.append(
                    MemoryItem(
                        id=rec["id"],
                        key=rec["key"],
                        content=rec["content"],
                        type=rec["type"],
                        score=round(overlap / len(query_tokens), 2),
                    )
                )
        return sorted(scored, key=lambda x: x.score, reverse=True)[:limit]


def test_synthetic_memory_store_and_retrieval():
    """Verify storing a user preference and retrieving it semantically."""
    mem_store = LocalMemorySimulator()

    # Step 1: Store synthetic public-safe memory
    stored = mem_store.store(
        key="user_learning_preference",
        content="Harshu prefers PUBG examples when learning programming.",
        memory_type="factual",
        metadata={"subject": "learning_style", "source": "user_explicit"},
    )
    assert stored["id"].startswith("mem_")
    assert stored["key"] == "user_learning_preference"

    # Step 2: Retrieve memory matching query
    query = "What kind of examples help Harshu learn?"
    results = mem_store.retrieve(query)

    # Step 3: Assert memory is retrieved
    assert len(results) >= 1
    top_memory = results[0]
    assert "PUBG" in top_memory.content
    assert top_memory.key == "user_learning_preference"
    assert top_memory.type == "factual"
