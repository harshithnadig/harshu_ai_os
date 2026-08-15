"""Pytest test cases for OmniRoute subsystem roles, schemas, and fallback chains."""

import json
import sys
from pathlib import Path
import pytest

# Ensure repo root is on sys.path for local module resolution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR.parent))

from omiroute.client.gateway_client import GatewayResponse, EmbeddingResponse

CONFIG_DIR = BASE_DIR / "config"
ROLES_FILE = CONFIG_DIR / "roles.json"
COMBOS_FILE = CONFIG_DIR / "combos.json"
PROVIDERS_FILE = CONFIG_DIR / "providers.json"


@pytest.fixture
def roles_data():
    with open(ROLES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["roles"]


@pytest.fixture
def combos_data():
    with open(COMBOS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["combos"]


@pytest.fixture
def providers_data():
    with open(PROVIDERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["providers"]


def test_roles_configuration_integrity(roles_data):
    """Verify all 6 required logical roles are defined."""
    required_roles = [
        "harshu-classifier",
        "harshu-judge",
        "harshu-general",
        "harshu-reasoning",
        "harshu-tools",
        "harshu-embedding",
    ]
    for role in required_roles:
        assert role in roles_data, f"Role {role} is missing from config/roles.json"


def test_test1_classifier_contract(roles_data):
    """TEST 1: Classifier role configuration and schema contract."""
    classifier = roles_data["harshu-classifier"]
    assert classifier["strategy"] == "fill-first"
    assert len(classifier["models"]) >= 2
    assert classifier["requirements"]["structured_json"] is True


def test_test2_sufficiency_judge_contract(roles_data):
    """TEST 2: Sufficiency Judge contract and required JSON keys."""
    judge = roles_data["harshu-judge"]
    assert judge["strategy"] == "fill-first"
    reqs = judge["requirements"]
    assert "answerable" in reqs["required_keys"]
    assert "reason" in reqs["required_keys"]
    assert "supporting_chunk_ids" in reqs["required_keys"]


def test_test3_general_generation_contract(roles_data):
    """TEST 3: General generation role configuration."""
    general = roles_data["harshu-general"]
    assert general["models"][0]["provider"] in ["gemini", "groq"]


def test_test4_reasoning_contract(roles_data):
    """TEST 4: Reasoning role configuration."""
    reasoning = roles_data["harshu-reasoning"]
    assert reasoning["strategy"] == "priority"
    assert reasoning["requirements"]["reasoning_capability"] is True


def test_test5_tool_calling_schema_preservation(roles_data):
    """TEST 5: Tool calling role and schema preservation rules."""
    tools_role = roles_data["harshu-tools"]
    assert tools_role["requirements"]["strict_tool_calling_support"] is True
    assert tools_role["requirements"]["preserve_tool_schema"] is True
    assert tools_role["requirements"]["preserve_tool_call_id"] is True


def test_test6_embedding_contract_and_dimension(roles_data):
    """TEST 6: Embedding vector dimension and single-model isolation."""
    embedding_role = roles_data["harshu-embedding"]
    assert embedding_role["strategy"] == "priority"
    assert embedding_role["models"][0]["dimension"] == 3072
    assert embedding_role["fallback_policy"] == "NO_CROSS_MODEL_FALLBACK"


def test_test7_deterministic_fallback_mechanics(combos_data):
    """TEST 7: Validate fallback combo definitions."""
    combos_map = {c["name"]: c for c in combos_data}
    assert "harshu-classifier" in combos_map
    classifier_combo = combos_map["harshu-classifier"]
    assert classifier_combo["strategy"] == "fill-first"
    assert len(classifier_combo["models"]) >= 2
    assert classifier_combo["maxRetries"] >= 1


def test_test8_tool_fallback_models_verified(roles_data):
    """TEST 8: Every model in harshu-tools must have supports_tools = True."""
    tools_role = roles_data["harshu-tools"]
    for m in tools_role["models"]:
        assert m["supports_tools"] is True, f"Model {m['model']} does not support function calling!"
