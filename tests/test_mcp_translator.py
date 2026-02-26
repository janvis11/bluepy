"""Tests for MCP translator."""

import pytest

from backend.rag.mcp_translator import MCPTranslator, QueryIntent, VisualizationType


@pytest.fixture
def mcp_translator():
    """Create MCP translator instance."""
    return MCPTranslator(llm_provider="openai")


def test_mcp_translator_initialization(mcp_translator):
    """Test MCP translator initialization."""
    assert mcp_translator is not None
    assert mcp_translator.provider == "openai"


def test_sql_validation():
    """Test SQL validation."""
    translator = MCPTranslator()

    valid_sql = "SELECT * FROM argo_profile_meta LIMIT 10"
    validated = translator._validate_sql(valid_sql)
    assert "SELECT" in validated
    assert "LIMIT" in validated

    with pytest.raises(ValueError):
        translator._validate_sql("DROP TABLE argo_profile_meta")

    with pytest.raises(ValueError):
        translator._validate_sql("DELETE FROM argo_profile_meta")

    with pytest.raises(ValueError):
        translator._validate_sql("UPDATE argo_profile_meta SET status='test'")


def test_parse_response():
    """Test response parsing."""
    translator = MCPTranslator()

    response_json = """
    {
        "intent": "SELECT",
        "sql": "SELECT * FROM argo_profile_meta LIMIT 10",
        "visualization": "map",
        "notes": "Test query",
        "confidence": 0.95,
        "clarification_needed": false,
        "clarification_questions": [],
        "parameters": {}
    }
    """

    mcp_response = translator._parse_response(response_json)

    assert mcp_response.intent == QueryIntent.SELECT
    assert mcp_response.sql is not None
    assert mcp_response.visualization == VisualizationType.MAP
    assert mcp_response.confidence == 0.95
    assert not mcp_response.clarification_needed


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Integration test requires API key",
)
def test_translate_query(mcp_translator):
    """Test query translation (integration test)."""
    query = "Show me all profiles from float 2902123"

    response = mcp_translator.translate(query)

    assert response is not None
    assert response.intent in [QueryIntent.SELECT, QueryIntent.PLOT]
    assert response.sql is not None
    assert "2902123" in response.sql


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )
