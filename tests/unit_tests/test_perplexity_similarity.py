# Standard Library
from unittest.mock import MagicMock

# Third-Party Library
import pytest
from requests.exceptions import RequestException

# SynThesisAI Modules
import core.search.perplexity_similarity as ps
from utils.exceptions import APIError


@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    config = MagicMock()
    config.get_api_key.return_value = "mock-key"
    monkeypatch.setattr(ps, "get_config_manager", lambda: config)
    return config


@pytest.fixture
def mock_post(monkeypatch):
    return monkeypatch.patch.object(ps.requests, "post")


def test_query_similarity_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": """{
                        "similarity_score": 0.76,
                        "matches": [
                            {"title": "Area of triangle with sin", "url": "https://math.stackexchange.com/q1", "similarity": 0.72, "source": "math.stackexchange"}
                        ]
                    }"""
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    result = ps.query_similarity_via_perplexity("Find the area of a triangle using sine")

    assert isinstance(result, dict)
    assert result["similarity_score"] == 0.76
    assert isinstance(result["top_matches"], list)
    assert result["top_matches"][0]["source"] == "math.stackexchange"


def test_query_similarity_fails_on_request_error(mock_post):
    mock_post.side_effect = RequestException("Network error")
    with pytest.raises(APIError, match="Perplexity API request failed"):
        ps.query_similarity_via_perplexity("Test")


def test_query_similarity_returns_fallback_on_parse_error(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "INVALID JSON"}}]}
    mock_post.return_value = mock_response

    result = ps.query_similarity_via_perplexity("x^2 + y^2 = z^2")
    assert result["similarity_score"] == 0.0
    assert "error" in result
