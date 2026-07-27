"""Unit tests for the LLM-based re-ranking stage of RRF search.

All tests mock the OpenAI client so no real API calls are made.
The fixture also mocks ``load_dotenv``, ``time.sleep`` (so tests run fast),
and injects a dummy ``OPENROUTER_API_KEY`` into the environment.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from cli.hybrid_search_cli import batch_rerank, individual_rerank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(content: str) -> MagicMock:
    """Build a mock OpenAI response whose message content is *content*."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _doc(doc_id: int, title: str = "", desc: str = "") -> dict:
    return {
        "id": doc_id,
        "title": title or f"Movie {doc_id}",
        "description": desc or f"Desc {doc_id}",
        "rrf_score": 1.0,
        "bm25_rank": doc_id,
        "semantic_rank": doc_id * 2,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """Mock env, load_dotenv, time.sleep, and OpenAI.  Yields the mock client."""
    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test-key"}),
        patch("cli.hybrid_search_cli.load_dotenv"),
        patch("cli.hybrid_search_cli.time.sleep"),
        patch("cli.hybrid_search_cli.OpenAI") as mock_openai,
    ):
        client = MagicMock()
        mock_openai.return_value = client
        yield client


# ---------------------------------------------------------------------------
# Tests — individual_rerank
# ---------------------------------------------------------------------------


class TestIndividualRerank:
    """Exercises ``individual_rerank()`` — the per-document LLM scorer.

    Every test patches the OpenAI client so that ``chat.completions.create``
    returns controlled scores, letting us verify sorting, retries, truncation,
    and edge cases without any real API cost.
    """

    def test_returns_limited_results(self, mock_client):
        """Only ``limit`` items should be returned even if the pool is larger."""
        results = [_doc(i) for i in range(1, 11)]
        mock_client.chat.completions.create.return_value = _mock_response("5.0")

        out = individual_rerank("q", results, 3)

        assert len(out) == 3

    def test_sorts_by_llm_score_descending(self, mock_client):
        """Results should be ordered by LLM score descending, not original order."""
        results = [_doc(10), _doc(20), _doc(30)]
        mock_client.chat.completions.create.side_effect = [
            _mock_response("1.0"),
            _mock_response("9.0"),
            _mock_response("5.0"),
        ]

        out = individual_rerank("q", results, 3)

        # Highest LLM score first
        assert [r["id"] for r in out] == [20, 30, 10]
        assert [r["llm_score"] for r in out] == [9.0, 5.0, 1.0]

    def test_retries_on_malformed_output(self, mock_client):
        """Non-numeric LLM responses should be retried (up to 3 attempts)."""
        doc = _doc(1)
        mock_client.chat.completions.create.side_effect = [
            _mock_response("not a number"),
            _mock_response("also bad"),
            _mock_response("8.0"),
        ]

        out = individual_rerank("q", [doc], 5)

        assert out[0]["llm_score"] == 8.0
        assert mock_client.chat.completions.create.call_count == 3

    def test_retries_on_api_error(self, mock_client):
        """Transient API exceptions (e.g. network blips) should be retried."""
        doc = _doc(1)
        mock_client.chat.completions.create.side_effect = [
            RuntimeError("API error"),
            _mock_response("not a number"),
            _mock_response("7.5"),
        ]

        out = individual_rerank("q", [doc], 5)

        assert out[0]["llm_score"] == 7.5
        assert mock_client.chat.completions.create.call_count == 3

    def test_defaults_to_zero_when_all_retries_fail(self, mock_client):
        """A document whose LLM call never succeeds gets a score of 0.0."""
        doc = _doc(1)
        mock_client.chat.completions.create.return_value = _mock_response("invalid")

        out = individual_rerank("q", [doc], 5)

        assert out[0]["llm_score"] == 0.0
        assert mock_client.chat.completions.create.call_count == 3

    def test_rejects_out_of_range_score(self, mock_client):
        """Scores outside the valid 0–10 range should be retried."""
        doc = _doc(1)
        mock_client.chat.completions.create.side_effect = [
            _mock_response("15.0"),  # out of range → retry
            _mock_response("-1.0"),  # out of range → retry
            _mock_response("6.0"),
        ]

        out = individual_rerank("q", [doc], 5)

        assert out[0]["llm_score"] == 6.0
        assert mock_client.chat.completions.create.call_count == 3

    def test_preserves_original_document_fields(self, mock_client):
        """Original metadata (title, description, RRF score, ranks) must survive."""
        results = [_doc(42, "Test Title", "Test description")]
        mock_client.chat.completions.create.return_value = _mock_response("7.0")

        out = individual_rerank("q", results, 5)

        assert out[0]["title"] == "Test Title"
        assert out[0]["description"] == "Test description"
        assert out[0]["rrf_score"] == 1.0
        assert out[0]["bm25_rank"] == 42

    def test_llm_score_added_to_each_result(self, mock_client):
        """Every returned result should carry its ``llm_score`` key."""
        results = [_doc(i) for i in range(1, 4)]
        mock_client.chat.completions.create.side_effect = [
            _mock_response("3.0"),
            _mock_response("8.0"),
            _mock_response("5.0"),
        ]

        out = individual_rerank("q", results, 3)

        for r in out:
            assert "llm_score" in r

    def test_handles_empty_results(self, mock_client):
        """An empty candidate pool should produce an empty result list."""
        out = individual_rerank("q", [], 5)
        assert out == []


# ---------------------------------------------------------------------------
# Tests — batch_rerank
# ---------------------------------------------------------------------------


class TestBatchRerank:
    """Exercises ``batch_rerank()`` — the single-call LLM ranker.

    The mock response returns a JSON array of document IDs in the desired
    order so we can verify re-ordering, truncation, and edge cases.
    """

    def test_returns_limited_results(self, mock_client):
        """Only ``limit`` items should be returned even if the pool is larger."""
        results = [_doc(i) for i in range(1, 11)]
        mock_client.chat.completions.create.return_value = _mock_response("[3, 1, 2]")

        out = batch_rerank("q", results, 2)

        assert len(out) == 2

    def test_orders_by_llm_rank(self, mock_client):
        """Results should follow the rank order returned by the LLM."""
        results = [_doc(10), _doc(20), _doc(30)]
        mock_client.chat.completions.create.return_value = _mock_response("[30, 10, 20]")

        out = batch_rerank("q", results, 3)

        assert [r["id"] for r in out] == [30, 10, 20]
        assert [r["llm_rank"] for r in out] == [1, 2, 3]

    def test_llm_rank_added_to_each_result(self, mock_client):
        """Every returned result should carry its ``llm_rank`` key."""
        results = [_doc(1), _doc(2)]
        mock_client.chat.completions.create.return_value = _mock_response("[2, 1]")

        out = batch_rerank("q", results, 2)

        for r in out:
            assert "llm_rank" in r

    def test_preserves_original_document_fields(self, mock_client):
        """Original metadata (title, description, RRF score, ranks) must survive."""
        results = [_doc(42, "Test Title", "Test description")]
        mock_client.chat.completions.create.return_value = _mock_response("[42]")

        out = batch_rerank("q", results, 5)

        assert out[0]["title"] == "Test Title"
        assert out[0]["description"] == "Test description"
        assert out[0]["rrf_score"] == 1.0
        assert out[0]["bm25_rank"] == 42

    def test_skips_unknown_ids(self, mock_client):
        """IDs returned by the LLM that aren't in the pool should be skipped."""
        results = [_doc(1), _doc(2)]
        mock_client.chat.completions.create.return_value = _mock_response("[99, 1, 2]")

        out = batch_rerank("q", results, 5)

        # Unknown ID 99 is skipped
        assert [r["id"] for r in out] == [1, 2]

    def test_handles_empty_results(self, mock_client):
        """An empty candidate pool should produce an empty result list."""
        out = batch_rerank("q", [], 5)
        assert out == []

    def test_fallback_on_parse_failure(self, mock_client):
        """If the LLM response can't be parsed, fall back to original order."""
        mock_client.chat.completions.create.return_value = _mock_response("not json")

        results = [_doc(10), _doc(20), _doc(30)]
        out = batch_rerank("q", results, 3)

        # Original order preserved with sequential ranks
        assert [r["id"] for r in out] == [10, 20, 30]
        assert [r["llm_rank"] for r in out] == [1, 2, 3]

    def test_retry_on_malformed_json(self, mock_client):
        """Transient JSON failures should be retried."""
        mock_client.chat.completions.create.side_effect = [
            _mock_response("not json"),
            _mock_response("{bad"),
            _mock_response("[2, 1]"),
        ]

        out = batch_rerank("q", [_doc(1), _doc(2)], 2)

        assert [r["id"] for r in out] == [2, 1]
        assert mock_client.chat.completions.create.call_count == 3
