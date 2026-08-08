"""Tests for MemorySummarizer backends (heuristic + LLM fallback)."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voltmem import HeuristicSummarizer, LLMSummarizer
from voltmem.summarize import MemorySummarizer


def test_heuristic_uses_latest_distinct_evidence():
    s: MemorySummarizer = HeuristicSummarizer()
    out = s.summarize(
        "User works as a data analyst",
        [
            "User mentioned a different role in passing",
            "User said they work as a nurse now",
        ],
        domain="professional_context",
    )
    assert out == "User said they work as a nurse now"
    assert "[consolidated]" not in out


def test_heuristic_empty_evidence_keeps_current():
    s = HeuristicSummarizer()
    assert s.summarize("tip", [], domain="location") == "tip"
    assert s.summarize("tip", ["", "  "], domain="location") == "tip"


def test_heuristic_dedupes_and_ignores_restatement():
    s = HeuristicSummarizer()
    tip = "User lives in Berlin"
    assert s.summarize(
        tip,
        ["User lives in Berlin", "user lives in berlin"],
        domain="location",
    ) == tip


def test_llm_summarizer_uses_model_output():
    s = LLMSummarizer()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({
                "response": '"User works as a nurse"',
            }).encode()

    with patch("urllib.request.urlopen", return_value=_Resp()):
        out = s.summarize(
            "User works as a data analyst",
            ["User mentioned nursing school", "User said they are a nurse"],
            domain="professional_context",
        )
    assert out == "User works as a nurse"


def test_llm_summarizer_falls_back_on_error():
    s = LLMSummarizer()

    with patch("urllib.request.urlopen", side_effect=OSError("down")):
        out = s.summarize(
            "User works as a data analyst",
            ["User said they work as a nurse now"],
            domain="professional_context",
        )
    assert out == "User said they work as a nurse now"


def test_llm_summarizer_empty_evidence_skips_network():
    s = LLMSummarizer()
    with patch("urllib.request.urlopen") as mock_open:
        out = s.summarize("kept tip", [], domain="core_preference")
    assert out == "kept tip"
    mock_open.assert_not_called()


if __name__ == "__main__":
    tests = [
        test_heuristic_uses_latest_distinct_evidence,
        test_heuristic_empty_evidence_keeps_current,
        test_heuristic_dedupes_and_ignores_restatement,
        test_llm_summarizer_uses_model_output,
        test_llm_summarizer_falls_back_on_error,
        test_llm_summarizer_empty_evidence_skips_network,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed + failed} tests passed")
    if failed:
        sys.exit(1)
