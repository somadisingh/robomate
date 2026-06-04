from backend.analyzers.gemini_eval import build_evaluation_prompt, response_schema
from backend.contracts import GeminiEvaluation


def test_prompt_includes_summary_and_task():
    prompt = build_evaluation_prompt("Pick up the cup.")

    assert "Pick up the cup." in prompt
    assert "summary" in prompt
    assert "score" in prompt
    assert "success_reasoning" in prompt


def test_gemini_evaluation_rejects_out_of_range_score():
    payload = {
        "summary": "The video shows a hand near a cup.",
        "success": True,
        "success_reasoning": "The cup is lifted.",
        "score": 11,
        "score_reasoning": "Too high.",
    }

    try:
        GeminiEvaluation.model_validate(payload)
    except Exception as exc:
        assert "score" in str(exc)
    else:
        raise AssertionError("Expected score validation to fail")


def test_response_schema_names_expected_fields():
    schema = response_schema()

    assert set(schema["properties"]) == {
        "summary",
        "success",
        "success_reasoning",
        "score",
        "score_reasoning",
    }
