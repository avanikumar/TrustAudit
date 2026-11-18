import pytest

from trustaudit.llm import TemplateWriter, build_prompt, parse_response

CTX = {
    "positive_label": "churn", "probability": 0.7, "prediction_set": "confident: churn",
    "coverage": 0.9, "feedback": [],
    "top_features": [
        {"feature": "tenure", "value": 2, "shap": 0.9, "rank": 1, "direction": "increases"},
        {"feature": "Contract", "value": "Two year", "shap": -0.5, "rank": 2,
         "direction": "decreases"},
    ],
}


def test_parse_response_handles_fences_and_chatter():
    text = 'Sure!\n```json\n{"explanation": "x", "claims": []}\n```'
    assert parse_response(text) == {"explanation": "x", "claims": []}


def test_parse_response_rejects_bad_shape():
    with pytest.raises(ValueError):
        parse_response("no json here")
    with pytest.raises(ValueError):
        parse_response('{"explanation": "x"}')


def test_prompt_contains_facts_and_feedback():
    p = build_prompt({**CTX, "feedback": ["tenure claim was wrong"]})
    assert "tenure = 2" in p and "Contract = Two year" in p
    assert "failed verification" in p and "tenure claim was wrong" in p


def test_template_writer_restates_shap():
    out = TemplateWriter().write(CTX)
    assert out["claims"][0] == {"feature": "tenure", "direction": "increases",
                                "strength": "strong"}
    assert "70%" in out["explanation"]
