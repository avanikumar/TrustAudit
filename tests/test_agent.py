"""Agent loop tests with scripted fake LLMs (no network)."""
from trustaudit.agent import Auditor

FLIP = {"increases": "decreases", "decreases": "increases"}


class ScriptedWriter:
    """Returns a wrong claim first, then a correct one after feedback."""

    def __init__(self, wrong_times=1):
        self.wrong_times = wrong_times
        self.calls = []

    def write(self, ctx):
        self.calls.append(ctx)
        top = ctx["top_features"][0]
        direction = top["direction"]
        if len(self.calls) <= self.wrong_times:
            direction = FLIP[direction]
        return {"explanation": "draft",
                "claims": [{"feature": top["feature"], "direction": direction,
                            "strength": "strong"}]}


class BrokenWriter:
    def write(self, ctx):
        raise ValueError("bad JSON")


def make(ds, writer, max_attempts=3):
    return Auditor.from_dataset(ds, writer=writer, max_attempts=max_attempts)


def test_retry_fixes_wrong_claim(ds, row):
    w = ScriptedWriter(wrong_times=1)
    res = make(ds, w).audit(row)
    assert res["status"] == "verified"
    assert res["attempts"] == 2
    assert w.calls[0]["feedback"] == []
    assert w.calls[1]["feedback"], "second draft must receive the rejection reasons"
    assert len(res["history"]) == 2


def test_gives_up_after_max_attempts(ds, row):
    res = make(ds, ScriptedWriter(wrong_times=99), max_attempts=3).audit(row)
    assert res["attempts"] == 3
    assert res["status"] == "failed"
    assert all(not c["supported"] for c in res["claims"])


def test_writer_errors_are_retried_not_raised(ds, row):
    res = make(ds, BrokenWriter(), max_attempts=2).audit(row)
    assert res["status"] == "failed"
    assert "writer failed" in res["history"][0]["rejection_reasons"][0]


def test_result_has_uncertainty_and_probability(ds, row):
    res = make(ds, ScriptedWriter(wrong_times=0)).audit(row)
    assert 0 <= res["probability"] <= 1
    assert res["prediction_set"]
    assert res["status"] == "verified" and res["attempts"] == 1
