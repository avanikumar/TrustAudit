"""
The TrustAudit agent: a LangGraph loop with reject-and-retry.

    analyze --> write --> verify --+--> finalize   (all claims supported)
                  ^                |
                  +---- feedback --+                (some rejected, retries left)
                                   +--> finalize    (retries exhausted: unsupported
                                                     claims are flagged, not hidden)

- analyze:  model prediction, SHAP attributions, conformal prediction set
            (deterministic tools; no LLM)
- write:    the LLM drafts an explanation and lists its claims
- verify:   every claim is checked with SHAP + deletion tests
- finalize: package the audited result

The LLM only writes. Every fact it is allowed to state comes from the tools,
and every claim it makes is checked against the model before it is accepted.
"""
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from .conformal import calibrate, describe_set, predict_sets
from .data import Dataset
from .explain import attributions, build_explainer
from .llm import TemplateWriter
from .model import train_model
from .verifier import Claim, Verifier


class AuditState(TypedDict, total=False):
    row: pd.DataFrame
    probability: float
    prediction_set: str
    attributions: list
    explanation: str
    claims: list
    verdicts: list          # verdicts for the current draft
    feedback: list          # reasons from the last rejected draft
    history: list           # every draft and its verdicts, for transparency
    attempts: int
    error: str | None
    status: str


class Auditor:
    def __init__(self, model, explainer, verifier, q, positive_label, writer=None,
                 alpha=0.1, top_k=6, max_attempts=3):
        self.model = model
        self.explainer = explainer
        self.verifier = verifier
        self.q = q
        self.positive_label = positive_label
        self.writer = writer or TemplateWriter()
        self.alpha = alpha
        self.top_k = top_k
        self.max_attempts = max_attempts
        self.graph = self._build_graph()

    @classmethod
    def from_dataset(cls, ds: Dataset, writer=None, alpha=0.1, **kwargs):
        model = train_model(ds.X_train, ds.y_train)
        q = calibrate(model, ds.X_cal, ds.y_cal, alpha)
        verifier = Verifier(model, ds.X_train)
        return cls(model, build_explainer(model), verifier, q, ds.positive_label,
                   writer=writer, alpha=alpha, **kwargs)

    # ---- nodes ---------------------------------------------------------
    def analyze(self, state: AuditState) -> dict:
        row = state["row"]
        row_set = predict_sets(self.model, row, self.q)[0]
        return {
            "probability": float(self.model.predict_proba(row)[0, 1]),
            "prediction_set": describe_set(row_set, self.positive_label),
            "attributions": attributions(self.explainer, row),
            "attempts": 0,
            "history": [],
            "feedback": [],
        }

    def write(self, state: AuditState) -> dict:
        ctx = {
            "positive_label": self.positive_label,
            "probability": state["probability"],
            "prediction_set": state["prediction_set"],
            "coverage": 1 - self.alpha,
            "top_features": state["attributions"][: self.top_k],
            "feedback": state.get("feedback", []),
        }
        attempts = state["attempts"] + 1
        try:
            out = self.writer.write(ctx)
            return {"explanation": out["explanation"],
                    "claims": [Claim.from_dict(c) for c in out["claims"]],
                    "attempts": attempts, "error": None}
        except Exception as exc:   # bad JSON, API error: count it as a failed attempt
            return {"explanation": "", "claims": [], "attempts": attempts,
                    "error": f"writer failed: {exc}"}

    def verify(self, state: AuditState) -> dict:
        if state.get("error"):
            reasons = [state["error"]]
            verdicts = []
        elif not state["claims"]:
            reasons = ["the explanation made no checkable claims about features"]
            verdicts = []
        else:
            verdicts = self.verifier.verify_all(state["claims"], state["row"],
                                                state["attributions"])
            reasons = [r for v in verdicts for r in v.reasons]
        history = state["history"] + [{
            "attempt": state["attempts"],
            "explanation": state["explanation"],
            "verdicts": [v.to_dict() for v in verdicts],
            "rejection_reasons": reasons,
        }]
        return {"verdicts": verdicts, "feedback": reasons, "history": history}

    def route(self, state: AuditState) -> str:
        if not state["feedback"]:
            return "finalize"
        if state["attempts"] >= self.max_attempts:
            return "finalize"
        return "write"

    def finalize(self, state: AuditState) -> dict:
        if not state["feedback"]:
            status = "verified"
        elif state["verdicts"] and any(v.supported for v in state["verdicts"]):
            status = "partially_verified"
        else:
            status = "failed"
        return {"status": status}

    def _build_graph(self):
        g = StateGraph(AuditState)
        g.add_node("analyze", self.analyze)
        g.add_node("write", self.write)
        g.add_node("verify", self.verify)
        g.add_node("finalize", self.finalize)
        g.set_entry_point("analyze")
        g.add_edge("analyze", "write")
        g.add_edge("write", "verify")
        g.add_conditional_edges("verify", self.route,
                                {"write": "write", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    # ---- public API ----------------------------------------------------
    def audit(self, row: pd.DataFrame) -> dict[str, Any]:
        state = self.graph.invoke({"row": row})
        return {
            "probability": round(state["probability"], 4),
            "prediction_set": state["prediction_set"],
            "explanation": state["explanation"],
            "status": state["status"],
            "attempts": state["attempts"],
            "claims": [v.to_dict() for v in state["verdicts"]],
            "top_features": state["attributions"][: self.top_k],
            "history": state["history"],
        }


if __name__ == "__main__":
    import json
    import sys

    from .data import load_dataset
    from .llm import writer_from_env

    ds = load_dataset(sys.argv[1] if len(sys.argv) > 1 else "telco")
    row_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    auditor = Auditor.from_dataset(ds, writer=writer_from_env())
    result = auditor.audit(ds.X_test.iloc[[row_index]])
    result.pop("history")
    print(json.dumps(result, indent=2, default=str))
