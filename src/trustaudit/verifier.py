"""
Claim verifier.

An LLM explanation is broken into structured claims such as
    {"feature": "tenure", "direction": "decreases", "strength": "strong"}
meaning "tenure strongly decreases the predicted risk for this row".

Each claim is checked against the model's actual behaviour in two
independent ways:

1. SHAP check     - does the attribution have the claimed sign, and is the
                    feature ranked highly enough for the claimed strength?
2. Deletion test  - replace the feature with values drawn from the training
                    data (so its information is "removed"), re-predict, and
                    check the probability moves the way the claim says.

A claim is supported only if every enabled check passes. The checks can be
switched off individually, which is how the ablation in `evaluate.py` works.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

DIRECTIONS = {"increases", "decreases"}
STRENGTHS = {"strong", "moderate", "weak"}

# A "strong" claim must be about one of the top-3 features, "moderate" top-6.
# "weak" only needs the right sign: understating an effect is not a falsehood.
MAX_RANK = {"strong": 3, "moderate": 6, "weak": None}


@dataclass
class Claim:
    feature: str
    direction: str
    strength: str = "moderate"

    @classmethod
    def from_dict(cls, d):
        return cls(
            feature=str(d.get("feature", "")).strip(),
            direction=str(d.get("direction", "")).strip().lower(),
            strength=str(d.get("strength", "moderate")).strip().lower(),
        )


@dataclass
class Verdict:
    claim: Claim
    supported: bool
    reasons: list = field(default_factory=list)
    shap: float | None = None
    rank: int | None = None
    deletion_delta: float | None = None   # log-odds(original) - log-odds(feature removed)
    probability_delta: float | None = None  # same change in probability, for readers

    def to_dict(self):
        d = asdict(self)
        d["claim"] = asdict(self.claim)
        return d


class Verifier:
    def __init__(self, model, background: pd.DataFrame, n_samples=50,
                 min_effect=0.05, use_shap=True, use_deletion=True, seed=0):
        """
        background:  training rows to draw replacement values from
        n_samples:   replacement draws per deletion test (averaged)
        min_effect:  smallest log-odds change that counts as a real effect.
                     Log-odds (the same scale as SHAP) rather than probability,
                     because probabilities saturate: for a row at 2% risk even a
                     genuinely important feature barely moves the probability.
        """
        self.model = model
        self.background = background.reset_index(drop=True)
        self.n_samples = n_samples
        self.min_effect = min_effect
        self.use_shap = use_shap
        self.use_deletion = use_deletion
        self.rng = np.random.default_rng(seed)

    # ---- deletion test -------------------------------------------------
    def deletion_delta(self, X_row: pd.DataFrame, feature: str):
        """Change in predicted log-odds when `feature` is removed.

        Returns (log-odds delta, probability delta). Positive: the feature was
        pushing the prediction towards the positive class.
        """
        idx = self.rng.integers(0, len(self.background), self.n_samples)
        X_rep = pd.concat([X_row] * self.n_samples, ignore_index=True)
        X_rep[feature] = self.background[feature].iloc[idx].to_numpy()
        X_rep[feature] = X_rep[feature].astype(X_row[feature].dtype)

        p_orig = float(self.model.predict_proba(X_row)[0, 1])
        p_rep = self.model.predict_proba(X_rep)[:, 1]
        logit_delta = _logit(p_orig) - float(np.mean(_logit(p_rep)))
        return logit_delta, p_orig - float(p_rep.mean())

    # ---- single claim --------------------------------------------------
    def verify(self, claim: Claim, X_row: pd.DataFrame, attributions: list) -> Verdict:
        reasons = []

        if claim.feature not in X_row.columns:
            return Verdict(claim, False, [f"unknown feature '{claim.feature}'"])
        if claim.direction not in DIRECTIONS:
            return Verdict(claim, False, [f"invalid direction '{claim.direction}'"])
        if claim.strength not in STRENGTHS:
            return Verdict(claim, False, [f"invalid strength '{claim.strength}'"])

        attr = next(a for a in attributions if a["feature"] == claim.feature)
        sign = 1 if claim.direction == "increases" else -1
        verdict = Verdict(claim, True, reasons, shap=attr["shap"], rank=attr["rank"])

        if self.use_shap:
            if np.sign(attr["shap"]) != sign:
                reasons.append(
                    f"SHAP says {claim.feature} {attr['direction']} risk "
                    f"(shap={attr['shap']:+.3f}), claim says {claim.direction}")
            max_rank = MAX_RANK[claim.strength]
            if max_rank and attr["rank"] > max_rank:
                reasons.append(
                    f"claimed {claim.strength} but {claim.feature} is only rank "
                    f"{attr['rank']} of {len(attributions)} by |SHAP|")

        if self.use_deletion:
            delta, p_delta = self.deletion_delta(X_row, claim.feature)
            verdict.deletion_delta = delta
            verdict.probability_delta = p_delta
            if sign * delta < self.min_effect:
                reasons.append(
                    f"removing {claim.feature} moves the prediction by "
                    f"{-delta:+.3f} log-odds ({-p_delta:+.1%}), which does not "
                    f"show that it {claim.direction} risk")

        verdict.supported = not reasons
        return verdict

    def verify_all(self, claims, X_row, attributions):
        return [self.verify(c, X_row, attributions) for c in claims]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))
