"""
Evaluation.

1. Verifier evaluation (no LLM needed)
   For each test row we build claims whose truth we know, then measure how
   often the verifier accepts the true ones and rejects the false ones:

     correct      top-3 feature, true direction, "strong"        -> should accept
     flipped      top-3 feature, opposite direction              -> should reject
     exaggerated  feature ranked in the bottom half, "strong"    -> should reject

   Ablation: SHAP check only, deletion test only, and both.
   Caveat: the ground truth is defined from SHAP, so the SHAP-only check is
   correct by construction. The deletion test is the independent check; its
   agreement with SHAP is the informative number.

2. Agent evaluation
   Runs the full agent on test rows with a writer and reports how many
   explanations end up verified, how many attempts that takes, and the share
   of claims that were supported on the first draft vs. at the end.
   - With GROQ_API_KEY (or another configured key) set: a real LLM.
   - `--writer noisy`: a SIMULATED writer that corrupts claims on purpose, to
     test that the reject-and-retry loop recovers. Not an LLM result.
   - Otherwise the offline template writer, which is faithful by construction.

Usage:
    python -m trustaudit.evaluate verifier --dataset telco --n 200
    python -m trustaudit.evaluate agent --dataset telco --n 50 [--writer noisy]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from .agent import Auditor
from .data import load_dataset
from .explain import attributions, build_explainer
from .llm import TemplateWriter, writer_from_env
from .model import train_model
from .verifier import Claim, Verifier

FLIP = {"increases": "decreases", "decreases": "increases"}


# ---------------------------------------------------------------------------
# 1. Verifier evaluation
# ---------------------------------------------------------------------------
def make_test_claims(attrs, rng):
    n = len(attrs)
    top = attrs[:3]
    bottom = attrs[n // 2:]
    a = rng.choice(top)
    b = rng.choice(top)
    c = rng.choice(bottom)
    return [
        ("correct", Claim(a["feature"], a["direction"], "strong"), True),
        ("flipped", Claim(b["feature"], FLIP[b["direction"]], "strong"), False),
        ("exaggerated", Claim(c["feature"], c["direction"], "strong"), False),
    ]


def evaluate_verifier(dataset="telco", n=200, seed=0):
    ds = load_dataset(dataset)
    model = train_model(ds.X_train, ds.y_train)
    explainer = build_explainer(model)
    rng = random.Random(seed)
    rows = ds.X_test.sample(min(n, len(ds.X_test)), random_state=seed)

    variants = {
        "shap_only": Verifier(model, ds.X_train, use_deletion=False, seed=seed),
        "deletion_only": Verifier(model, ds.X_train, use_shap=False, seed=seed),
        "shap+deletion": Verifier(model, ds.X_train, seed=seed),
    }
    counts = {v: {} for v in variants}
    for i in range(len(rows)):
        row = rows.iloc[[i]]
        attrs = attributions(explainer, row)
        for kind, claim, truth in make_test_claims(attrs, rng):
            for vname, verifier in variants.items():
                ok = verifier.verify(claim, row, attrs).supported
                c = counts[vname].setdefault(kind, {"n": 0, "accepted": 0})
                c["n"] += 1
                c["accepted"] += int(ok)

    results = {}
    for vname, by_kind in counts.items():
        r = {kind: c["accepted"] / c["n"] for kind, c in by_kind.items()}
        results[vname] = {
            "accept_correct": r["correct"],
            "reject_flipped": 1 - r["flipped"],
            "reject_exaggerated": 1 - r["exaggerated"],
        }
    return {"dataset": dataset, "rows": len(rows), "results": results}


# ---------------------------------------------------------------------------
# 2. Agent evaluation
# ---------------------------------------------------------------------------
class NoisyWriter:
    """SIMULATION ONLY: a faithful writer whose claims are corrupted at random.

    Each claim is flipped or exaggerated with probability `p` on a first draft,
    and with probability `p / 2` once feedback is given (as if feedback helps). Used to test
    the loop's mechanics without an API key.
    """

    def __init__(self, p=0.4, seed=0):
        self.base = TemplateWriter()
        self.p = p
        self.rng = random.Random(seed)

    def write(self, ctx):
        out = self.base.write(ctx)
        p = self.p / 2 if ctx.get("feedback") else self.p
        tail = ctx["top_features"][-1]
        claims = []
        for c in out["claims"]:
            r = self.rng.random()
            if r < p / 2:
                c = {**c, "direction": FLIP[c["direction"]]}
            elif r < p:
                c = {"feature": tail["feature"], "direction": tail["direction"],
                     "strength": "strong"}
            claims.append(c)
        return {"explanation": out["explanation"], "claims": claims}


def evaluate_agent(dataset="telco", n=50, writer=None, seed=0, max_attempts=3):
    ds = load_dataset(dataset)
    writer = writer or writer_from_env()
    auditor = Auditor.from_dataset(ds, writer=writer, max_attempts=max_attempts)
    rows = ds.X_test.sample(min(n, len(ds.X_test)), random_state=seed)

    statuses, attempts, first, final, latency = {}, [], [], [], []
    for i in range(len(rows)):
        t0 = time.perf_counter()
        res = auditor.audit(rows.iloc[[i]])
        latency.append(time.perf_counter() - t0)
        statuses[res["status"]] = statuses.get(res["status"], 0) + 1
        attempts.append(res["attempts"])
        v0 = res["history"][0]["verdicts"]
        if v0:
            first.append(np.mean([v["supported"] for v in v0]))
        if res["claims"]:
            final.append(np.mean([c["supported"] for c in res["claims"]]))

    return {
        "dataset": dataset,
        "writer": type(writer).__name__,
        "rows": len(rows),
        "status": {k: v / len(rows) for k, v in statuses.items()},
        "mean_attempts": float(np.mean(attempts)),
        "claims_supported_first_draft": float(np.mean(first)) if first else None,
        "claims_supported_final": float(np.mean(final)) if final else None,
        "mean_seconds_per_row": float(np.mean(latency)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["verifier", "agent"])
    ap.add_argument("--dataset", default="telco")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--writer", choices=["auto", "noisy"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.what == "verifier":
        res = evaluate_verifier(args.dataset, args.n, args.seed)
    else:
        writer = NoisyWriter(seed=args.seed) if args.writer == "noisy" else None
        res = evaluate_agent(args.dataset, args.n, writer, args.seed)

    print(json.dumps(res, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
