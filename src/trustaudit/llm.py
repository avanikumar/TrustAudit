"""
LLM writers.

A writer turns the audit context (prediction, uncertainty, SHAP facts and
any feedback about rejected claims) into:

    {"explanation": "<plain-language text>",
     "claims": [{"feature": ..., "direction": ..., "strength": ...}, ...]}

The LLM writes in one call and lists its own claims in the same JSON, so
every sentence it makes about a feature has a matching checkable claim.

Two writers are provided:
- OpenAICompatWriter: any OpenAI-compatible endpoint (Groq, OpenRouter, OpenAI).
- TemplateWriter:     deterministic and offline, used when no API key is set.
                      It only restates SHAP, so it is faithful by construction
                      and is NOT a test of an LLM; use it for demos and CI.
"""
import json
import os
import re
import time

import httpx

SYSTEM_PROMPT = (
    "You explain individual predictions of a machine-learning model to a "
    "non-technical reader. Only describe effects that appear in the FACTS. "
    "Never invent features. Use the exact feature names from the FACTS in the "
    "claims list. Reply with JSON only."
)


def build_prompt(ctx: dict) -> str:
    facts = "\n".join(
        f"- {a['feature']} = {a['value']}: {a['direction']} {ctx['positive_label']} risk "
        f"(SHAP {a['shap']:+.3f}, rank {a['rank']})"
        for a in ctx["top_features"]
    )
    prompt = (
        f"Predicted probability of {ctx['positive_label']}: {ctx['probability']:.1%}\n"
        f"Conformal prediction set ({ctx['coverage']:.0%} coverage): {ctx['prediction_set']}\n"
        f"FACTS (top features by SHAP):\n{facts}\n\n"
        "Write a 2-4 sentence explanation, then list every feature effect you "
        "mentioned as a claim.\n"
        'Return JSON: {"explanation": str, "claims": [{"feature": str, '
        '"direction": "increases"|"decreases", "strength": "strong"|"moderate"|"weak"}]}\n'
        "strength: strong = among the top 3 features, moderate = top 6, weak = smaller."
    )
    if ctx.get("feedback"):
        prompt += (
            "\n\nYour previous draft contained claims that failed verification "
            "against the model:\n" + "\n".join(f"- {r}" for r in ctx["feedback"]) +
            "\nRewrite the explanation without these errors."
        )
    return prompt


def parse_response(text: str) -> dict:
    """Pull the JSON object out of an LLM reply (tolerates ``` fences and chatter)."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in LLM reply")
    data = json.loads(match.group(0))
    if not isinstance(data.get("explanation"), str) or not isinstance(data.get("claims"), list):
        raise ValueError("LLM reply is missing 'explanation' or 'claims'")
    return data


class OpenAICompatWriter:
    def __init__(self, base_url, model, api_key, temperature=0.0, max_retries=4, timeout=60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def _post(self, messages):
        for attempt in range(self.max_retries + 1):
            try:
                resp = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": messages,
                          "temperature": self.temperature},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                transient = (isinstance(exc, httpx.TransportError) or
                             exc.response.status_code == 429 or
                             exc.response.status_code >= 500)
                if not transient or attempt == self.max_retries:
                    raise
                time.sleep(2 ** attempt)

    def write(self, ctx: dict) -> dict:
        reply = self._post([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(ctx)},
        ])
        return parse_response(reply)


class TemplateWriter:
    """Offline writer: restates the top SHAP features. Faithful by construction."""

    def __init__(self, k=3):
        self.k = k

    def write(self, ctx: dict) -> dict:
        label = ctx["positive_label"]
        top = ctx["top_features"][: self.k]
        claims, parts = [], []
        for a in top:
            strength = "strong" if a["rank"] <= 3 else "moderate" if a["rank"] <= 6 else "weak"
            claims.append({"feature": a["feature"], "direction": a["direction"],
                           "strength": strength})
            parts.append(f"{a['feature']} = {a['value']} {a['direction']} the risk")
        text = (f"The model estimates a {ctx['probability']:.0%} chance of {label} "
                f"({ctx['prediction_set']}). Main drivers: " + "; ".join(parts) + ".")
        return {"explanation": text, "claims": claims}


def writer_from_env():
    """Use a real LLM if TRUSTAUDIT_API_KEY_ENV points at a set key, else go offline."""
    key_env = os.environ.get("TRUSTAUDIT_API_KEY_ENV", "GROQ_API_KEY")
    api_key = os.environ.get(key_env)
    if not api_key:
        return TemplateWriter()
    return OpenAICompatWriter(
        base_url=os.environ.get("TRUSTAUDIT_BASE_URL", "https://api.groq.com/openai/v1"),
        model=os.environ.get("TRUSTAUDIT_MODEL", "llama-3.3-70b-versatile"),
        api_key=api_key,
    )
