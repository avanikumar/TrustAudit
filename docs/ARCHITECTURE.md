# Architecture and design decisions

## Principle: the LLM writes, the tools decide

The model, SHAP, conformal prediction and the verifier are deterministic Python. The LLM gets facts from them and returns prose plus structured claims. It is never the source of truth about the model, and nothing it says is accepted without a check.

## Data flow

1. `analyze` computes the probability, the full SHAP attribution list (sorted, with rank and share) and the conformal prediction set. These go into the graph state once and never change.
2. `write` sends the top-6 features, the probability, the prediction set and any feedback to the writer. The writer returns `{"explanation", "claims"}`.
3. `verify` checks each claim. Rejection reasons are stored as `feedback`, and the draft and its verdicts are appended to `history`.
4. `route`: no feedback → finalize. Feedback with attempts left → write again. Otherwise → finalize.
5. `finalize` sets the status: `verified` (all claims pass), `partially_verified` (some pass) or `failed` (none pass, or the writer kept failing).

## State

`AuditState` (a TypedDict) holds: the input row, probability, prediction set, attributions, current explanation and claims, current verdicts, feedback, full history and attempt count. History makes every run auditable: you can see each rejected draft and why it was rejected.

## Decisions

**Structured claims rather than free-text checking.** Checking prose directly would need NLI or another LLM, which is the thing we don't trust. Asking the writer for `{feature, direction, strength}` turns verification into plain computation. The cost is coverage: only these claim types are checked (see Limitations in the README).

**Two independent checks.** SHAP is an attribution method with its own assumptions. The deletion test intervenes on the model directly. Requiring both catches errors either one would miss alone: deletion catches nothing about rank, and SHAP can't confirm that intervening on a feature actually changes the output.

**Deletion by resampling, not by a fixed baseline.** Replacing a feature with one value (such as the median) can produce unrealistic rows. Averaging over 50 values drawn from the training data approximates "this feature's information is removed" (an interventional, marginal approach).

**Log-odds rather than probability for the deletion threshold.** Probabilities saturate near 0 and 1. On Telco, a true top feature for a 3%-risk customer moved the probability by only 1.2 points but the log-odds by 0.27. A probability threshold would have rejected a correct claim.

**Strength from rank, not raw SHAP magnitude.** Raw SHAP scale differs across rows and datasets. Rank is comparable everywhere and maps naturally onto "strong / moderate / weak".

**Writer failures count as attempts.** Bad JSON or an API error doesn't crash the audit. It is recorded as a failed attempt with the reason, and the loop continues up to `max_attempts` (default 3).

**Flag, don't hide.** When retries run out, unsupported claims are returned with `supported: false` and their reasons, so a person can see exactly what was and wasn't verified.

**Offline writer for demos and CI.** `TemplateWriter` restates SHAP, so the whole system runs and is tested without a key or network access. It is labelled clearly so its results are never mistaken for an LLM's.

## Testing

The tests use the breast cancer dataset bundled with scikit-learn, and LLM behaviour comes from scripted fake writers: one that is wrong once and then right, one that is always wrong, and one that raises an error. That exercises the retry, give-up and error paths deterministically, with no network.
