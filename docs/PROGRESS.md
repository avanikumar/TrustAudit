# Progress log

Each entry lists what was done and what result it produced. Nothing is claimed in the README that isn't recorded here.

## Data, model, SHAP, conformal
- Telco loader with a stratified 60/20/20 split (seed 42).
- LightGBM baseline: Telco test AUC 0.807 (calibration AUC 0.836).
- Per-row SHAP top features.
- Split conformal prediction on Telco: coverage 88.9% at a 90% target and 95.1% at 95%.

## Package restructure and second dataset
- Moved the code into the `src/trustaudit` package.
- Added the breast cancer dataset (bundled with scikit-learn): test AUC 0.995; coverage 93.0% at 90% and 97.4% at 95% (114-row calibration set).

## Verifier
- Claim schema `{feature, direction, strength}`.
- SHAP check: sign, plus a rank limit for strength (strong ≤ 3, moderate ≤ 6).
- Deletion test: resample the feature from training data 50 times and compare mean log-odds. Switched from probability to log-odds after a correct claim on a 3%-risk row moved the probability by only 0.012.
- Ablation (`results/verifier_*.json`): with both checks, accepts 97.0% / 99.1% of correct claims (Telco / cancer) and rejects 100% of flipped and exaggerated claims. The deletion test alone agrees with SHAP direction on 97% / 100% of top features.

## LLM writer and agent
- OpenAI-compatible writer with retry on 429/5xx/network errors, and JSON parsing that tolerates code fences.
- Offline TemplateWriter for demos and CI.
- LangGraph loop: analyze → write → verify → (retry with feedback | finalize), max 3 attempts.
- Simulated noisy writer (40% corrupted claims): supported claims rise from 59.7% on the first draft to 88.0% at the end on Telco, and from 62.7% to 89.3% on cancer.

## API, tests, CI
- FastAPI service: `/health`, `/schema`, `/audit`, with validation of unknown categorical values.
- 21 pytest tests: core, verifier, LLM parsing, agent loop with scripted writers, API.
- GitHub Actions on Python 3.10 and 3.12.
- Dockerfile written but not yet built or run.

## Next
- Real-LLM evaluation with Groq on both datasets.
- Separate claim-extraction step.
