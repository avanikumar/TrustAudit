"""
FastAPI service.

Run:
    uvicorn trustaudit.api:app --port 8000
Then open http://localhost:8000/docs

The auditor is trained once at startup on the dataset named by
TRUSTAUDIT_DATASET (default "telco"; "cancer" needs no data file).
"""
import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .agent import Auditor
from .data import load_dataset
from .llm import writer_from_env

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ds = load_dataset(os.environ.get("TRUSTAUDIT_DATASET", "telco"))
    state["dataset"] = ds
    state["auditor"] = Auditor.from_dataset(ds, writer=writer_from_env())
    yield
    state.clear()


app = FastAPI(
    title="TrustAudit",
    description="Predicts, explains, quantifies uncertainty, and verifies every "
                "claim in the explanation against the model.",
    version="0.1.0",
    lifespan=lifespan,
)


class AuditRequest(BaseModel):
    features: dict   # one row: {column name: value}


@app.get("/health")
def health():
    ds = state["dataset"]
    return {"status": "ok", "dataset": ds.name,
            "writer": type(state["auditor"].writer).__name__}


@app.get("/schema")
def schema():
    """Columns the model expects, with an example row from the test set."""
    ds = state["dataset"]
    example = ds.X_test.iloc[0].to_dict()
    return {"columns": list(ds.X_train.columns),
            "example": {k: (None if pd.isna(v) else v) for k, v in example.items()}}


@app.post("/audit")
def audit(req: AuditRequest):
    ds = state["dataset"]
    missing = [c for c in ds.X_train.columns if c not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing columns: {missing}")

    row = pd.DataFrame([{c: req.features[c] for c in ds.X_train.columns}])
    for col in ds.X_train.columns:
        dtype = ds.X_train[col].dtype
        if isinstance(dtype, pd.CategoricalDtype):
            unknown = set(row[col].dropna()) - set(dtype.categories)
            if unknown:
                raise HTTPException(status_code=422,
                                    detail=f"{col}: unknown value(s) {sorted(unknown)}")
        row[col] = row[col].astype(dtype)

    result = state["auditor"].audit(row)
    result.pop("history")
    return result
