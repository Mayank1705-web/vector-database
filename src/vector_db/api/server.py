from __future__ import annotations
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from vector_db.api.index import VectorIndex
from vector_db.approximate.ivf.index import IVFIndex


ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT_DIR / "configs" / "default.yaml"
DATASET_PATH = ROOT_DIR / "data" / "processed" / "dataset.npz"


class InsertRequest(BaseModel):
    id: int = Field(..., description="Unique vector ID")
    vector: list[float] = Field(..., min_length=1)


class SearchRequest(BaseModel):
    vector: list[float] = Field(..., min_length=1)
    k: int = Field(..., gt=0)


class SearchResultResponse(BaseModel):
    id: int
    score: float


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_index() -> VectorIndex:
    config = load_config()

    dataset = np.load(DATASET_PATH)
    vectors = dataset["vectors"]

    ivf_config = config["ivf"]

    backend = IVFIndex(
        n_clusters=ivf_config["n_clusters"],
        n_probe=ivf_config["n_probe"],
        kmeans_iterations=ivf_config["kmeans_iterations"],
        seed=config["dataset"]["seed"],
    )

    backend.fit(vectors)

    return VectorIndex(backend)


app = FastAPI(
    title="Vector Database API",
    description="Lightweight HTTP API around the from-scratch vector database.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

index = build_index()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/insert")
def insert(request: InsertRequest) -> dict[str, object]:
    try:
        index.insert(request.id, request.vector)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "inserted",
        "id": request.id,
    }


@app.post("/search", response_model=list[SearchResultResponse])
def search(request: SearchRequest) -> list[SearchResultResponse]:
    try:
        results = index.search(request.vector, request.k)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return [
        SearchResultResponse(
            id=result.id,
            score=result.score,
        )
        for result in results
    ]


@app.delete("/delete/{vector_id}")
def delete(vector_id: int) -> dict[str, object]:
    try:
        index.delete(vector_id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "deleted",
        "id": vector_id,
    }