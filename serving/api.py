"""
FastAPI serving layer for Discovery Engine Stage 1.

Endpoints:
  POST /recommend
  POST /search
  POST /complete-the-look
  GET  /health
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.candidate_agent import CandidateAgent
from agents.complete_the_look_agent import CompleteTheLookAgent
from agents.search_agent import SearchAgent
from index.faiss_store import FaissStore
from models.discovery_model import DiscoveryModel
from models.item_tower import ItemTower

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
CONFIG_PATH = ROOT / "configs" / "config.yaml"

app = FastAPI(title="Discovery Engine API", version="1.0.0")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {"embedding_dim": 128, "hidden_dim": 256, "max_history": 10, "temperature": 0.05}


def _bootstrap():
    global candidate_agent, search_agent, complete_agent, user_map, item_map

    config = _load_config()
    device = _device()
    logger.info("Using device: %s", device)

    with open(ARTIFACTS / "user_map.pkl", "rb") as f:
        user_map = pickle.load(f)
    with open(ARTIFACTS / "item_map.pkl", "rb") as f:
        item_map = pickle.load(f)
    with open(ARTIFACTS / "user_interactions.pkl", "rb") as f:
        user_interactions = pickle.load(f)
    with open(ARTIFACTS / "user_histories.pkl", "rb") as f:
        user_histories = pickle.load(f)
    with open(ARTIFACTS / "reverse_item_map.pkl", "rb") as f:
        reverse_item_map = pickle.load(f)

    item_emb = np.load(ARTIFACTS / "item_embeddings.npy")
    item_meta = pd.read_json(ARTIFACTS / "item_meta.json").to_dict("records")

    item_tower = ItemTower(output_dim=config.get("embedding_dim", 128))
    model = DiscoveryModel(
        num_items=len(item_map),
        item_embedding_table=torch.from_numpy(item_emb).float(),
        embedding_dim=config.get("embedding_dim", 128),
        hidden_dim=config.get("hidden_dim", 256),
        max_history=config.get("max_history", 10),
        item_tower=item_tower,
    )

    ckpt_path = ARTIFACTS / "discovery_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        item_tower.load_state_dict(ckpt["item_tower_state"])
        logger.info("Loaded checkpoint %s", ckpt_path.name)

    model.to(device)
    model.eval()
    item_tower.to(device)

    faiss_store = FaissStore.load(ARTIFACTS / "faiss.index")

    candidate_agent = CandidateAgent(
        model=model,
        faiss_store=faiss_store,
        user_histories=user_histories,
        user_interactions=user_interactions,
        reverse_item_map=reverse_item_map,
        device=device,
        max_history=config.get("max_history", 10),
    )
    search_agent = SearchAgent(item_tower=item_tower, faiss_store=faiss_store, device=device)
    complete_agent = CompleteTheLookAgent(
        faiss_store=faiss_store,
        item_embeddings=item_emb,
        item_meta=item_meta,
    )


# Lazy init on first request if artifacts missing at import
candidate_agent = None
search_agent = None
complete_agent = None
user_map = {}
item_map = {}


class RecommendRequest(BaseModel):
    user_id: str | int
    top_k: int = Field(default=10, ge=1, le=100)


class SearchRequest(BaseModel):
    query_text: Optional[str] = None
    query_image: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=100)


class CompleteTheLookRequest(BaseModel):
    seed_item_idx: int = Field(ge=0)
    top_k: int = Field(default=10, ge=1, le=100)


@app.on_event("startup")
def startup():
    if (ARTIFACTS / "faiss.index").exists():
        _bootstrap()


@app.get("/health")
def health():
    ready = candidate_agent is not None
    return {"status": "ok" if ready else "starting", "artifacts_loaded": ready}


@app.post("/recommend")
def recommend(request: RecommendRequest):
    # TODO(stage2): Add guardrails — toxicity filter, brand safety, budget caps
    if candidate_agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded; run make train && make index")
    try:
        return candidate_agent.recommend(request.user_id, user_map, request.top_k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/search")
def search(request: SearchRequest):
    # TODO(stage2): Query moderation and PII redaction
    if search_agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return search_agent.search(request.query_text, request.query_image, request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/complete-the-look")
def complete_the_look(request: CompleteTheLookRequest):
    # TODO(stage2): Style coherence scoring and inventory-aware filtering
    if complete_agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        return complete_agent.complete(request.seed_item_idx, request.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
