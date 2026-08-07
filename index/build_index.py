"""
Build FAISS IndexFlatIP from item tower embeddings.

Adapted from external/two-tower-retrieval-system/src/indexing/build_index.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


def build_index_from_embeddings(embeddings: np.ndarray) -> faiss.Index:
    """
    Build IndexFlatIP from precomputed L2-normalized item vectors.

    Args:
        embeddings: (num_items, D) float32, L2-normalized
    """
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    logger.info("FAISS index built with %d vectors (dim=%d)", index.ntotal, embeddings.shape[1])
    return index


def build_faiss_index(
    model,
    num_items: int,
    device: torch.device,
    item_embedding_table: torch.Tensor | None = None,
) -> faiss.Index:
    """
    Extract item embeddings and build FAISS inner-product index.

    Supports DiscoveryModel (embedding table) or any model with encode_items().
    """
    model.eval()
    with torch.no_grad():
        if item_embedding_table is not None:
            embeddings = item_embedding_table.cpu().numpy().astype("float32")
        else:
            item_ids = torch.arange(num_items).to(device)
            item_embeddings = model.encode_items(item_ids)
            embeddings = item_embeddings.cpu().numpy().astype("float32")

    logger.info("Item embeddings shape: %s", embeddings.shape)
    return build_index_from_embeddings(embeddings)


def save_index(index: faiss.Index, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    logger.info("Saved FAISS index to %s", path)


def load_index(path: Path | str) -> faiss.Index:
    return faiss.read_index(str(path))
