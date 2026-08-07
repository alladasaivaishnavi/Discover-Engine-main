"""
Complete-the-look agent: seed item → complementary categories + diversity rules.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

import numpy as np
import torch

from index.faiss_store import FaissStore

# Category complementarity map for outfit building
COMPLEMENTARY_CATEGORIES: dict[str, list[str]] = {
    "Topwear": ["Bottomwear", "Footwear", "Accessories", "Outerwear"],
    "Bottomwear": ["Topwear", "Footwear", "Accessories", "Outerwear"],
    "Dress": ["Footwear", "Accessories", "Outerwear"],
    "Footwear": ["Topwear", "Bottomwear", "Dress", "Accessories"],
    "Accessories": ["Topwear", "Bottomwear", "Dress", "Footwear"],
    "Outerwear": ["Topwear", "Bottomwear", "Dress", "Footwear"],
}

MAX_CATEGORY_FRACTION = 0.35


class CompleteTheLookAgent:
    def __init__(
        self,
        faiss_store: FaissStore,
        item_embeddings: np.ndarray,
        item_meta: list[dict],
    ):
        self.faiss_store = faiss_store
        self.item_embeddings = item_embeddings.astype("float32")
        self.item_meta = item_meta
        self.idx_to_category = {m["item_idx"]: m.get("category", "Unknown") for m in item_meta}

    def _seed_vector(self, seed_item_idx: int) -> np.ndarray:
        return self.item_embeddings[seed_item_idx].reshape(1, -1)

    def _is_complementary(self, seed_category: str, candidate_category: str) -> bool:
        if seed_category == candidate_category:
            return False
        allowed = COMPLEMENTARY_CATEGORIES.get(seed_category, [])
        return candidate_category in allowed or seed_category in COMPLEMENTARY_CATEGORIES.get(
            candidate_category, []
        )

    def _apply_diversity(self, candidates: list[int], top_k: int) -> list[int]:
        """Ensure no single category exceeds 35% of results."""
        max_per_category = max(1, int(top_k * MAX_CATEGORY_FRACTION + 0.999))
        selected: list[int] = []
        counts: Counter = Counter()

        for idx in candidates:
            cat = self.idx_to_category.get(idx, "Unknown")
            if counts[cat] >= max_per_category:
                continue
            selected.append(idx)
            counts[cat] += 1
            if len(selected) >= top_k:
                break

        return selected

    def complete(
        self,
        seed_item_idx: int,
        top_k: int = 10,
        fetch_multiplier: int = 10,
    ) -> dict[str, Any]:
        start = time.time()

        if seed_item_idx < 0 or seed_item_idx >= len(self.item_embeddings):
            raise ValueError(f"Invalid seed_item_idx: {seed_item_idx}")

        seed_category = self.idx_to_category.get(seed_item_idx, "Unknown")
        seed_vec = self._seed_vector(seed_item_idx)

        fetch_k = min(top_k * fetch_multiplier, self.faiss_store.ntotal)
        _, indices = self.faiss_store.search(seed_vec, fetch_k, exclude={seed_item_idx})

        filtered = [
            int(i)
            for i in indices[0]
            if i >= 0 and self._is_complementary(seed_category, self.idx_to_category.get(i, ""))
        ]

        recommendations = self._apply_diversity(filtered, top_k)

        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "seed_item_idx": seed_item_idx,
            "seed_category": seed_category,
            "recommendations": recommendations,
            "latency_ms": latency_ms,
        }
