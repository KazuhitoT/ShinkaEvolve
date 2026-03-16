"""Reranker score loader — numpy mmap-based, no transformer/torch dependency.

Precomputed reranker scores are stored per model as:
  doc_indices.npy  (num_queries, max_candidates) int32  — -1 = padding
  scores.npy       (num_queries, max_candidates) float32 — sigmoid [0,1]
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np


class RerankerStore:
    """Lazy mmap-backed store for a single reranker model's precomputed scores."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.doc_indices = np.load(
            os.path.join(model_dir, "doc_indices.npy"), mmap_mode="r"
        )
        self.scores = np.load(
            os.path.join(model_dir, "scores.npy"), mmap_mode="r"
        )
        with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as f:
            self.metadata = json.load(f)

    @property
    def model_name(self) -> str:
        return self.metadata["model_name"]

    @property
    def max_candidates(self) -> int:
        return self.doc_indices.shape[1]


class RerankerRegistry:
    """Registry of all available reranker models under a rerankers directory.

    Stores are lazily loaded on first access.
    All numpy arrays use mmap_mode="r" so fork children share page cache.
    """

    def __init__(self, rerankers_dir: str):
        self.rerankers_dir = rerankers_dir
        manifest_path = os.path.join(rerankers_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self._stores: Dict[str, RerankerStore] = {}

    def available_models(self) -> List[str]:
        return list(self.manifest.get("models", {}).keys())

    def get_store(self, model_name: str) -> RerankerStore:
        if model_name not in self._stores:
            model_info = self.manifest["models"][model_name]
            model_dir = os.path.join(self.rerankers_dir, model_info["dir"])
            self._stores[model_name] = RerankerStore(model_dir)
        return self._stores[model_name]


def load_reranker_registry(rerankers_dir: Optional[str] = None) -> Optional[RerankerRegistry]:
    """Convenience loader — returns None if directory doesn't exist or has no manifest."""
    if rerankers_dir is None:
        return None
    if not os.path.isdir(rerankers_dir):
        return None
    manifest_path = os.path.join(rerankers_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    return RerankerRegistry(rerankers_dir)
