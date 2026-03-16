"""Embedding loader — numpy mmap-based, no transformer/torch dependency."""

import json
import os
from typing import Dict, List, Optional


class EmbeddingStore:
    """Lazy mmap-backed store for a single embedding model's vectors."""

    def __init__(self, model_dir: str):
        import numpy as np

        self.model_dir = model_dir
        self.corpus = np.load(
            os.path.join(model_dir, "corpus.npy"), mmap_mode="r"
        )
        self.queries = np.load(
            os.path.join(model_dir, "queries.npy"), mmap_mode="r"
        )
        with open(os.path.join(model_dir, "metadata.json"), encoding="utf-8") as f:
            self.metadata = json.load(f)

    @property
    def dimension(self) -> int:
        return self.metadata["dimension"]

    @property
    def model_name(self) -> str:
        return self.metadata["model_name"]


class EmbeddingRegistry:
    """Registry of all available embedding models under an embeddings directory.

    Stores are lazily loaded on first access to minimise startup cost.
    All numpy arrays use mmap_mode="r" so fork children share page cache (COW).
    """

    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = embeddings_dir
        manifest_path = os.path.join(embeddings_dir, "manifest.json")
        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        self._stores: Dict[str, EmbeddingStore] = {}

    def available_models(self) -> List[str]:
        """Return list of model names that have precomputed embeddings."""
        return list(self.manifest.get("models", {}).keys())

    def get_store(self, model_name: str) -> EmbeddingStore:
        """Get (or lazily load) the EmbeddingStore for a model."""
        if model_name not in self._stores:
            model_info = self.manifest["models"][model_name]
            model_dir = os.path.join(self.embeddings_dir, model_info["dir"])
            self._stores[model_name] = EmbeddingStore(model_dir)
        return self._stores[model_name]


def load_registry(embeddings_dir: Optional[str] = None) -> Optional[EmbeddingRegistry]:
    """Convenience loader — returns None if directory doesn't exist or has no manifest."""
    if embeddings_dir is None:
        return None
    if not os.path.isdir(embeddings_dir):
        return None
    manifest_path = os.path.join(embeddings_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    return EmbeddingRegistry(embeddings_dir)
