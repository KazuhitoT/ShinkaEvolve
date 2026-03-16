#!/usr/bin/env python3
"""Precompute reranker scores using CrossEncoder models.

Requires precomputed embeddings (run precompute_embeddings.py first).
Candidates are selected via dense retrieval from all available embedding models,
then scored with each reranker.

Usage:
    python precompute_rerankers.py [--data_dir data] [--models MODEL ...]

Output:
    data/rerankers/
      manifest.json
      cl-nagoya__ruri-v3-reranker-310m/
        doc_indices.npy  (860, max_candidates) int32
        scores.npy       (860, max_candidates) float32
        metadata.json
      hotchpotch__japanese-reranker-tiny-v2/
        ...
"""

import argparse
import json
import os
import time

import numpy as np

DEFAULT_MODELS = [
    "cl-nagoya/ruri-v3-reranker-310m",
    "hotchpotch/japanese-reranker-tiny-v2",
]


def _safe_dir_name(name: str) -> str:
    return name.replace("/", "__")


def _load_corpus(corpus_path: str):
    """Load corpus as list of dicts."""
    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return corpus


def _load_queries(queries_path: str):
    """Load queries as list of dicts."""
    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def _get_candidates_for_query(embedding_registry, qi, per_model_top_n=200):
    """Get union of top-N candidates from all embedding models for one query."""
    all_candidates = set()
    for model_name in embedding_registry.available_models():
        store = embedding_registry.get_store(model_name)
        q_emb = store.queries[qi]
        sims = store.corpus @ q_emb
        top_n = min(per_model_top_n, len(sims))
        top_indices = np.argpartition(-sims, top_n)[:top_n]
        all_candidates.update(top_indices.tolist())
    return sorted(all_candidates)


def compute_reranker_scores(
    model_name: str,
    queries: list,
    corpus: list,
    embedding_registry,
    output_dir: str,
    per_model_top_n: int = 200,
    batch_size: int = 64,
):
    """Compute and save reranker scores for one model."""
    from sentence_transformers import CrossEncoder

    print(f"\nLoading reranker: {model_name}")
    # Force CPU to avoid MPS buffer size limits on Apple Silicon
    model = CrossEncoder(model_name, device="cpu")

    num_queries = len(queries)

    # First pass: gather candidates for all queries
    print(f"Gathering candidates (top-{per_model_top_n} per embedding model) ...")
    all_candidates = []
    max_candidates = 0
    for qi in range(num_queries):
        cands = _get_candidates_for_query(embedding_registry, qi, per_model_top_n)
        all_candidates.append(cands)
        max_candidates = max(max_candidates, len(cands))

    print(f"  Max candidates per query: {max_candidates}")
    total_pairs = sum(len(c) for c in all_candidates)
    print(f"  Total (query, doc) pairs to score: {total_pairs:,}")

    # Allocate output arrays (padded to max_candidates)
    doc_indices = np.full((num_queries, max_candidates), -1, dtype=np.int32)
    scores = np.zeros((num_queries, max_candidates), dtype=np.float32)

    # Second pass: score candidates per query
    t0 = time.time()
    for qi in range(num_queries):
        cands = all_candidates[qi]
        if not cands:
            continue

        query_text = queries[qi]["query"]
        pairs = []
        for doc_idx in cands:
            doc = corpus[doc_idx]
            doc_text = doc.get("title", "") + " " + doc["text"]
            pairs.append((query_text, doc_text))

        # CrossEncoder outputs sigmoid scores in [0, 1]
        raw_scores = model.predict(pairs, batch_size=batch_size)

        # Sort by score descending
        sorted_order = np.argsort(-raw_scores)
        n = len(cands)
        doc_indices[qi, :n] = [cands[j] for j in sorted_order]
        scores[qi, :n] = raw_scores[sorted_order]

        if (qi + 1) % 100 == 0 or qi == num_queries - 1:
            elapsed = time.time() - t0
            qps = (qi + 1) / elapsed
            print(f"  Query {qi + 1}/{num_queries}  "
                  f"({qps:.1f} q/s, elapsed {elapsed:.0f}s)")

    total_time = time.time() - t0
    print(f"  Done in {total_time:.1f}s")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "doc_indices.npy"), doc_indices)
    np.save(os.path.join(output_dir, "scores.npy"), scores)

    metadata = {
        "model_name": model_name,
        "num_queries": num_queries,
        "max_candidates": max_candidates,
        "per_model_top_n": per_model_top_n,
        "total_pairs_scored": total_pairs,
        "computation_time_s": round(total_time, 1),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Precompute reranker scores using CrossEncoder models"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Data directory (default: <script_dir>/data)",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="CrossEncoder model names",
    )
    parser.add_argument(
        "--per_model_top_n", type=int, default=200,
        help="Top-N candidates per embedding model for candidate pool",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Batch size for CrossEncoder inference",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "data")
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries_eval.jsonl")
    embeddings_dir = os.path.join(data_dir, "embeddings")
    rerankers_dir = os.path.join(data_dir, "rerankers")

    # Load embedding registry (required for candidate selection)
    from embedding_loader import load_registry

    embedding_registry = load_registry(embeddings_dir)
    if embedding_registry is None:
        print(f"ERROR: No embeddings found at {embeddings_dir}")
        print("Run precompute_embeddings.py first.")
        return
    print(f"Embedding models: {embedding_registry.available_models()}")

    # Load corpus and queries
    print("Loading corpus ...")
    corpus = _load_corpus(corpus_path)
    queries = _load_queries(queries_path)
    print(f"Corpus: {len(corpus)} docs, Queries: {len(queries)}")

    # Load or create manifest
    manifest_path = os.path.join(rerankers_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"models": {}}

    for model_name in args.models:
        safe_name = _safe_dir_name(model_name)
        output_dir = os.path.join(rerankers_dir, safe_name)
        metadata = compute_reranker_scores(
            model_name, queries, corpus, embedding_registry, output_dir,
            per_model_top_n=args.per_model_top_n,
            batch_size=args.batch_size,
        )
        manifest["models"][model_name] = {
            "dir": safe_name,
            "max_candidates": metadata["max_candidates"],
        }

    os.makedirs(rerankers_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved to {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
