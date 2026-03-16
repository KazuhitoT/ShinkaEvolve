#!/usr/bin/env python3
"""Precompute corpus & query embeddings using HuggingFace SentenceTransformers.

Usage:
    python precompute_embeddings.py [--data_dir data] [--models MODEL ...]

Output:
    data/embeddings/
      manifest.json
      cl-nagoya__ruri-v3-30m/
        corpus.npy, queries.npy, metadata.json
      hotchpotch__static-embedding-japanese/
        corpus.npy, queries.npy, metadata.json
"""

import argparse
import gc
import json
import os
import time

import numpy as np

DEFAULT_MODELS = [
    "cl-nagoya/ruri-v3-30m",
    "hotchpotch/static-embedding-japanese",
]


def _safe_dir_name(model_name: str) -> str:
    """Convert model name to safe directory name (replace / with __)."""
    return model_name.replace("/", "__")


def _load_corpus_texts(corpus_path: str):
    """Load corpus docids and texts (title + ' ' + text)."""
    docids = []
    texts = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docids.append(doc["docid"])
            texts.append(doc.get("title", "") + " " + doc["text"])
    return docids, texts


def _load_query_texts(queries_path: str):
    """Load query ids and texts."""
    query_ids = []
    texts = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            query_ids.append(q["query_id"])
            texts.append(q["query"])
    return query_ids, texts


def _add_prefix(texts, model_name, text_type):
    """Add model-specific prefix to texts (e.g. ruri requires prefixes)."""
    if "ruri" in model_name.lower():
        if text_type == "corpus":
            return ["検索文書: " + t for t in texts]
        else:
            return ["検索クエリ: " + t for t in texts]
    return texts


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings in-place."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings /= norms
    return embeddings


def _encode_chunked(model, texts, batch_size=32, chunk_size=10000):
    """Encode in chunks to limit peak memory inside SentenceTransformer.encode()."""
    chunks = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        emb = model.encode(
            chunk, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )
        chunks.append(emb)
    return np.vstack(chunks)


def compute_embeddings(
    model_name: str,
    corpus_texts: list,
    query_texts: list,
    output_dir: str,
    batch_size: int = 32,
):
    """Compute and save embeddings for one model."""
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading model: {model_name}")
    # Force CPU to avoid MPS buffer size limits on Apple Silicon.
    # StaticEmbedding models are faster on CPU anyway; small Transformer
    # models (ruri-v3-30m, 37M params) also run fine on CPU.
    model = SentenceTransformer(model_name, device="cpu")

    # Corpus embeddings
    print(f"Encoding corpus ({len(corpus_texts)} docs) ...")
    prefixed_corpus = _add_prefix(corpus_texts, model_name, "corpus")
    t0 = time.time()
    corpus_emb = _encode_chunked(model, prefixed_corpus, batch_size=batch_size)
    del prefixed_corpus
    corpus_emb = _l2_normalize(corpus_emb)
    print(f"  Corpus done in {time.time() - t0:.1f}s, shape={corpus_emb.shape}")

    # Query embeddings
    print(f"Encoding queries ({len(query_texts)} queries) ...")
    prefixed_queries = _add_prefix(query_texts, model_name, "query")
    t0 = time.time()
    query_emb = _encode_chunked(model, prefixed_queries, batch_size=batch_size)
    del prefixed_queries
    query_emb = _l2_normalize(query_emb)
    print(f"  Queries done in {time.time() - t0:.1f}s, shape={query_emb.shape}")

    # Release model before saving
    del model
    gc.collect()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "corpus.npy"), corpus_emb)
    np.save(os.path.join(output_dir, "queries.npy"), query_emb)

    metadata = {
        "model_name": model_name,
        "dimension": int(corpus_emb.shape[1]),
        "corpus_count": int(corpus_emb.shape[0]),
        "query_count": int(query_emb.shape[0]),
        "dtype": "float32",
        "normalized": True,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_dir}")
    return metadata


def compute_corpus_only(
    model_name: str,
    corpus_texts: list,
    output_dir: str,
    batch_size: int = 32,
):
    """Compute and save corpus embeddings only (skip queries)."""
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")

    print(f"Encoding corpus ({len(corpus_texts)} docs) ...")
    prefixed_corpus = _add_prefix(corpus_texts, model_name, "corpus")
    t0 = time.time()
    corpus_emb = _encode_chunked(model, prefixed_corpus, batch_size=batch_size)
    del prefixed_corpus
    corpus_emb = _l2_normalize(corpus_emb)
    print(f"  Corpus done in {time.time() - t0:.1f}s, shape={corpus_emb.shape}")

    # Release model before saving
    del model
    gc.collect()

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "corpus.npy"), corpus_emb)

    metadata = {
        "model_name": model_name,
        "dimension": int(corpus_emb.shape[1]),
        "corpus_count": int(corpus_emb.shape[0]),
        "query_count": 0,
        "dtype": "float32",
        "normalized": True,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embeddings using HuggingFace models"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (default: <script_dir>/data)",
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        default=None,
        help="Custom corpus JSONL file (default: <data_dir>/corpus.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Custom output directory (default: <data_dir>/embeddings)",
    )
    parser.add_argument(
        "--skip_queries",
        action="store_true",
        help="Skip query embedding computation (corpus only)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="HuggingFace model names",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "data")
    corpus_path = args.corpus_file or os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries_eval.jsonl")
    embeddings_dir = args.output_dir or os.path.join(data_dir, "embeddings")

    print(f"Corpus: {corpus_path}")
    if not args.skip_queries:
        print(f"Queries: {queries_path}")
    print(f"Output: {embeddings_dir}")

    _, corpus_texts = _load_corpus_texts(corpus_path)
    print(f"Loaded {len(corpus_texts)} corpus docs")

    query_texts = None
    if not args.skip_queries:
        _, query_texts = _load_query_texts(queries_path)
        print(f"Loaded {len(query_texts)} queries")

    # Load or create manifest
    manifest_path = os.path.join(embeddings_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"models": {}}

    for model_name in args.models:
        safe_name = _safe_dir_name(model_name)
        output_dir = os.path.join(embeddings_dir, safe_name)

        if args.skip_queries:
            metadata = compute_corpus_only(
                model_name, corpus_texts, output_dir, args.batch_size
            )
        else:
            metadata = compute_embeddings(
                model_name, corpus_texts, query_texts, output_dir, args.batch_size
            )

        manifest["models"][model_name] = {
            "dir": safe_name,
            "dimension": metadata["dimension"],
        }

    # Save manifest
    os.makedirs(embeddings_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved to {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
