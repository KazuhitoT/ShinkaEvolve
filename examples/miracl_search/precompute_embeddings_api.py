#!/usr/bin/env python3
"""Precompute corpus & query embeddings using OpenAI / Gemini APIs.

Usage:
    python precompute_embeddings_api.py [--data_dir data] [--models MODEL ...]

Requires API keys:
    - OPENAI_API_KEY for text-embedding-3-small
    - GOOGLE_API_KEY (or GEMINI_API_KEY) for gemini-embedding-001
"""

import argparse
import json
import os
import time

import numpy as np

MODELS = {
    "text-embedding-3-small": {"provider": "openai", "dim": 1536},
    "gemini-embedding-001": {"provider": "google", "dim": 768},
}


def _safe_dir_name(model_name: str) -> str:
    return model_name.replace("/", "__")


def _load_corpus_texts(corpus_path: str):
    docids, texts = [], []
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
    query_ids, texts = [], []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line)
            query_ids.append(q["query_id"])
            texts.append(q["query"])
    return query_ids, texts


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings /= norms
    return embeddings


def _is_token_limit_error(err_str: str) -> bool:
    """Check if the error is a token/context length limit error."""
    return ("max_tokens_per_request" in err_str
            or "maximum context length" in err_str)


def _truncate_text(text: str, max_chars: int = 6000) -> str:
    """Truncate text to roughly fit within model token limits.

    For cl100k_base (used by text-embedding-3-small), Japanese text
    averages ~1-2 tokens per character, so 6000 chars ≈ ~6K-12K tokens.
    We start conservative and let the retry loop reduce further if needed.
    """
    return text[:max_chars]


def _encode_openai(texts: list, model: str, batch_size: int = 512) -> np.ndarray:
    """Encode texts using OpenAI embeddings API.

    Uses adaptive batch splitting: if a batch exceeds the API token limit
    (300K tokens), it is halved and retried automatically. If a single text
    exceeds the model's context length, it is truncated.
    """
    from openai import OpenAI

    client = OpenAI()
    all_embeddings = []

    # Process in a queue to allow splitting oversized batches
    queue = []
    for i in range(0, len(texts), batch_size):
        queue.append(texts[i : i + batch_size])

    batch_num = 0
    total_batches = len(queue)
    while queue:
        batch = queue.pop(0)
        batch_num += 1
        print(f"  OpenAI batch {batch_num}/{total_batches} ({len(batch)} texts)")

        for attempt in range(5):
            try:
                response = client.embeddings.create(input=batch, model=model)
                batch_emb = [item.embedding for item in response.data]
                all_embeddings.extend(batch_emb)
                break
            except Exception as e:
                err_str = str(e)
                if _is_token_limit_error(err_str):
                    if len(batch) > 1:
                        # Split batch in half and re-queue
                        mid = len(batch) // 2
                        queue.insert(0, batch[mid:])
                        queue.insert(0, batch[:mid])
                        total_batches += 1
                        print(f"    Token limit exceeded, splitting batch "
                              f"into {mid} + {len(batch) - mid}")
                        break
                    else:
                        # Single text too long — truncate it
                        max_chars = len(batch[0]) // 2
                        batch[0] = _truncate_text(batch[0], max_chars)
                        print(f"    Single text too long, truncating to "
                              f"{max_chars} chars")
                        continue
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/5 after error: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"OpenAI API failed after 5 retries")

    return np.array(all_embeddings, dtype=np.float32)


def _encode_gemini(
    texts: list, model: str, dim: int, task_type: str = "RETRIEVAL_DOCUMENT",
    batch_size: int = 100,
) -> np.ndarray:
    """Encode texts using Gemini embeddings API."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.getenv('GEMINI_API_KEY')
    )
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        print(f"  Gemini batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} "
              f"({len(batch)} texts)")

        for attempt in range(5):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=dim,
                        task_type=task_type,
                    ),
                )
                batch_emb = [e.values for e in result.embeddings]
                all_embeddings.extend(batch_emb)
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    Retry {attempt + 1}/5 after error: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Gemini API failed after 5 retries at batch {i}")

    return np.array(all_embeddings, dtype=np.float32)


def compute_embeddings_api(
    model_name: str,
    corpus_texts: list,
    query_texts: list,
    output_dir: str,
):
    """Compute and save embeddings for one API model."""
    model_info = MODELS[model_name]
    provider = model_info["provider"]
    dim = model_info["dim"]

    print(f"\nEncoding with {model_name} (provider={provider}, dim={dim})")

    if provider == "openai":
        encode_corpus = lambda texts: _encode_openai(texts, model_name)
        encode_queries = encode_corpus
    elif provider == "google":
        encode_corpus = lambda texts: _encode_gemini(
            texts, model_name, dim, task_type="RETRIEVAL_DOCUMENT",
        )
        encode_queries = lambda texts: _encode_gemini(
            texts, model_name, dim, task_type="RETRIEVAL_QUERY",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Corpus
    print(f"Encoding corpus ({len(corpus_texts)} docs) ...")
    t0 = time.time()
    corpus_emb = encode_corpus(corpus_texts)
    corpus_emb = _l2_normalize(corpus_emb)
    print(f"  Corpus done in {time.time() - t0:.1f}s, shape={corpus_emb.shape}")

    # Queries
    print(f"Encoding queries ({len(query_texts)} queries) ...")
    t0 = time.time()
    query_emb = encode_queries(query_texts)
    query_emb = _l2_normalize(query_emb)
    print(f"  Queries done in {time.time() - t0:.1f}s, shape={query_emb.shape}")

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
        "provider": provider,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_dir}")
    return metadata


def compute_corpus_only_api(
    model_name: str,
    corpus_texts: list,
    output_dir: str,
):
    """Compute and save corpus embeddings only (skip queries) via API."""
    model_info = MODELS[model_name]
    provider = model_info["provider"]
    dim = model_info["dim"]

    print(f"\nEncoding corpus with {model_name} (provider={provider}, dim={dim})")

    if provider == "openai":
        encode_fn = lambda texts: _encode_openai(texts, model_name)
    elif provider == "google":
        encode_fn = lambda texts: _encode_gemini(
            texts, model_name, dim, task_type="RETRIEVAL_DOCUMENT",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

    print(f"Encoding corpus ({len(corpus_texts)} docs) ...")
    t0 = time.time()
    corpus_emb = encode_fn(corpus_texts)
    corpus_emb = _l2_normalize(corpus_emb)
    print(f"  Corpus done in {time.time() - t0:.1f}s, shape={corpus_emb.shape}")

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "corpus.npy"), corpus_emb)

    metadata = {
        "model_name": model_name,
        "dimension": int(corpus_emb.shape[1]),
        "corpus_count": int(corpus_emb.shape[0]),
        "query_count": 0,
        "dtype": "float32",
        "normalized": True,
        "provider": provider,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"  Saved to {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Precompute embeddings using OpenAI / Gemini APIs"
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
        default=list(MODELS.keys()),
        help="API model names",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "data")
    corpus_path = args.corpus_file or os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries_eval.jsonl")
    embeddings_dir = args.output_dir or os.path.join(data_dir, "embeddings")

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
        if model_name not in MODELS:
            print(f"Skipping unknown model: {model_name}")
            continue
        safe_name = _safe_dir_name(model_name)
        output_dir = os.path.join(embeddings_dir, safe_name)

        if args.skip_queries:
            metadata = compute_corpus_only_api(
                model_name, corpus_texts, output_dir
            )
        else:
            metadata = compute_embeddings_api(
                model_name, corpus_texts, query_texts, output_dir
            )

        manifest["models"][model_name] = {
            "dir": safe_name,
            "dimension": metadata["dimension"],
        }

    os.makedirs(embeddings_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest saved to {manifest_path}")
    print("Done!")


if __name__ == "__main__":
    main()
