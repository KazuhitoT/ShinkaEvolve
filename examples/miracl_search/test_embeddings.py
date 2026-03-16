#!/usr/bin/env python3
"""Embedding テスト — 各モデルのロード・類似度検索・正解照合を検証する。

Usage:
    python test_embeddings.py
"""

import json
import os
import sys
import time

import numpy as np

# embedding_loader は同ディレクトリにあるので sys.path に追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_loader import EmbeddingRegistry, load_registry


def load_queries_with_annotations(queries_path: str):
    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_corpus_docids(corpus_path: str):
    docids = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docids.append(json.loads(line)["docid"])
    return docids


def test_model(
    model_name: str,
    registry: EmbeddingRegistry,
    queries: list,
    corpus_docids: list,
    top_k: int = 10,
):
    """1モデルのテスト: ロード、形状確認、類似度検索、Recall/DCG 計算。"""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # 1. ロード
    t0 = time.time()
    store = registry.get_store(model_name)
    load_time = time.time() - t0
    print(f"  Load time:  {load_time:.3f}s")

    # 2. 形状・型チェック
    print(f"  Corpus:     {store.corpus.shape}  dtype={store.corpus.dtype}")
    print(f"  Queries:    {store.queries.shape}  dtype={store.queries.dtype}")
    print(f"  Dimension:  {store.dimension}")

    assert store.corpus.shape[0] == len(corpus_docids), (
        f"Corpus count mismatch: {store.corpus.shape[0]} != {len(corpus_docids)}"
    )
    assert store.queries.shape[0] == len(queries), (
        f"Query count mismatch: {store.queries.shape[0]} != {len(queries)}"
    )
    assert store.corpus.shape[1] == store.dimension
    assert store.queries.shape[1] == store.dimension

    # 3. L2 正規化チェック（サンプル）
    sample_indices = [0, len(corpus_docids) // 2, len(corpus_docids) - 1]
    norms = [np.linalg.norm(store.corpus[i]) for i in sample_indices]
    print(f"  L2 norms (corpus sample): {[f'{n:.6f}' for n in norms]}")
    for i, n in zip(sample_indices, norms):
        assert abs(n - 1.0) < 0.01, f"Corpus[{i}] not normalized: norm={n}"

    q_norms = [np.linalg.norm(store.queries[i]) for i in [0, len(queries) - 1]]
    print(f"  L2 norms (query sample):  {[f'{n:.6f}' for n in q_norms]}")
    for n in q_norms:
        assert abs(n - 1.0) < 0.01, f"Query not normalized: norm={n}"

    # 4. 全クエリで dense retrieval → Recall@k, DCG@k
    print(f"  Running dense retrieval (top_{top_k}) for {len(queries)} queries ...")
    t0 = time.time()

    recall_scores = []
    dcg_scores = []
    hit_counts = []

    for qi, q in enumerate(queries):
        q_emb = store.queries[qi]
        # cosine similarity (= dot product for L2-normalized vectors)
        sims = store.corpus @ q_emb
        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]
        retrieved_docids = [corpus_docids[idx] for idx in top_indices]

        relevant = {
            a["docid"] for a in q["annotations"] if a["relevance"] > 0
        }

        # Recall@k
        hits = len(set(retrieved_docids) & relevant)
        hit_counts.append(hits)
        recall = hits / len(relevant) if relevant else 0.0
        recall_scores.append(recall)

        # DCG@k
        dcg = 0.0
        for rank, docid in enumerate(retrieved_docids):
            if docid in relevant:
                dcg += 1.0 / np.log2(rank + 2)
        dcg_scores.append(dcg)

    elapsed = time.time() - t0
    mean_recall = np.mean(recall_scores)
    mean_dcg = np.mean(dcg_scores)
    total_hits = sum(hit_counts)
    queries_with_hit = sum(1 for h in hit_counts if h > 0)

    print(f"  Search time:        {elapsed:.3f}s ({elapsed/len(queries)*1000:.1f}ms/query)")
    print(f"  Mean Recall@{top_k}:    {mean_recall:.4f}")
    print(f"  Mean DCG@{top_k}:       {mean_dcg:.4f}")
    print(f"  Queries with hit:   {queries_with_hit}/{len(queries)}")
    print(f"  Total hits:         {total_hits}")

    # 5. 自己類似度チェック（query[0] と自分自身）
    self_sim = float(np.dot(store.queries[0], store.queries[0]))
    print(f"  Self-similarity:    {self_sim:.6f} (should be ~1.0)")
    assert abs(self_sim - 1.0) < 0.01

    return {
        "model": model_name,
        "dimension": store.dimension,
        "mean_recall": round(mean_recall, 4),
        "mean_dcg": round(mean_dcg, 4),
        "queries_with_hit": queries_with_hit,
        "search_time": round(elapsed, 3),
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    embeddings_dir = os.path.join(script_dir, "data", "embeddings")
    queries_path = os.path.join(script_dir, "data", "queries_eval.jsonl")
    corpus_path = os.path.join(script_dir, "data", "corpus.jsonl")

    # ロード
    registry = load_registry(embeddings_dir)
    if registry is None:
        print(f"ERROR: No embeddings found at {embeddings_dir}")
        sys.exit(1)

    models = registry.available_models()
    print(f"Embeddings dir: {embeddings_dir}")
    print(f"Available models: {models}")

    queries = load_queries_with_annotations(queries_path)
    print(f"Queries: {len(queries)}")

    print("Loading corpus docids ...")
    corpus_docids = load_corpus_docids(corpus_path)
    print(f"Corpus: {len(corpus_docids)} docs")

    # 各モデルをテスト
    results = []
    for model_name in models:
        r = test_model(model_name, registry, queries, corpus_docids)
        results.append(r)

    # サマリ
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Dim':>4} {'Recall@10':>10} {'DCG@10':>8} {'Time':>7}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['model']:<45} {r['dimension']:>4} "
            f"{r['mean_recall']:>10.4f} {r['mean_dcg']:>8.4f} "
            f"{r['search_time']:>6.3f}s"
        )

    print(f"\nAll {len(results)} models passed.")


if __name__ == "__main__":
    main()
