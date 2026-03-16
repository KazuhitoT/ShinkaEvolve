import heapq
import json
import math
import unicodedata
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import numpy as np

_tokenizer = None

def _get_tokenizer():
    """Lazily build lindera tokenizer (only when actually called)."""
    global _tokenizer
    if _tokenizer is None:
        from lindera import TokenizerBuilder
        builder = TokenizerBuilder()
        builder.set_mode("normal")
        builder.set_dictionary("lindera-ipadic-2.7.0-20250920")
        _tokenizer = builder.build()
    return _tokenizer

def prep(text: str) -> str:
    """単語をスペース区切りにして出力"""
    tok = _get_tokenizer()
    tokens = tok.tokenize(text)
    return " ".join(token.surface for token in tokens)

def preprocess_text(text: str) -> str:
    """Normalize text for comparison."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = " ".join(text.split())
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into character bigrams."""
    tokens = []
    for i in range(len(text) - 1):
        bigram = text[i : i + 2]
        if bigram.strip():
            tokens.append(bigram)
    return tokens


def build_index(
    corpus: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Build an inverted index from the corpus."""
    inverted: Dict[str, List[int]] = {}
    doc_lengths: List[int] = []
    doc_token_sets: List[set] = []

    for idx, doc in tqdm(enumerate(corpus)):
        text = preprocess_text(doc.get("title", "") + " " + doc["text"])
        tokens = tokenize(text)
        doc_lengths.append(len(tokens))
        token_set = set(tokens)
        doc_token_sets.append(token_set)

        for t in token_set:
            if t not in inverted:
                inverted[t] = []
            inverted[t].append(idx)

    num_docs = len(corpus)
    avg_dl = sum(doc_lengths) / num_docs if num_docs else 1.0

    # Pre-compute IDF values and filter out near-zero IDF terms
    idf_map: Dict[str, float] = {}
    for t, posting in inverted.items():
        df = len(posting)
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        if idf > 0.01:  # Skip near-zero IDF terms (very common bigrams)
            idf_map[t] = idf

    return {
        "inverted": inverted,
        "doc_lengths": doc_lengths,
        "avg_dl": avg_dl,
        "num_docs": num_docs,
        "doc_token_sets": doc_token_sets,
        "idf_map": idf_map,
    }


def retrieve(
    query: str,
    index: Dict[str, Any],
    top_k: int = 100,
) -> List[Tuple[int, float]]:
    """Retrieve top-k candidate document indices from the index."""
    q_text = preprocess_text(query)
    q_tokens = tokenize(q_text)
    q_tf: Dict[str, int] = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    inverted = index["inverted"]
    idf_map = index.get("idf_map")

    scores: Dict[int, float] = {}
    if idf_map is not None:
        # Fast path: use pre-computed IDF, skip huge posting lists (low discriminative power)
        for token, qtf in q_tf.items():
            idf = idf_map.get(token)
            if idf is None:
                continue
            posting = inverted[token]
            if len(posting) > 50000:
                continue
            idf_qtf = idf * qtf
            for doc_idx in posting:
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf_qtf
    else:
        # Fallback for indexes without pre-computed IDF
        num_docs = index["num_docs"]
        for token, qtf in q_tf.items():
            posting = inverted.get(token)
            if posting is None:
                continue
            df = len(posting)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_idx in posting:
                scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * qtf

    return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


def score_passage(
    query: str,
    title: str,
    text: str,
) -> float:
    """Score a single passage for reranking."""
    q_processed = preprocess_text(query)
    p_processed = preprocess_text(title + " " + text)

    q_tokens = tokenize(q_processed)
    p_tokens = tokenize(p_processed)

    if not q_tokens or not p_tokens:
        return 0.0

    q_set = set(q_tokens)
    p_set = set(p_tokens)
    overlap = len(q_set & p_set)
    if overlap == 0:
        return 0.0

    # Jaccard-like score weighted by IDF proxy (rarer bigrams count more)
    return overlap / (len(q_set) + len(p_set) - overlap)


def batch_dense_score(
    candidate_indices: list,
    embeddings: Dict[str, Any],
    query_index: int,
) -> Dict[int, float]:
    """Vectorized dense scoring across all embedding models."""
    idx_arr = np.array(candidate_indices, dtype=np.int64)
    model_scores = []
    for emb_data in embeddings.values():
        q_emb = emb_data["queries"][query_index]
        d_embs = emb_data["corpus"][idx_arr]
        model_scores.append(d_embs @ q_emb)
    avg = np.mean(model_scores, axis=0)
    return {candidate_indices[i]: float(avg[i]) for i in range(len(candidate_indices))}


# EVOLVE-BLOCK-START

def _normalize_scores(scores_dict: Dict[int, float]) -> Dict[int, float]:
    """Normalize score values to [0, 1] range via min-max scaling."""
    if not scores_dict:
        return scores_dict
    vals = list(scores_dict.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.5 for k in scores_dict}
    return {k: (v - lo) / (hi - lo) for k, v in scores_dict.items()}


def search(
    query: str,
    corpus: List[Dict[str, str]],
    index: Dict[str, Any],
    top_k: int = 10,
    embeddings: Optional[Dict[str, Any]] = None,
    query_index: Optional[int] = None,
    rerankers: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    candidates = retrieve(query, index, top_k=top_k * 10)

    # Build reranker score lookups for this query
    reranker_lookups = {}
    if rerankers is not None and query_index is not None:
        for model_name, data in rerankers.items():
            idx = data["doc_indices"][query_index]
            scr = data["scores"][query_index]
            mask = idx >= 0
            reranker_lookups[model_name] = dict(
                zip(idx[mask].tolist(), scr[mask].tolist())
            )

    # --- Phase 1: Signal collection ---
    sparse_scores: Dict[int, float] = {}
    dense_scores: Dict[int, float] = {}
    reranker_scores: Dict[int, float] = {}

    # Precompute query token set for Jaccard scoring
    q_text = preprocess_text(query)
    q_set = set(tokenize(q_text))
    doc_token_sets = index.get("doc_token_sets")

    for doc_idx, _retrieval_score in candidates:
        # Sparse signal via precomputed doc_token_sets (avoids score_passage overhead)
        if doc_token_sets is not None and q_set:
            p_set = doc_token_sets[doc_idx]
            overlap = len(q_set & p_set)
            sparse_scores[doc_idx] = overlap / (len(q_set) + len(p_set) - overlap) if overlap > 0 else 0.0
        else:
            doc = corpus[doc_idx]
            sparse_scores[doc_idx] = score_passage(query, doc.get("title", ""), doc["text"])

        # Reranker signal: average across all reranker models
        if reranker_lookups:
            scores_list = [lk.get(doc_idx, 0.0) for lk in reranker_lookups.values()]
            reranker_scores[doc_idx] = sum(scores_list) / len(scores_list)

    # Batch dense scoring (vectorized across all candidates)
    if embeddings is not None and query_index is not None:
        candidate_indices = [doc_idx for doc_idx, _ in candidates]
        dense_scores = batch_dense_score(candidate_indices, embeddings, query_index)

    # --- Phase 2: Normalize each signal to [0, 1] ---
    sparse_norm = _normalize_scores(sparse_scores)
    dense_norm = _normalize_scores(dense_scores)
    reranker_norm = _normalize_scores(reranker_scores)

    w_sparse, w_dense, w_rerank = 0.3, 0.3, 0.4

    results = []
    for doc_idx, _retrieval_score in candidates:
        doc = corpus[doc_idx]
        s_sparse = sparse_norm.get(doc_idx, 0.0)
        s_dense = dense_norm.get(doc_idx, 0.0)
        s_rerank = reranker_norm.get(doc_idx, 0.0)

        if reranker_lookups and embeddings is not None:
            combined = w_sparse * s_sparse + w_dense * s_dense + w_rerank * s_rerank
        elif embeddings is not None:
            combined = 0.4 * s_sparse + 0.6 * s_dense
        else:
            combined = s_sparse

        results.append({"passage_id": doc["docid"], "score": combined})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# EVOLVE-BLOCK-END


def load_corpus(corpus_path: str) -> List[Dict[str, str]]:
    """Load corpus from JSONL file."""
    corpus = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                corpus.append(json.loads(line))
    return corpus


def run_search_batch(
    queries: List[Dict[str, str]],
    corpus_path: str = None,
    corpus: List[Dict[str, str]] = None,
    embedding_registry=None,
    reranker_registry=None,
    prebuilt_index=None,
) -> List[List[Dict[str, Any]]]:
    """Entry point: load corpus, build index, search all queries.

    Args:
        queries: List of {"query_id", "query"} dicts.
        corpus_path: Path to corpus.jsonl (used when corpus is not provided).
        corpus: Pre-loaded corpus list (used by corpus_server to avoid re-loading).
        embedding_registry: EmbeddingRegistry instance (optional).
        reranker_registry: RerankerRegistry instance (optional).
        prebuilt_index: Pre-built inverted index (used by corpus_server to avoid re-building).

    Returns:
        List of search results per query.
    """
    if corpus is None:
        corpus = load_corpus(corpus_path)

    if prebuilt_index is not None:
        index = prebuilt_index
    else:
        index = build_index(corpus)

    # Prepare embeddings dict if registry is available
    embeddings = None
    if embedding_registry is not None:
        embeddings = {}
        for name in embedding_registry.available_models():
            store = embedding_registry.get_store(name)
            embeddings[name] = {
                "corpus": store.corpus,
                "queries": store.queries,
                "dimension": store.metadata["dimension"],
            }

    # Prepare rerankers dict if registry is available
    rerankers = None
    if reranker_registry is not None:
        rerankers = {}
        for name in reranker_registry.available_models():
            store = reranker_registry.get_store(name)
            rerankers[name] = {
                "doc_indices": store.doc_indices,
                "scores": store.scores,
            }

    all_results = []
    for qi, q in enumerate(queries):
        result = search(
            q["query"], corpus, index, top_k=10,
            embeddings=embeddings, query_index=qi,
            rerankers=rerankers,
        )
        all_results.append(result)
    return all_results
