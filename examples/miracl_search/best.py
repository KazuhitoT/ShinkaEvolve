"""MIRACL Japanese information retrieval (日本語情報検索 — recall + rerank)."""

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

def _normalize_scores(scores_dict: Dict[int, float], method: str = "hybrid", sharpen: float = 1.4, rank_weight: float = 0.6) -> Dict[int, float]:
    """Normalized scores with flexible fusion strategies.

    Supports:
      - method="hybrid"  : percentile + sharpened min-max (default)
      - method="percentile": pure percentile ranking
      - method="minmax"    : pure min-max with sharpening
    sharpening is applied to the min-max path and to the final combined distribution
    to emphasize top signals.
    """
    if not scores_dict:
        return {}
    keys = list(scores_dict.keys())
    vals = np.array([scores_dict[k] for k in keys], dtype=np.float32)

    if len(vals) == 0:
        return {}
    if len(vals) == 1:
        return {keys[0]: 1.0}

    if method == "percentile":
        sorted_indices = np.argsort(vals)
        ranks = np.zeros_like(vals)
        ranks[sorted_indices] = np.linspace(0, 1, len(vals))
        return {keys[i]: float(ranks[i]) for i in range(len(keys))}

    elif method == "minmax":
        v_min, v_max = np.min(vals), np.max(vals)
        if v_max - v_min < 1e-9:
            return {k: 0.5 for k in keys}
        v_norm = (vals - v_min) / (v_max - v_min)
        if sharpen != 1.0:
            v_norm = v_norm ** sharpen
        return {keys[i]: float(v_norm[i]) for i in range(len(keys))}

    else:  # hybrid
        sorted_indices = np.argsort(vals)
        ranks = np.zeros_like(vals)
        ranks[sorted_indices] = np.linspace(0, 1, len(vals))

        v_min, v_max = np.min(vals), np.max(vals)
        if v_max - v_min > 1e-9:
            v_norm = (vals - v_min) / (v_max - v_min)
            v_norm = v_norm ** sharpen
        else:
            v_norm = np.zeros_like(vals)

        combined = rank_weight * ranks + (1.0 - rank_weight) * v_norm
        # Normalize to [0,1]
        c_min, c_max = combined.min(), combined.max()
        if c_max - c_min > 1e-9:
            combined = (combined - c_min) / (c_max - c_min)
        return {keys[i]: float(combined[i]) for i in range(len(keys))}

def _compute_signal_confidence(scores_dict: Dict[int, float]) -> float:
    """Estimate signal discriminativeness (variance-based)."""
    if not scores_dict or len(scores_dict) <= 1:
        return 0.0
    vals = np.array(list(scores_dict.values()), dtype=np.float32)
    variance = float(np.var(vals))
    nonzero_ratio = float(np.sum(vals > 1e-6) / len(vals))
    confidence = (variance * 0.75 + nonzero_ratio * 0.25)
    return min(confidence, 1.0)

def _model_weight(name: str) -> float:
    """Heuristic per-model base weight based on known model types."""
    if "text-embedding-3" in name:
        return 3.2
    if "gemini" in name:
        return 2.0
    if "ruri" in name:
        return 1.6
    if "static" in name or "cl" in name:
        return 0.4
    return 1.4

def _calibrate_sims_adaptive(sims: np.ndarray, variance: float) -> np.ndarray:
    """Calibrate cosine sims to [0,1] with adaptive sharpening based on variance.

    Higher variance models get more sharpening to emphasize their discriminative power.
    """
    v_min, v_max = sims.min(), sims.max()
    if v_max - v_min > 1e-9:
        norm = (sims - v_min) / (v_max - v_min)
        # Adaptive sharpening: high variance -> more sharpening (1.3-1.6)
        sharpness = 1.3 + min(variance * 0.3, 0.3)
        norm = norm ** sharpness
    else:
        norm = np.zeros_like(sims)
    return norm

def _compute_soft_negative_penalty(s_val: float, d_val: float, r_val: float,
                                    s_conf: float, d_conf: float, r_conf: float,
                                    q_len: int) -> float:
    """Compute penalty for soft negatives (high sparse/dense but low reranker).

    Soft negatives are likely false positives that should be penalized.
    The penalty is adaptive based on query length and signal confidence.
    """
    if r_conf < 0.01:
        return 0.0  # No reranker signal, can't identify soft negatives

    # Detect soft negatives: high sparse/dense but low reranker
    sparse_dense_avg = (s_val + d_val) / 2.0 if d_val > 0 else s_val
    disagreement = max(0.0, sparse_dense_avg - r_val)

    # Query-adaptive threshold: longer queries are stricter
    if q_len <= 8:
        threshold = 0.35
    elif q_len <= 20:
        threshold = 0.30
    else:
        threshold = 0.25

    if disagreement < threshold:
        return 0.0

    # Scale penalty by confidence in reranker signal and disagreement magnitude
    penalty_weight = (disagreement - threshold) * r_conf * 0.18
    return penalty_weight

def search(
    query: str,
    corpus: List[Dict[str, str]],
    index: Dict[str, Any],
    top_k: int = 10,
    embeddings: Optional[Dict[str, Any]] = None,
    query_index: Optional[int] = None,
    rerankers: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Full search pipeline with adaptive calibration and soft negative mining."""

    # 1. Candidate retrieval with query-adaptive pool size
    q_len = len(query)
    if q_len <= 8:
        pool_size = min(top_k * 40, 400)
    elif q_len <= 20:
        pool_size = min(top_k * 45, 450)
    else:
        pool_size = min(top_k * 50, 600)

    candidates = retrieve(query, index, top_k=pool_size)
    if not candidates:
        return [{"passage_id": corpus[i]["docid"], "score": 0.0} for i in range(min(top_k, len(corpus)))]
    candidate_indices = [c[0] for c in candidates]
    sparse_scores_raw = {idx: s for idx, s in candidates}

    # 2. Reranker lookups (calibrated)
    reranker_lookups: Dict[str, Dict[int, float]] = {}
    if rerankers is not None and query_index is not None:
        for mname, data in rerankers.items():
            idx_arr = data["doc_indices"][query_index]
            scr_arr = data["scores"][query_index]
            mask = idx_arr >= 0
            if mask.any():
                reranker_lookups[mname] = dict(zip(idx_arr[mask].tolist(), scr_arr[mask].tolist()))

    # 3. Dense scores from embeddings (vectorized with adaptive calibration)
    dense_scores: Dict[int, float] = {}
    if embeddings is not None and query_index is not None and len(candidate_indices) > 0:
        idx_arr = np.array(candidate_indices, dtype=np.int64)
        dense_sum = np.zeros(len(candidate_indices), dtype=np.float64)
        total_model_weight = 0.0

        for name, data in embeddings.items():
            sims = data["corpus"][idx_arr] @ data["queries"][query_index]  # (N,)
            sims = (sims + 1.0) / 2.0  # map to [0,1]

            # Compute variance for adaptive calibration
            var = float(np.var(sims))
            sims_cal = _calibrate_sims_adaptive(sims, var)

            w_base = _model_weight(name)
            dense_sum += sims_cal * w_base
            total_model_weight += w_base

        if total_model_weight > 0:
            dense_sum = dense_sum / total_model_weight
        dense_scores = {candidate_indices[i]: float(dense_sum[i]) for i in range(len(candidate_indices))}

    # 4. Reranker scores (weighted ensemble with per-model calibration)
    reranker_scores: Dict[int, float] = {}
    if reranker_lookups:
        per_model_calibrated = {}
        for name, lookup in reranker_lookups.items():
            vals = np.array(list(lookup.values()), dtype=np.float32)
            if len(vals) > 0:
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin > 1e-9:
                    calib = (vals - vmin) / (vmax - vmin)
                    calib = calib ** 1.6  # Increased sharpening for reranker
                else:
                    calib = np.ones_like(vals)
                per_model_calibrated[name] = dict(zip(lookup.keys(), calib.tolist()))
            else:
                per_model_calibrated[name] = lookup

        for idx in candidate_indices:
            r_sum, r_w = 0.0, 0.0
            for name, calib_lookup in per_model_calibrated.items():
                w = 6.0 if "ruri" in name else 1.0
                val = calib_lookup.get(idx, 0.0)
                r_sum += val * w
                r_w += w
            if r_w > 0:
                reranker_scores[idx] = r_sum / r_w

    # 5. Normalize each signal with adaptive sharpening and method selection
    # Sparse: Hybrid is good for robustness.
    s_norm = _normalize_scores(sparse_scores_raw, method="hybrid", sharpen=1.0, rank_weight=0.85 if q_len > 20 else 0.75)
    # Dense/Reranker: MinMax with sharpening is better for precision.
    d_norm = _normalize_scores(dense_scores, method="minmax", sharpen=1.8) if dense_scores else {}
    r_norm = _normalize_scores(reranker_scores, method="minmax", sharpen=2.2) if reranker_scores else {}

    # 6. Signal confidences and Clarity
    s_conf = _compute_signal_confidence(sparse_scores_raw)
    d_conf = _compute_signal_confidence(dense_scores)
    r_conf = _compute_signal_confidence(reranker_scores)

    # 7. Query-adaptive fusion weights with Clarity-based adjustment
    if q_len <= 6:
        w_s, w_d, w_r = 0.03, 0.22, 0.75
    elif q_len <= 15:
        w_s, w_d, w_r = 0.05, 0.30, 0.65
    elif q_len <= 30:
        w_s, w_d, w_r = 0.10, 0.35, 0.55
    else:
        w_s, w_d, w_r = 0.15, 0.35, 0.50

    # Boost the weight of the most "clear" signal (highest max-mean gap)
    if s_norm and d_norm and r_norm:
        s_vals = list(s_norm.values())
        d_vals = list(d_norm.values())
        r_vals = list(r_norm.values())
        s_clarity = max(s_vals) - np.mean(s_vals)
        d_clarity = max(d_vals) - np.mean(d_vals)
        r_clarity = max(r_vals) - np.mean(r_vals)
        c_sum = s_clarity + d_clarity + r_clarity
        if c_sum > 1e-6:
            w_s *= (1.0 + 0.2 * s_clarity / c_sum)
            w_d *= (1.0 + 0.2 * d_clarity / c_sum)
            w_r *= (1.0 + 0.2 * r_clarity / c_sum)
            _tw = w_s + w_d + w_r
            w_s, w_d, w_r = w_s/_tw, w_d/_tw, w_r/_tw

    if not d_norm and not r_norm:
        w_s, w_d, w_r = 1.0, 0.0, 0.0
    elif not r_norm:
        w_s, w_d, w_r = 0.20, 0.80, 0.0
    elif not d_norm:
        w_s, w_d, w_r = 0.20, 0.0, 0.80

    # 8. Final non-linear fusion with enhanced consensus, soft negative mining, and title boost
    q_processed = preprocess_text(query)
    q_tokens = set(tokenize(q_processed))

    results = []
    for pos, idx in enumerate(candidate_indices):
        s_val = s_norm.get(idx, 0.0)
        d_val = d_norm.get(idx, 0.0)
        r_val = r_norm.get(idx, 0.0)

        base = w_s * s_val + w_d * d_val + w_r * r_val

        # Enhanced consensus bonuses (using geometric mean for stability)
        consensus = 0.0
        # Dense-Reranker agreement (most reliable pair)
        if d_norm and r_norm and d_conf > 0.01 and r_conf > 0.01:
            consensus += 0.40 * np.sqrt(d_val * r_val)
        # Sparse-Reranker agreement
        if s_norm and r_norm and s_conf > 0.005:
            consensus += 0.12 * np.sqrt(s_val * r_val)
        # Sparse-Dense agreement (secondary signal)
        if s_norm and d_norm and s_conf > 0.005 and d_conf > 0.01:
            consensus += 0.10 * np.sqrt(s_val * d_val)

        # Soft negative mining: penalize documents with high sparse/dense but low reranker
        soft_neg_penalty = _compute_soft_negative_penalty(s_val, d_val, r_val,
                                                           s_conf, d_conf, r_conf, q_len)

        # Title Match Boost: Precision-oriented keyword matching
        title_boost = 0.0
        if q_tokens:
            title_raw = corpus[idx].get("title", "")
            title_text = preprocess_text(title_raw)
            title_tokens = set(tokenize(title_text))
            overlap = len(q_tokens & title_tokens)
            if overlap > 0:
                ratio = overlap / len(q_tokens)
                # Exact title match or head match bonus
                exact_bonus = 0.10 if q_processed == title_text or q_processed in title_text else 0.0
                # Boost is gated by reranker confidence to avoid false positives
                title_boost = (0.18 * ratio + exact_bonus) * (r_val**1.2 + 0.1)

        # MRR-like tie-breaker using initial BM25 position
        mrr_bonus = 0.005 / (pos + 1)

        final_score = base + consensus - soft_neg_penalty + title_boost + mrr_bonus
        results.append({"passage_id": corpus[idx]["docid"], "score": float(final_score)})

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
