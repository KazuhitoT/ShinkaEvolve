"""
Evaluator for MIRACL Japanese information retrieval (日本語情報検索 — recall + rerank) evolution.
"""

import json
import math
import os
import pickle
import socket
import struct
import time
import argparse
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval
from shinka.core.wrap_eval import save_json_results


def _default_queries_file() -> str:
    """Return default eval queries path relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "data", "queries_eval.jsonl")


def _default_corpus_file() -> str:
    """Return default corpus path relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "data", "corpus.jsonl")


def _default_embeddings_dir() -> Optional[str]:
    """Return default embeddings dir if it exists, else None."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emb_dir = os.path.join(script_dir, "data", "embeddings")
    if os.path.isdir(emb_dir) and os.path.exists(os.path.join(emb_dir, "manifest.json")):
        return emb_dir
    return None


def _default_rerankers_dir() -> Optional[str]:
    """Return default rerankers dir if it exists, else None."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rr_dir = os.path.join(script_dir, "data", "rerankers")
    if os.path.isdir(rr_dir) and os.path.exists(os.path.join(rr_dir, "manifest.json")):
        return rr_dir
    return None


# ---------------------------------------------------------------------------
# Corpus-server communication helpers
# ---------------------------------------------------------------------------

CORPUS_SERVER_SOCKET = "/tmp/miracl_corpus_server.sock"


def _server_available() -> bool:
    """Return True if the corpus server socket exists."""
    return os.path.exists(CORPUS_SERVER_SOCKET)


def _send_msg(sock, data):
    raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed before all bytes received")
        buf += chunk
    return buf


def _recv_msg(sock):
    header = _recv_exact(sock, 4)
    if not header:
        return None
    size = struct.unpack("!I", header)[0]
    return pickle.loads(_recv_exact(sock, size))


def _call_corpus_server(program_path: str, queries: list) -> list:
    """Send an evaluation request to the corpus server and return results."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5)  # connect timeout: short
    sock.connect(CORPUS_SERVER_SOCKET)
    sock.settimeout(120)  # send/recv timeout: well within scheduler's 300s limit
    try:
        _send_msg(sock, {
            "program_path": os.path.abspath(program_path),
            "queries": queries,
        })
        response = _recv_msg(sock)
    finally:
        sock.close()
    if response is None:
        raise RuntimeError("Corpus server returned empty response")
    if not response["success"]:
        raise RuntimeError(f"Corpus server error:\n{response['error']}")
    return response["results"]


# ---------------------------------------------------------------------------

def load_queries(queries_file: str) -> List[Dict[str, Any]]:
    """Load evaluation queries from a JSONL file.

    Each line: {"query_id", "query", "annotations": [{"docid", "relevance"}]}
    """
    queries = []
    with open(queries_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            queries.append(json.loads(line))
    return queries


def ndcg_at_k(predicted_order: List[str], relevant_set: set, k: int = 10) -> float:
    """Compute NDCG@k with binary relevance."""
    dcg = 0.0
    for i, pid in enumerate(predicted_order[:k]):
        if pid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    # IDCG: 関連文書が全て上位に並んだ場合の DCG
    num_rel = min(len(relevant_set), k)
    if num_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(num_rel))
    return dcg / idcg


def precision_at_k(predicted_order: List[str], relevant_set: set, k: int = 10) -> float:
    """Compute Precision@k: fraction of top-k results that are relevant."""
    if not predicted_order:
        return 0.0
    hits = sum(1 for pid in predicted_order[:k] if pid in relevant_set)
    return hits / min(k, len(predicted_order))


def reciprocal_rank(predicted_order: List[str], relevant_set: set, k: int = 10) -> float:
    """Compute MRR@k: reciprocal of the rank of the first relevant result."""
    for i, pid in enumerate(predicted_order[:k]):
        if pid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def main(
    program_path: str,
    results_dir: str,
    queries_file: Optional[str] = None,
    corpus_path: Optional[str] = None,
    embeddings_dir: Optional[str] = None,
    rerankers_dir: Optional[str] = None,
):
    """Runs the MIRACL retrieval + reranking evaluation using shinka.eval."""
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    resolved_corpus = corpus_path or _default_corpus_file()
    eval_queries = load_queries(queries_file or _default_queries_file())
    print(f"Loaded {len(eval_queries)} evaluation queries")
    print(f"Using corpus: {resolved_corpus}")

    # Load embedding registry for direct evaluation path
    resolved_embeddings_dir = embeddings_dir or _default_embeddings_dir()
    embedding_registry = None
    if resolved_embeddings_dir:
        from embedding_loader import load_registry

        embedding_registry = load_registry(resolved_embeddings_dir)
        if embedding_registry is not None:
            models = embedding_registry.available_models()
            print(f"Embeddings: {len(models)} models — {models}")
        else:
            print("No valid embeddings found")

    # Load reranker registry for direct evaluation path
    resolved_rerankers_dir = rerankers_dir or _default_rerankers_dir()
    reranker_registry = None
    if resolved_rerankers_dir:
        from reranker_loader import load_reranker_registry

        reranker_registry = load_reranker_registry(resolved_rerankers_dir)
        if reranker_registry is not None:
            models = reranker_registry.available_models()
            print(f"Rerankers: {len(models)} models — {models}")
        else:
            print("No valid rerankers found")

    def get_search_kwargs(run_index: int) -> Dict[str, Any]:
        """Provides queries, corpus path, embedding and reranker registries."""
        batch = [
            {"query_id": q["query_id"], "query": q["query"]}
            for q in eval_queries
        ]
        kwargs = {"queries": batch, "corpus_path": resolved_corpus}
        if embedding_registry is not None:
            kwargs["embedding_registry"] = embedding_registry
        if reranker_registry is not None:
            kwargs["reranker_registry"] = reranker_registry
        return kwargs

    def validate_search(
        run_output: List[List[Dict[str, Any]]],
    ) -> Tuple[bool, Optional[str]]:
        """Validates search results."""
        if not isinstance(run_output, list):
            return False, f"Output is not a list, got {type(run_output)}"

        expected_len = len(eval_queries)
        if len(run_output) != expected_len:
            return False, (
                f"Expected {expected_len} query results, got {len(run_output)}"
            )

        for qi, query_result in enumerate(run_output):
            if not isinstance(query_result, list):
                return False, f"Query {qi} result is not a list"

            for pi, item in enumerate(query_result):
                if not isinstance(item, dict):
                    return False, f"Query {qi}, item {pi} is not a dict"
                if "passage_id" not in item:
                    return False, f"Query {qi}, item {pi} missing 'passage_id'"
                if "score" not in item:
                    return False, f"Query {qi}, item {pi} missing 'score'"
                score = item["score"]
                if not isinstance(score, (int, float)):
                    return False, (
                        f"Query {qi}, item {pi} score is not numeric: "
                        f"{type(score)}"
                    )
                if math.isnan(score) or math.isinf(score):
                    return False, f"Query {qi}, item {pi} score is {score}"

        return True, "Valid search results."

    def _aggregator(
        results: List[List[List[Dict[str, Any]]]],
        elapsed: float = 0.0,
    ) -> Dict[str, Any]:
        """Aggregates retrieval metrics (multi-objective).

        combined_score = mean_ndcg_at_10
        Public metrics are independent Pareto axes:
          mean_ndcg_at_10, mean_precision_at_10, mean_mrr_at_10, time_seconds
        """
        if not results:
            return {"combined_score": 0.0, "error": "No results to aggregate"}

        query_results = results[0]  # num_runs=1
        ndcg_scores = []
        prec_scores = []
        mrr_scores = []
        per_query_details = []

        # Build relevant sets
        relevant_sets = []
        for qi, q_data in enumerate(eval_queries):
            relevant_sets.append({
                a["docid"]
                for a in q_data["annotations"]
                if a["relevance"] > 0
            })

        for qi, query_result in enumerate(query_results):
            q_data = eval_queries[qi]
            relevant_set = relevant_sets[qi]

            predicted_order = [item["passage_id"] for item in query_result]

            score_ndcg = ndcg_at_k(predicted_order, relevant_set, k=10)
            score_prec = precision_at_k(predicted_order, relevant_set, k=10)
            score_mrr = reciprocal_rank(predicted_order, relevant_set, k=10)

            ndcg_scores.append(score_ndcg)
            prec_scores.append(score_prec)
            mrr_scores.append(score_mrr)

            per_query_details.append({
                "query_id": q_data.get("query_id", f"q{qi}"),
                "ndcg_at_10": round(score_ndcg, 4),
                "precision_at_10": round(score_prec, 4),
                "mrr_at_10": round(score_mrr, 4),
                "num_relevant": len(relevant_set),
                "num_retrieved": len(query_result),
            })

        n = len(ndcg_scores) if ndcg_scores else 1
        mean_ndcg = sum(ndcg_scores) / n
        mean_prec = sum(prec_scores) / n
        mean_mrr = sum(mrr_scores) / n

        return {
            "combined_score": round(mean_ndcg, 6),
            "public": {
                "mean_ndcg_at_10": round(mean_ndcg, 4),
                "mean_precision_at_10": round(mean_prec, 4),
                "mean_mrr_at_10": round(mean_mrr, 4),
                "time_seconds": round(elapsed, 2),
            },
            "private": {
                "per_query": per_query_details,
                "all_ndcg_scores": [round(s, 4) for s in ndcg_scores],
            },
        }

    server_success = False
    if _server_available():
        # ---------- server path (corpus shared via COW fork) ----------
        print("Corpus server detected, delegating evaluation ...")
        queries_batch = [
            {"query_id": q["query_id"], "query": q["query"]}
            for q in eval_queries
        ]
        try:
            start_time = time.perf_counter()
            run_output = _call_corpus_server(
                os.path.abspath(program_path), queries_batch,
            )
            elapsed = time.perf_counter() - start_time

            valid, err = validate_search(run_output)
            if valid:
                metrics = _aggregator([run_output], elapsed=elapsed)
                metrics["execution_time_mean"] = elapsed
                metrics["execution_time_std"] = 0.0
                metrics["num_valid_runs"] = 1
                metrics["num_invalid_runs"] = 0
                metrics["all_validation_errors"] = []
                correct = True
                error_msg = None
            else:
                metrics = {
                    "combined_score": 0.0,
                    "public": {
                        "mean_ndcg_at_10": 0.0,
                        "mean_precision_at_10": 0.0,
                        "mean_mrr_at_10": 0.0,
                        "time_seconds": round(elapsed, 2),
                    },
                    "execution_time_mean": elapsed,
                    "execution_time_std": 0.0,
                    "num_valid_runs": 0,
                    "num_invalid_runs": 1,
                    "all_validation_errors": [err],
                }
                correct = False
                error_msg = f"Validation failed: {err}"
            server_success = True
        except (ConnectionRefusedError, ConnectionError, OSError) as e:
            print(f"Corpus server connection failed ({e}), falling back to direct evaluation ...")
        except Exception as e:
            metrics = {
                "combined_score": 0.0,
                "public": {
                    "mean_ndcg_at_10": 0.0,
                    "mean_precision_at_10": 0.0,
                    "mean_mrr_at_10": 0.0,
                    "time_seconds": 0.0,
                },
                "execution_time_mean": 0.0,
                "execution_time_std": 0.0,
                "num_successful_runs": 0,
                "num_valid_runs": 0,
                "num_invalid_runs": 0,
                "all_validation_errors": [str(e)],
            }
            correct = False
            error_msg = str(e)
            server_success = True  # エラーとして処理済み、フォールバック不要

        if server_success:
            save_json_results(results_dir, metrics, correct, error_msg)

    if not server_success:
        # ---------- fallback: direct evaluation (no server) ----------
        _fallback_start = time.perf_counter()

        def _aggregator_with_time(results):
            return _aggregator(results, elapsed=time.perf_counter() - _fallback_start)

        metrics, correct, error_msg = run_shinka_eval(
            program_path=program_path,
            results_dir=results_dir,
            experiment_fn_name="run_search_batch",
            num_runs=1,
            get_experiment_kwargs=get_search_kwargs,
            validate_fn=validate_search,
            aggregate_metrics_fn=_aggregator_with_time,
        )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k2, v2 in value.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIRACL Japanese IR (日本語情報検索) evaluator"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to program to evaluate (must contain 'run_search_batch')",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save results (metrics.json, correct.json)",
    )
    parser.add_argument(
        "--queries_file",
        type=str,
        default=None,
        help="Path to JSONL queries file (default: data/queries_eval.jsonl)",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default=None,
        help="Path to corpus JSONL file (default: data/corpus.jsonl)",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Path to precomputed embeddings directory (default: data/embeddings/)",
    )
    parser.add_argument(
        "--rerankers_dir",
        type=str,
        default=None,
        help="Path to precomputed rerankers directory (default: data/rerankers/)",
    )
    parsed_args = parser.parse_args()
    main(
        parsed_args.program_path,
        parsed_args.results_dir,
        parsed_args.queries_file,
        parsed_args.corpus_path,
        parsed_args.embeddings_dir,
        parsed_args.rerankers_dir,
    )
