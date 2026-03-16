#!/usr/bin/env python3
"""Download MIRACL hard negatives and write NEW docs to a separate corpus file.

This downloads top-250 retrieval candidates (BM25 + e5-multilingual-large
+ e5-mistral-instruct) that are NOT relevant, and writes only the documents
not already in the existing corpus to a separate file.

Usage:
    python prepare_hard_negatives.py [--data_dir data]

After running:
    1. Compute embeddings for new docs only:
       python precompute_embeddings.py --corpus_file data/corpus_hard_negatives.jsonl \
           --output_dir data/embeddings_hn --skip_queries
       python precompute_embeddings_api.py --corpus_file data/corpus_hard_negatives.jsonl \
           --output_dir data/embeddings_hn --skip_queries
    2. Merge everything:
       python merge_corpora.py
    3. Re-run reranker precomputation (corpus indices changed):
       python precompute_rerankers.py
    4. Evaluate
"""

import argparse
import json
import os


def load_existing_corpus_ids(corpus_path):
    """Load set of docids from existing corpus."""
    docids = set()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                doc = json.loads(line)
                docids.add(doc["docid"])
    return docids


def load_existing_queries(queries_path):
    """Load existing queries as dict keyed by query_id."""
    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                queries[q["query_id"]] = q
    return queries


def download_hard_negatives():
    """Download the MIRACL hard negatives dataset for Japanese."""
    from datasets import load_dataset

    ds_name = "mteb/MIRACLRetrieval_ja_top_250_only_w_correct-v2"

    print("Downloading hard negatives corpus ...")
    hn_corpus = load_dataset(ds_name, "corpus", split="test")
    print(f"  Hard negatives corpus: {len(hn_corpus)} docs")

    print("Downloading hard negatives queries ...")
    hn_queries = load_dataset(ds_name, "queries", split="test")
    print(f"  Hard negatives queries: {len(hn_queries)} queries")

    print("Downloading hard negatives qrels ...")
    hn_qrels = load_dataset(ds_name, "default", split="test")
    print(f"  Hard negatives qrels: {len(hn_qrels)} entries")

    return hn_corpus, hn_queries, hn_qrels


def main():
    parser = argparse.ArgumentParser(
        description="Download MIRACL hard negatives to a separate corpus file"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Data directory (default: <script_dir>/data)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "data")
    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    queries_path = os.path.join(data_dir, "queries_eval.jsonl")
    hn_corpus_path = os.path.join(data_dir, "corpus_hard_negatives.jsonl")
    hn_queries_path = os.path.join(data_dir, "queries_eval_hn.jsonl")

    # Load existing docids
    print(f"Loading existing corpus docids from {corpus_path} ...")
    existing_docids = load_existing_corpus_ids(corpus_path)
    print(f"  Existing corpus: {len(existing_docids)} docs")

    # Load existing queries
    print(f"Loading existing queries from {queries_path} ...")
    existing_queries = load_existing_queries(queries_path)
    print(f"  Existing queries: {len(existing_queries)} queries")

    # Download hard negatives
    hn_corpus, hn_queries, hn_qrels = download_hard_negatives()

    # Analyze overlap
    hn_docids = {row["_id"] for row in hn_corpus}
    overlap = hn_docids & existing_docids
    new_docs = hn_docids - existing_docids
    print(f"\n--- Overlap analysis ---")
    print(f"  Existing corpus:     {len(existing_docids):>10}")
    print(f"  Hard negatives:      {len(hn_docids):>10}")
    print(f"  Overlap:             {len(overlap):>10}")
    print(f"  New docs to add:     {len(new_docs):>10}")
    print(f"  Merged total:        {len(existing_docids | hn_docids):>10}")

    # Analyze qrels
    positive_qrels = sum(1 for row in hn_qrels if row["score"] == 1)
    negative_qrels = sum(1 for row in hn_qrels if row["score"] == 0)
    print(f"\n--- Qrels analysis ---")
    print(f"  Total entries:       {len(hn_qrels):>10}")
    print(f"  Positive (score=1):  {positive_qrels:>10}")
    print(f"  Negative (score=0):  {negative_qrels:>10}")

    # Write NEW docs only (not in existing corpus) to separate file
    print(f"\n--- Writing hard negative docs ---")
    added = 0
    with open(hn_corpus_path, "w", encoding="utf-8") as f:
        for row in hn_corpus:
            docid = row["_id"]
            if docid not in existing_docids:
                doc = {
                    "docid": docid,
                    "title": row["title"],
                    "text": row["text"],
                }
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                added += 1
    print(f"  Written {added} new docs to {hn_corpus_path}")

    # Update query annotations with hard negative info
    qrels_by_query = {}
    for row in hn_qrels:
        qid = str(row["query-id"])
        did = row["corpus-id"]
        score = row["score"]
        if qid not in qrels_by_query:
            qrels_by_query[qid] = {}
        qrels_by_query[qid][did] = score

    # Copy queries and add new positive annotations
    import copy
    updated_queries = copy.deepcopy(existing_queries)
    num_updated = 0
    new_positives = 0
    for qid, q in updated_queries.items():
        if qid not in qrels_by_query:
            continue
        existing_rel_docids = {a["docid"] for a in q["annotations"]}
        for did, score in qrels_by_query[qid].items():
            if score == 1 and did not in existing_rel_docids:
                q["annotations"].append({"docid": did, "relevance": 1})
                new_positives += 1
                num_updated += 1
    print(f"  Added {new_positives} new positive annotations to {num_updated} queries")

    # Write updated queries to separate file
    print(f"Writing updated queries to {hn_queries_path} ...")
    with open(hn_queries_path, "w", encoding="utf-8") as f:
        for qid in sorted(updated_queries.keys(), key=lambda x: int(x)):
            f.write(json.dumps(updated_queries[qid], ensure_ascii=False) + "\n")
    print(f"  Written {len(updated_queries)} queries")

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE — {added} new hard negative docs written")
    print(f"{'='*60}")
    print(f"\nFiles created:")
    print(f"  {hn_corpus_path}")
    print(f"  {hn_queries_path}")
    print(f"\nOriginal files NOT modified:")
    print(f"  {corpus_path}")
    print(f"  {queries_path}")
    print(f"\nNext steps:")
    print(f"  1. Compute embeddings for new docs only:")
    print(f"     python precompute_embeddings.py \\")
    print(f"         --corpus_file {hn_corpus_path} \\")
    print(f"         --output_dir data/embeddings_hn --skip_queries")
    print(f"     python precompute_embeddings_api.py \\")
    print(f"         --corpus_file {hn_corpus_path} \\")
    print(f"         --output_dir data/embeddings_hn --skip_queries")
    print(f"  2. Merge corpus + embeddings:")
    print(f"     python merge_corpora.py")
    print(f"  3. Re-run reranker precomputation:")
    print(f"     python precompute_rerankers.py")
    print(f"  4. Evaluate:")
    print(f"     python evaluate.py --program_path initial.py --results_dir /tmp/test")


if __name__ == "__main__":
    main()
