"""Extract full corpus and evaluation queries from MIRACL for Japanese information retrieval."""

import json
import os
import random

import datasets


def main():
    lang = "ja"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Save evaluation queries with relevance annotations ---
    print(f"Loading MIRACL queries for language: {lang}")
    miracl = datasets.load_dataset("miracl/miracl", lang, trust_remote_code=True)
    dev = miracl["dev"]

    queries = []
    required_docids = set()
    for data in dev:
        annotations = []
        for p in data["positive_passages"]:
            annotations.append({"docid": p["docid"], "relevance": 1})
            required_docids.add(p["docid"])
        for p in data["negative_passages"]:
            annotations.append({"docid": p["docid"], "relevance": 0})
            required_docids.add(p["docid"])
        queries.append({
            "query_id": data["query_id"],
            "query": data["query"],
            "annotations": annotations,
        })

    print(f"Total queries in dev split: {len(queries)}")
    print(f"Required docids from annotations: {len(required_docids)}")

    random.seed(42)
    # sampled = random.sample(queries, min(50, len(queries)))
    # print(f"Sampled {len(sampled)} queries")

    queries_path = os.path.join(out_dir, "queries_eval.jsonl")
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # --- 2. Save sampled corpus (required docs + 10% random sample) ---
    print(f"Loading MIRACL corpus for language: {lang}")
    corpus_ds = datasets.load_dataset(
        "miracl/miracl-corpus", lang, trust_remote_code=True
    )["train"]

    corpus_path = os.path.join(out_dir, "corpus.jsonl")
    print(f"Writing corpus to {corpus_path} ...")
    random.seed(42)
    num_docs = 0
    num_total = 0
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc in corpus_ds:
            num_total += 1
            # if doc["docid"] in required_docids or random.random() < 0.01:
            if doc["docid"] in required_docids or random.random() < 0.03:
                f.write(
                    json.dumps(
                        {
                            "docid": doc["docid"],
                            "title": doc["title"],
                            "text": doc["text"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                num_docs += 1
    print(f"Corpus saved: {num_docs} / {num_total} documents (sampled ~{num_docs * 100 // num_total}%)")

    print(f"Queries saved to {queries_path}")


if __name__ == "__main__":
    main()
