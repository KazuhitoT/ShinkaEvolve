#!/usr/bin/env python3
"""Merge hard negative corpus and embeddings into the main dataset.

Appends new documents from corpus_hard_negatives.jsonl to corpus.jsonl,
and concatenates embedding numpy arrays so existing indices stay valid.

Usage:
    python merge_corpora.py [--data_dir data]

This will:
    1. Append corpus_hard_negatives.jsonl docs to corpus.jsonl
    2. For each embedding model in embeddings/:
       - Concatenate corpus.npy with embeddings_hn/{model}/corpus.npy
       - queries.npy stays unchanged
    3. Replace queries_eval.jsonl with queries_eval_hn.jsonl (updated annotations)
    4. Remove stale reranker data (must re-run precompute_rerankers.py)
    5. Remove stale Arrow cache
"""

import argparse
import json
import os
import shutil

import numpy as np


def _count_lines(path):
    """Count non-empty lines in a JSONL file."""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Merge hard negative corpus and embeddings into main dataset"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Data directory (default: <script_dir>/data)",
    )
    parser.add_argument(
        "--hn_corpus_file", type=str, default=None,
        help="Hard negatives corpus file (default: <data_dir>/corpus_hard_negatives.jsonl)",
    )
    parser.add_argument(
        "--hn_embeddings_dir", type=str, default=None,
        help="Hard negatives embeddings dir (default: <data_dir>/embeddings_hn)",
    )
    parser.add_argument(
        "--hn_queries_file", type=str, default=None,
        help="Updated queries file (default: <data_dir>/queries_eval_hn.jsonl)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.join(script_dir, "data")

    corpus_path = os.path.join(data_dir, "corpus.jsonl")
    hn_corpus_path = args.hn_corpus_file or os.path.join(data_dir, "corpus_hard_negatives.jsonl")
    embeddings_dir = os.path.join(data_dir, "embeddings")
    hn_embeddings_dir = args.hn_embeddings_dir or os.path.join(data_dir, "embeddings_hn")
    queries_path = os.path.join(data_dir, "queries_eval.jsonl")
    hn_queries_path = args.hn_queries_file or os.path.join(data_dir, "queries_eval_hn.jsonl")
    rerankers_dir = os.path.join(data_dir, "rerankers")

    # --- Validate inputs ---
    if not os.path.exists(hn_corpus_path):
        print(f"ERROR: {hn_corpus_path} not found. Run prepare_hard_negatives.py first.")
        return

    original_count = _count_lines(corpus_path)
    hn_count = _count_lines(hn_corpus_path)
    print(f"Original corpus: {original_count} docs")
    print(f"Hard negatives:  {hn_count} docs")
    print(f"Merged total:    {original_count + hn_count} docs")

    # --- 1. Backup original corpus ---
    backup_path = corpus_path + ".pre_merge.bak"
    if not os.path.exists(backup_path):
        shutil.copy2(corpus_path, backup_path)
        print(f"\nBacked up corpus to {backup_path}")

    # --- 2. Append hard negative docs to corpus ---
    print(f"\nAppending {hn_count} docs to {corpus_path} ...")
    with open(corpus_path, "a", encoding="utf-8") as out:
        with open(hn_corpus_path, encoding="utf-8") as inp:
            for line in inp:
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
    merged_count = _count_lines(corpus_path)
    print(f"  Merged corpus: {merged_count} docs")

    # --- 3. Merge embeddings ---
    if os.path.isdir(embeddings_dir) and os.path.isdir(hn_embeddings_dir):
        manifest_path = os.path.join(embeddings_dir, "manifest.json")
        hn_manifest_path = os.path.join(hn_embeddings_dir, "manifest.json")

        if os.path.exists(manifest_path) and os.path.exists(hn_manifest_path):
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            with open(hn_manifest_path, encoding="utf-8") as f:
                hn_manifest = json.load(f)

            print(f"\n--- Merging embeddings ---")
            for model_name, model_info in manifest.get("models", {}).items():
                model_dir = os.path.join(embeddings_dir, model_info["dir"])
                hn_model_info = hn_manifest.get("models", {}).get(model_name)
                if hn_model_info is None:
                    print(f"  {model_name}: no hard negative embeddings found, skipping")
                    continue

                hn_model_dir = os.path.join(hn_embeddings_dir, hn_model_info["dir"])
                corpus_npy = os.path.join(model_dir, "corpus.npy")
                hn_corpus_npy = os.path.join(hn_model_dir, "corpus.npy")

                if not os.path.exists(hn_corpus_npy):
                    print(f"  {model_name}: {hn_corpus_npy} not found, skipping")
                    continue

                # Load arrays
                original_emb = np.load(corpus_npy)
                hn_emb = np.load(hn_corpus_npy)

                if original_emb.shape[1] != hn_emb.shape[1]:
                    print(f"  {model_name}: dimension mismatch "
                          f"({original_emb.shape[1]} vs {hn_emb.shape[1]}), skipping")
                    continue

                # Concatenate
                merged_emb = np.concatenate([original_emb, hn_emb], axis=0)
                print(f"  {model_name}: {original_emb.shape} + {hn_emb.shape} "
                      f"-> {merged_emb.shape}")

                # Save (overwrite original corpus.npy)
                np.save(corpus_npy, merged_emb)

                # Update metadata
                meta_path = os.path.join(model_dir, "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, encoding="utf-8") as f:
                        meta = json.load(f)
                    meta["corpus_count"] = int(merged_emb.shape[0])
                    with open(meta_path, "w", encoding="utf-8") as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
        else:
            print("\nManifest files missing, skipping embedding merge")
    else:
        print(f"\nEmbeddings directories not found, skipping embedding merge")

    # --- 4. Replace queries with updated version ---
    if os.path.exists(hn_queries_path):
        queries_backup = queries_path + ".pre_merge.bak"
        if not os.path.exists(queries_backup):
            shutil.copy2(queries_path, queries_backup)
            print(f"\nBacked up queries to {queries_backup}")
        shutil.copy2(hn_queries_path, queries_path)
        print(f"Replaced queries with updated version from {hn_queries_path}")
    else:
        print(f"\n{hn_queries_path} not found, keeping original queries")

    # --- 5. Remove stale Arrow cache ---
    arrow_path = corpus_path + ".arrow"
    if os.path.exists(arrow_path):
        os.remove(arrow_path)
        print(f"Removed stale Arrow cache: {arrow_path}")

    # --- 6. Warn about stale rerankers ---
    if os.path.isdir(rerankers_dir):
        print(f"\nWARNING: rerankers/ exists — corpus indices have changed!")
        print(f"  You MUST re-run: python precompute_rerankers.py")

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE — Corpus merged: {original_count} + {hn_count} = {merged_count} docs")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Re-run reranker precomputation:")
    print(f"     python precompute_rerankers.py")
    print(f"  2. Evaluate:")
    print(f"     python evaluate.py --program_path initial.py --results_dir /tmp/test")


if __name__ == "__main__":
    main()
