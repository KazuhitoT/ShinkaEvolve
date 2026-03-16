# MIRACL 日本語情報検索タスク

[MIRACL](https://huggingface.co/datasets/miracl/miracl) (Multilingual Information Retrieval Across a Continuum of Languages) の日本語サブセットを対象とした情報検索タスク。コーパスからクエリに関連するパッセージを検索・ランキングする `search()` 関数を進化させる。

## ディレクトリ構成

| ファイル | 説明 |
|---------|------|
| `initial.py` | 進化対象プログラム。`search()` 関数（EVOLVE-BLOCK内）と固定のインデックス構築・検索基盤 |
| `evaluate.py` | 評価スクリプト。NDCG@10, Precision@10, MRR@10 を計算 |
| `run_evo.py` | スタンドアロン同期版の進化実行スクリプト |
| `run_evo_async.py` | スタンドアロン非同期版（corpus_server 自動起動） |
| `corpus_server.py` | コーパスサーバー。Arrow mmap + fork でメモリ共有し高速評価 |
| `shinka_small.yaml` | スタンドアロン用設定ファイル |
| `gen_data.py` | MIRACL データセットからコーパス・クエリを抽出 |
| `precompute_embeddings.py` | HuggingFace モデルで埋め込みを事前計算 |
| `precompute_embeddings_api.py` | OpenAI / Gemini API で埋め込みを事前計算 |
| `precompute_rerankers.py` | CrossEncoder リランカースコアを事前計算 |
| `embedding_loader.py` | 埋め込みデータのロードユーティリティ |
| `reranker_loader.py` | リランカーデータのロードユーティリティ |
| `data/` | コーパス、クエリ、埋め込み、リランカーデータ |

## データ準備

### 1. コーパス・クエリの抽出

```bash
# プロジェクトのルートフォルダで下記を実行
.venv/bin/python examples/miracl_search/gen_data.py
```

`data/corpus.jsonl`（アノテーション必須文書 + ランダム3%サンプル）と `data/queries_eval.jsonl` が生成される。

### 2. 埋め込みの事前計算

ローカル HuggingFace モデル（`cl-nagoya/ruri-v3-30m`, `hotchpotch/static-embedding-japanese`）:

```bash
.venv/bin/python examples/miracl_search/precompute_embeddings.py
```

API モデル（`text-embedding-3-small`, `gemini-embedding-001`）:

```bash
# 環境変数に API キーを設定
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
.venv/bin/python examples/miracl_search/precompute_embeddings_api.py
```

出力先: `data/embeddings/` に各モデルの `corpus.npy`, `queries.npy`, `metadata.json` と `manifest.json`。

### 3. リランカースコアの事前計算

埋め込みの事前計算が完了した後に実行:

```bash
.venv/bin/python examples/miracl_search/precompute_rerankers.py
```

出力先: `data/rerankers/` に各モデルの `doc_indices.npy`, `scores.npy`, `metadata.json` と `manifest.json`。

## 評価テスト

```bash
.venv/bin/python examples/miracl_search/evaluate.py \
  --program_path examples/miracl_search/initial.py \
  --results_dir /tmp/miracl_test
```

オプション引数:
- `--queries_file` — クエリファイル（デフォルト: `data/queries_eval.jsonl`）
- `--corpus_path` — コーパスファイル（デフォルト: `data/corpus.jsonl`）
- `--embeddings_dir` — 埋め込みディレクトリ（デフォルト: `data/embeddings/`）
- `--rerankers_dir` — リランカーディレクトリ（デフォルト: `data/rerankers/`）

## 進化実行

### スタンドアロン同期版

```bash
cd examples/miracl_search
../../.venv/bin/python run_evo.py
```

### スタンドアロン非同期版（推奨）

corpus_server を自動起動し、コーパスをメモリ共有して並列評価を高速化:

```bash
cd examples/miracl_search
../../.venv/bin/python run_evo_async.py
```

設定は `shinka_small.yaml` で変更可能（`--config_path` で別ファイル指定も可）。

### Hydra 経由

```bash
shinka_launch variant=miracl_search_example
```

世代数を変更する場合:
```bash
shinka_launch variant=miracl_search_example evo_config.num_generations=5
```

## 評価メトリクス

| メトリクス | 説明 |
|-----------|------|
| `combined_score` | フィットネス値 = `mean_ndcg_at_10`（高いほど良い） |
| `mean_ndcg_at_10` | 全クエリの NDCG@10 平均。ランキング品質の主指標 |
| `mean_precision_at_10` | 全クエリの Precision@10 平均。上位10件中の関連文書率 |
| `mean_mrr_at_10` | 全クエリの MRR@10 平均。最初の関連文書の順位の逆数 |
| `time_seconds` | 評価実行時間（秒） |

## 進化対象（EVOLVE-BLOCK）

`initial.py` 内の `search()` 関数と `_normalize_scores()` ヘルパーが進化対象。

`search()` は以下の信号を統合してクエリに対する上位 k 件のパッセージを返す:

- **疎ベクトルスコア** — bigram Jaccard 類似度（`score_passage()` / `doc_token_sets`）
- **蜜ベクトルスコア** — 事前計算済み埋め込みベクトルのコサイン類似度（4モデル）
- **リランカースコア** — CrossEncoder による精密スコア（2モデル）

### 利用可能な埋め込みモデル

| モデル | 次元数 |
|-------|--------|
| `cl-nagoya/ruri-v3-30m` | 256 |
| `hotchpotch/static-embedding-japanese` | 1024 |
| `text-embedding-3-small` | 1536 |
| `gemini-embedding-001` | 768 |

### 利用可能なリランカーモデル

| モデル |
|-------|
| `cl-nagoya/ruri-v3-reranker-310m` |
| `hotchpotch/japanese-reranker-tiny-v2` |
