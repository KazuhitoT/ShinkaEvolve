#!/usr/bin/env python3
"""Corpus server: loads corpus once, serves evaluation requests via Unix socket.

Each evaluation request is handled in a forked child process.  The corpus is
stored in Arrow IPC format and memory-mapped so that the OS page cache is
shared across all processes without CPython reference-count COW issues.

Usage:
    python corpus_server.py --corpus_path data/corpus.jsonl
"""

import argparse
import gc
import importlib.util
import json
import os
import pickle
import signal
import socket
import struct
import sys
import threading
import time
import traceback

import pyarrow as pa
import pyarrow.ipc as ipc


DEFAULT_SOCKET_PATH = "/tmp/miracl_corpus_server.sock"


class ArrowCorpusView:
    """Arrow Table を List[Dict[str, str]] のように振る舞わせるラッパー。

    各 doc は __getitem__ / __iter__ 時にオンデマンドで Python dict を生成。
    Arrow のバッファは mmap backed なのでプロセス間で物理メモリを共有。
    """

    def __init__(self, table: pa.Table):
        self._docid = table.column("docid")
        self._title = table.column("title")
        self._text = table.column("text")
        self._len = len(table)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._len))]
        if idx < 0:
            idx += self._len
        return {
            "docid": self._docid[idx].as_py(),
            "title": self._title[idx].as_py(),
            "text": self._text[idx].as_py(),
        }

    def __iter__(self):
        for i in range(self._len):
            yield self[i]


def _load_jsonl(path):
    """Load corpus from JSONL file into a list of dicts."""
    corpus = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    return corpus


def load_corpus_arrow(jsonl_path):
    """Load corpus as an Arrow Table via memory-mapped IPC file.

    On the first call the JSONL is converted to Arrow IPC format.
    Subsequent calls just open the existing .arrow file via mmap.
    """
    arrow_path = jsonl_path + ".arrow"
    needs_convert = not os.path.exists(arrow_path) or (
        os.path.getmtime(jsonl_path) > os.path.getmtime(arrow_path)
    )
    if needs_convert:
        print(f"Converting {jsonl_path} -> {arrow_path} ...")
        corpus = _load_jsonl(jsonl_path)
        table = pa.table(
            {
                "docid": [d["docid"] for d in corpus],
                "title": [d.get("title", "") for d in corpus],
                "text": [d["text"] for d in corpus],
            }
        )
        with pa.OSFile(arrow_path, "wb") as f:
            writer = ipc.new_file(f, table.schema)
            for batch in table.to_batches(max_chunksize=100_000):
                writer.write_batch(batch)
            writer.close()
        del corpus, table

    source = pa.memory_map(arrow_path, "r")
    reader = ipc.open_file(source)
    return reader.read_all()


def handle_evaluation(
    corpus_table, program_path, queries,
    embedding_registry=None, reranker_registry=None,
    prebuilt_index=None,
):
    """Run evaluation in a forked child process.

    GC is disabled to prevent unnecessary collection cycles in the
    short-lived child.  The ArrowCorpusView wraps the mmap-backed table
    so that the evolved program can use ``corpus[i]`` / ``for doc in corpus``
    without materialising all dicts at once.
    """
    gc.disable()
    corpus_view = ArrowCorpusView(corpus_table)

    spec = importlib.util.spec_from_file_location("evolved", program_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return module.run_search_batch(
            queries, corpus=corpus_view,
            embedding_registry=embedding_registry,
            reranker_registry=reranker_registry,
            prebuilt_index=prebuilt_index,
        )
    except TypeError:
        # Fallback for older programs that don't accept prebuilt_index/new kwargs
        try:
            return module.run_search_batch(
                queries, corpus=corpus_view,
                embedding_registry=embedding_registry,
                reranker_registry=reranker_registry,
            )
        except TypeError:
            try:
                return module.run_search_batch(
                    queries, corpus=corpus_view,
                    embedding_registry=embedding_registry,
                )
            except TypeError:
                return module.run_search_batch(queries, corpus=corpus_view)


def send_msg(sock, data):
    """Send a length-prefixed pickle message."""
    raw = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(raw)) + raw)


def recv_msg(sock):
    """Receive a length-prefixed pickle message."""
    header = _recv_exact(sock, 4)
    if not header:
        return None
    size = struct.unpack("!I", header)[0]
    return pickle.loads(_recv_exact(sock, size))


def _recv_exact(sock, n):
    """Read exactly *n* bytes from *sock*."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed before all bytes received")
        buf += chunk
    return buf


def _reap_children(signum, frame):
    """Reap finished child processes to avoid zombies."""
    while True:
        try:
            os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break


def main(corpus_path, socket_path=DEFAULT_SOCKET_PATH, embeddings_dir=None,
         rerankers_dir=None):
    print(f"Loading corpus from {corpus_path} (Arrow mmap) ...")
    corpus_table = load_corpus_arrow(corpus_path)
    print(f"Corpus loaded: {len(corpus_table)} docs", flush=True)

    # Load embedding registry if available
    embedding_registry = None
    if embeddings_dir:
        from embedding_loader import load_registry

        embedding_registry = load_registry(embeddings_dir)
        if embedding_registry is not None:
            models = embedding_registry.available_models()
            print(f"Embeddings loaded: {len(models)} models — {models}", flush=True)
            # Eagerly load all stores so mmap pages are faulted in before fork
            for m in models:
                embedding_registry.get_store(m)
        else:
            print(f"No embeddings found at {embeddings_dir}", flush=True)

    # Load reranker registry if available
    reranker_registry = None
    if rerankers_dir:
        from reranker_loader import load_reranker_registry

        reranker_registry = load_reranker_registry(rerankers_dir)
        if reranker_registry is not None:
            models = reranker_registry.available_models()
            print(f"Rerankers loaded: {len(models)} models — {models}", flush=True)
            for m in models:
                reranker_registry.get_store(m)
        else:
            print(f"No rerankers found at {rerankers_dir}", flush=True)

    # Pre-build inverted index in parent process (shared with children via COW)
    print("Pre-building inverted index ...", flush=True)
    corpus_view = ArrowCorpusView(corpus_table)
    from initial import build_index as _build_index
    prebuilt_index = _build_index(corpus_view)
    print("Index built.", flush=True)

    # Clean up stale socket file
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(128)
    print(f"Listening on {socket_path}", flush=True)

    # Automatically reap zombie children
    signal.signal(signal.SIGCHLD, _reap_children)

    try:
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                # Interrupted by SIGCHLD -- retry
                continue

            try:
                request = recv_msg(conn)
            except Exception:
                conn.close()
                continue
            if request is None:
                conn.close()
                continue

            try:
                pid = os.fork()
            except OSError as e:
                print(f"fork() failed: {e}", flush=True)
                send_msg(conn, {"success": False, "error": f"Server fork failed: {e}"})
                conn.close()
                continue
            if pid == 0:
                # --- child process ---
                srv.close()  # child does not need the listening socket
                try:
                    results = handle_evaluation(
                        corpus_table,
                        request["program_path"],
                        request["queries"],
                        embedding_registry=embedding_registry,
                        reranker_registry=reranker_registry,
                        prebuilt_index=prebuilt_index,
                    )
                    send_msg(conn, {"success": True, "results": results})
                except Exception:
                    send_msg(conn, {"success": False, "error": traceback.format_exc()})
                finally:
                    conn.close()
                    os._exit(0)
            else:
                # --- parent process ---
                conn.close()
                # Kill child if it runs longer than 40 minutes
                def _kill_after_timeout(child_pid, timeout=90):
                    time.sleep(timeout)
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # already exited

                threading.Thread(
                    target=_kill_after_timeout, args=(pid,), daemon=True
                ).start()
    except KeyboardInterrupt:
        print("\nShutting down corpus server.")
    finally:
        srv.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIRACL corpus server (fork-based)")
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Path to corpus.jsonl",
    )
    parser.add_argument(
        "--socket_path",
        type=str,
        default=DEFAULT_SOCKET_PATH,
        help=f"Unix socket path (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help="Path to precomputed embeddings directory",
    )
    parser.add_argument(
        "--rerankers_dir",
        type=str,
        default=None,
        help="Path to precomputed rerankers directory",
    )
    args = parser.parse_args()
    main(args.corpus_path, args.socket_path,
         embeddings_dir=args.embeddings_dir,
         rerankers_dir=args.rerankers_dir)
