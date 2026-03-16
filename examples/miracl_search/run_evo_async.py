#!/usr/bin/env python3
from shinka.core import AsyncEvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
import argparse
import asyncio
import atexit
import os
import subprocess
import sys
import yaml

CORPUS_SERVER_SOCKET = "/tmp/miracl_corpus_server.sock"


def _start_corpus_server(
    corpus_path: str, embeddings_dir: str = None, rerankers_dir: str = None,
) -> subprocess.Popen:
    """Launch the corpus server as a background process."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(script_dir, "corpus_server.py")
    cmd = [sys.executable, server_script, "--corpus_path", corpus_path]
    if embeddings_dir:
        cmd.extend(["--embeddings_dir", embeddings_dir])
    if rerankers_dir:
        cmd.extend(["--rerankers_dir", rerankers_dir])
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    atexit.register(lambda: _stop_corpus_server(proc))
    return proc


def _stop_corpus_server(proc: subprocess.Popen):
    """Terminate the corpus server if still running."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


async def _wait_for_server(timeout: float = 120.0, poll_interval: float = 1.0):
    """Poll until the corpus server socket appears (corpus loading done)."""
    elapsed = 0.0
    while elapsed < timeout:
        if os.path.exists(CORPUS_SERVER_SOCKET):
            print("Corpus server is ready.")
            return
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(
        f"Corpus server did not become ready within {timeout}s"
    )


async def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, "data", "corpus.jsonl")

    # Detect embeddings directory
    embeddings_dir = os.path.join(script_dir, "data", "embeddings")
    if not os.path.isdir(embeddings_dir):
        embeddings_dir = None

    # Detect rerankers directory
    rerankers_dir = os.path.join(script_dir, "data", "rerankers")
    if not os.path.isdir(rerankers_dir):
        rerankers_dir = None

    # Start corpus server
    server_proc = _start_corpus_server(
        corpus_path, embeddings_dir=embeddings_dir, rerankers_dir=rerankers_dir,
    )
    print(f"Corpus server started (pid={server_proc.pid}), waiting for ready ...")
    await _wait_for_server()

    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(eval_program_path="evaluate.py", time="00:05:00")
    db_config = DatabaseConfig(**config["db_config"])

    runner = AsyncEvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config["max_evaluation_jobs"],
        max_proposal_jobs=config["max_proposal_jobs"],
        max_db_workers=config["max_db_workers"],
        debug=False,
        verbose=True,
    )

    try:
        await runner.run()
    finally:
        _stop_corpus_server(server_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_small.yaml")
    args = parser.parse_args()
    asyncio.run(main(args.config_path))
