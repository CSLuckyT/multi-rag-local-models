#!/usr/bin/env python3
"""Prepare HotPotQA artifacts and build FAISS retrieval artifacts."""

import argparse

from arag import Config
from arag.data.hotpotqa import build_hotpotqa_artifacts
from arag.retrieval.faiss_store import FaissArtifactStore
from arag.utils.device import format_device_message


def build_index(config: Config, force_rebuild: bool = False):
    data_dir = config.get("data.prepared_dir", "data/hotpotqa")
    retrieval_dir = config.get("retrieval.artifact_dir", "data/hotpotqa/index")
    embed_limit = config.get("data.embed_sample_limit")
    question_limit = config.get("data.question_sample_limit")
    split_seed = config.get("data.split_seed", 42)
    reuse_prepared = config.get("data.reuse_prepared_data", True)
    reuse_existing = config.get("retrieval.reuse_existing", True) and not force_rebuild

    print(format_device_message(config.get("embedding.device") or config.get("runtime.device")))

    artifacts = build_hotpotqa_artifacts(
        output_dir=data_dir,
        seed=split_seed,
        embed_sample_limit=embed_limit,
        question_sample_limit=question_limit,
    )

    import json

    chunk_records = json.loads(open(artifacts.get("corpus_chunks", artifacts["train_chunks"]), "r", encoding="utf-8").read())
    store = FaissArtifactStore(
        artifact_dir=retrieval_dir,
        embedding_model=config.get("embedding.model", "BAAI/bge-m3"),
        device=config.get("embedding.device") or config.get("runtime.device"),
    )

    expected_metadata = {
        "embed_sample_limit": config.get("data.embed_sample_limit"),
        "question_sample_limit": config.get("data.question_sample_limit"),
        "split_seed": split_seed,
    }
    if reuse_existing and store.exists() and store.matches(expected_metadata):
        print(f"Reusing existing retrieval artifacts from {retrieval_dir}")
        return artifacts

    batch_size = config.get("embedding.batch_size", 32)
    store.build(chunk_records=chunk_records, batch_size=batch_size, expected_metadata=expected_metadata)
    print(f"Built FAISS retrieval artifacts in {retrieval_dir}")
    return artifacts


def main():
    parser = argparse.ArgumentParser(description="Build HotPotQA FAISS index")
    parser.add_argument("--config", "-c", required=True, help="Config file path")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild prepared data and retrieval index")

    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    build_index(config=config, force_rebuild=args.force_rebuild)


if __name__ == "__main__":
    main()
