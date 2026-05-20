"""Shared retrieval layer that loads the index produced by build_index.py
and exposes a top-K cosine-similarity search.

The query is encoded with the same sentence-transformer model that built
the corpus index, so retrieval scores are consistent with how the corpus
was embedded.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class Hit:
    """A single retrieval result: similarity score plus chunk metadata."""

    score: float
    chunk: dict[str, Any]

    @property
    def source_label(self) -> str:
        """Human-readable source location, e.g. 'foo.pdf (page 7)'."""
        label = self.chunk["source_path"]
        start = self.chunk.get("page_start")
        end = self.chunk.get("page_end")
        if start and end and end != start:
            label += f" (pages {start}-{end})"
        elif start:
            label += f" (page {start})"
        return label


class Retriever:
    """Top-K cosine retrieval over a FAISS IndexFlatIP."""

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        meta_path = index_dir / "meta.json"
        chunks_path = index_dir / "chunks.jsonl"
        faiss_path = index_dir / "faiss.index"

        for p in (meta_path, chunks_path, faiss_path):
            if not p.exists():
                print(f"Missing {p}. Run build_index.py first.", file=sys.stderr)
                sys.exit(1)

        self.meta: dict[str, Any] = json.loads(meta_path.read_text())
        self.chunks: list[dict[str, Any]] = [
            json.loads(line) for line in chunks_path.read_text().splitlines() if line
        ]
        self.index = faiss.read_index(str(faiss_path))

        if self.index.ntotal != len(self.chunks):
            print(f"Index inconsistent: {self.index.ntotal} vectors vs "
                  f"{len(self.chunks)} chunks. Rebuild.", file=sys.stderr)
            sys.exit(1)

        self.model = SentenceTransformer(self.meta["model"])

    def search(self, query: str, k: int = 5) -> list[Hit]:
        q_vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        k = min(k, self.index.ntotal)
        scores, ids = self.index.search(q_vec, k)
        return [
            Hit(score=float(s), chunk=self.chunks[int(i)])
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]
