"""Chunk every PDF and .py under the corpus, embed the chunks with a local
sentence-transformers model, and persist a FAISS index for retrieval.

Output layout (under index/):
    embeddings.npy   float32 array, shape (N, dim), row-normalized
    chunks.jsonl     one JSON object per line, aligned to embedding rows
    faiss.index      IndexFlatIP over the embeddings
    meta.json        model name, dim, count, chunk parameters, build timestamp

Usage:
    python build_index.py
    python build_index.py --corpus some/dir
    python build_index.py --chunk-size 800 --overlap 100
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build-index")


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CORPUS = Path("canvas_data")
DEFAULT_INDEX_DIR = Path("index")
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150

SKIP_SUFFIXES = {".json", ".md", ".png", ".jpg", ".jpeg", ".gif", ".svg",
                 ".zip", ".docx", ".pptx", ".xlsx", ".csv"}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str                      # "<module>/<file>#<idx>"
    text: str
    source_path: str             # relative to corpus root
    module: str
    source_type: str             # "pdf" | "code"
    page_start: Optional[int]    # 1-indexed, None for non-PDFs
    page_end: Optional[int]
    chunk_idx: int


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks no larger than `size` characters.

    Splits preferentially on paragraph, then line, then sentence boundaries,
    then on a hard character limit. Adjacent chunks share `overlap` chars
    from the tail of the previous chunk to preserve cross-boundary context.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    pieces = _explode([text], "\n\n", size)
    pieces = _explode(pieces, "\n", size)
    pieces = _explode(pieces, ". ", size)
    pieces = _hard_split(pieces, size)

    chunks: list[str] = []
    cur = ""
    for p in pieces:
        sep = "\n\n" if cur and cur.endswith((".", "!", "?")) else "\n"
        candidate = (cur + sep + p) if cur else p
        if len(candidate) <= size:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
        tail = cur[-overlap:] if overlap and overlap < len(cur) else ""
        cur = (tail + "\n" + p).strip() if tail else p
    if cur:
        chunks.append(cur)
    return chunks


def _explode(pieces: list[str], sep: str, size: int) -> list[str]:
    """Split any piece longer than `size` by `sep`; keep shorter pieces intact."""
    out: list[str] = []
    for p in pieces:
        if len(p) <= size:
            out.append(p)
            continue
        parts = [s for s in p.split(sep) if s.strip()]
        out.extend(parts if parts else [p])
    return out


def _hard_split(pieces: list[str], size: int) -> list[str]:
    """Slice any piece still longer than `size` at fixed character width."""
    out: list[str] = []
    for p in pieces:
        if len(p) <= size:
            out.append(p)
        else:
            for i in range(0, len(p), size):
                out.append(p[i:i + size])
    return out


# ---------------------------------------------------------------------------
# Source-file handlers
# ---------------------------------------------------------------------------

def iter_pdf_chunks(path: Path, module: str, corpus_root: Path,
                    chunk_size: int, overlap: int) -> Iterator[Chunk]:
    try:
        reader = PdfReader(str(path))
    except Exception as e:
        log.warning("Failed to open %s: %s", path, e)
        return

    rel = path.relative_to(corpus_root).as_posix()
    chunk_idx = 0
    for pnum, page in enumerate(reader.pages, start=1):
        try:
            ptext = (page.extract_text() or "").strip()
        except Exception as e:
            log.warning("Failed to extract page %d of %s: %s", pnum, path, e)
            continue
        if not ptext:
            continue
        for piece in chunk_text(ptext, chunk_size, overlap):
            yield Chunk(
                id=f"{module}/{path.name}#{chunk_idx}",
                text=piece,
                source_path=rel,
                module=module,
                source_type="pdf",
                page_start=pnum,
                page_end=pnum,
                chunk_idx=chunk_idx,
            )
            chunk_idx += 1


def iter_code_chunks(path: Path, module: str, corpus_root: Path,
                     chunk_size: int, overlap: int) -> Iterator[Chunk]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        log.warning("Failed to read %s: %s", path, e)
        return
    rel = path.relative_to(corpus_root).as_posix()
    for i, piece in enumerate(chunk_text(text, chunk_size, overlap)):
        yield Chunk(
            id=f"{module}/{path.name}#{i}",
            text=piece,
            source_path=rel,
            module=module,
            source_type="code",
            page_start=None,
            page_end=None,
            chunk_idx=i,
        )


def walk_corpus(root: Path, chunk_size: int, overlap: int) -> Iterator[Chunk]:
    if not root.exists():
        log.error("Corpus root %s does not exist. Run scrape_canvas.py first.", root)
        sys.exit(1)

    for module_dir in sorted(root.iterdir()):
        if not module_dir.is_dir():
            continue
        module = module_dir.name
        for path in sorted(module_dir.rglob("*")):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in SKIP_SUFFIXES:
                continue
            if suffix == ".pdf":
                yield from iter_pdf_chunks(path, module, root, chunk_size, overlap)
            elif suffix == ".py":
                yield from iter_code_chunks(path, module, root, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk + embed the Canvas corpus.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                        help="Root folder with scraped Canvas content (default: ./canvas_data)")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
                        help="Where to write embeddings.npy + chunks.jsonl + meta.json")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"sentence-transformers model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Max chars per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP,
                        help=f"Chars carried between chunks (default: {DEFAULT_OVERLAP})")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Embedding batch size (default: 32)")
    args = parser.parse_args()

    corpus_root = args.corpus.resolve()
    index_dir = args.index_dir.resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    log.info("Corpus:     %s", corpus_root)
    log.info("Index dir:  %s", index_dir)
    log.info("Model:      %s", args.model)
    log.info("Chunk size: %d chars, overlap %d", args.chunk_size, args.overlap)

    log.info("Walking corpus and chunking...")
    chunks: list[Chunk] = list(walk_corpus(corpus_root, args.chunk_size, args.overlap))
    if not chunks:
        log.error("No chunks produced. Is the corpus empty?")
        sys.exit(1)
    log.info("Collected %d chunks from %d unique source files.",
             len(chunks), len({c.source_path for c in chunks}))

    log.info("Loading embedding model (first run downloads ~90MB)...")
    model = SentenceTransformer(args.model)
    dim = (getattr(model, "get_embedding_dimension", None)
           or model.get_sentence_embedding_dimension)()
    log.info("Model loaded. Embedding dim = %d", dim)

    log.info("Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    assert embeddings.shape == (len(chunks), dim)

    np.save(index_dir / "embeddings.npy", embeddings)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    # Inner product over L2-normalized vectors == cosine similarity.
    log.info("Building FAISS index (IndexFlatIP)...")
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss_path = index_dir / "faiss.index"
    faiss.write_index(faiss_index, str(faiss_path))

    meta = {
        "model": args.model,
        "dim": dim,
        "count": len(chunks),
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "corpus_root": str(corpus_root),
        "faiss_index_type": "IndexFlatIP",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Wrote %s (%.1f MB)", index_dir / "embeddings.npy",
             embeddings.nbytes / 1e6)
    log.info("Wrote %s", index_dir / "chunks.jsonl")
    log.info("Wrote %s", faiss_path)
    log.info("Wrote %s", index_dir / "meta.json")
    log.info("Done. Try: python query.py \"what is a linked list\"  "
             "or: python chat.py")


if __name__ == "__main__":
    main()
