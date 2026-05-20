"""Top-K retrieval CLI for sanity-checking the local index.

Embeds a query string, pulls the top-K chunks from the FAISS index, and
prints each hit with its source label and a short preview. No LLM is
involved.

Usage:
    python query.py "what is a linked list"
    python query.py --k 10 "how do you balance an AVL tree"
    python query.py --full "explain the quicksort partition step"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from retriever import Hit, Retriever


DEFAULT_INDEX_DIR = Path("index")
DEFAULT_K = 5
DEFAULT_PREVIEW = 280


def format_hit(rank: int, hit: Hit, preview_chars: int, show_full: bool) -> str:
    text = hit.chunk["text"]
    if not show_full and len(text) > preview_chars:
        text = text[:preview_chars].rstrip() + "..."
    header = f"#{rank}  score={hit.score:.4f}  {hit.source_label}"
    divider = "-" * len(header)
    return f"{header}\n{divider}\n{text}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-K retrieval over the local Canvas index.")
    parser.add_argument("query", nargs="+", help="The natural-language query.")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help=f"How many hits to show (default: {DEFAULT_K})")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR,
                        help="Directory containing the built index.")
    parser.add_argument("--full", action="store_true",
                        help="Print full chunk text instead of truncated preview.")
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW,
                        help=f"Preview length in chars (default: {DEFAULT_PREVIEW})")
    args = parser.parse_args()

    query = " ".join(args.query)
    retriever = Retriever(args.index_dir.resolve())
    hits = retriever.search(query, k=args.k)

    print(f"\nQuery: {query}")
    print(f"Index: {len(retriever.chunks)} chunks, model={retriever.meta['model']}\n")
    for rank, hit in enumerate(hits, start=1):
        print(format_hit(rank, hit, args.preview_chars, args.full))


if __name__ == "__main__":
    main()
