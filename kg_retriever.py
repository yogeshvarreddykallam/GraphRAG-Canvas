"""kg_retriever.py — Graph-enhanced retrieval layer for GraphRAG-Canvas.

Wraps the original vector Retriever with a two-stage pipeline:
  1. KG Expansion  — extract concepts from the query, walk the knowledge
                     graph to find related/prerequisite concepts, expand
                     the query string with neighbour labels.
  2. Merged Ranking — run both the expanded vector search and a KG-concept
                     filter, then merge and deduplicate results ranked by a
                     combined score.

Usage (standalone test):
    python kg_retriever.py "how does a binary search tree work"
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import spacy
from rdflib import Graph, Namespace
from rdflib.namespace import RDFS
from spacy.matcher import PhraseMatcher

from retriever import Hit, Retriever

log = logging.getLogger("kg-retriever")

CKG = Namespace("http://yogeshvarreddykallam.github.io/canvas-kg#")
RELATED_TO = CKG.relatedTo
HAS_PREREQ = CKG.hasPrerequisite


# ─────────────────────────────────────────────────────────────
#  Data types
# ─────────────────────────────────────────────────────────────

@dataclass
class KGHit:
    """A retrieval result that carries both vector score and KG provenance."""
    score: float
    chunk: dict[str, Any]
    kg_boost: float = 0.0           # extra score from KG expansion
    matched_concepts: list[str] = field(default_factory=list)

    @property
    def combined_score(self) -> float:
        return self.score + self.kg_boost

    @property
    def source_label(self) -> str:
        label = self.chunk["source_path"]
        start = self.chunk.get("page_start")
        end   = self.chunk.get("page_end")
        if start and end and end != start:
            label += f" (pages {start}–{end})"
        elif start:
            label += f" (page {start})"
        return label


# ─────────────────────────────────────────────────────────────
#  KGRetriever
# ─────────────────────────────────────────────────────────────

class KGRetriever:
    """
    Augments the base vector Retriever with knowledge-graph expansion.

    Pipeline
    --------
    query
      → [spaCy + PhraseMatcher]  extract query concepts
      → [NetworkX traversal]     find K-hop neighbours in KG
      → [expanded query string]  append neighbour labels
      → [FAISS search]           vector search over expanded query
      → [KG boost]               score bonus for chunks that contain
                                 query concepts or their neighbours
      → merged, re-ranked results
    """

    def __init__(
        self,
        index_dir: Path = Path("index"),
        kg_dir: Path = Path("kg"),
        ontology_path: Path = Path("ontology/canvas_kg.ttl"),
        spacy_model: str = "en_core_web_sm",
        hop_depth: int = 1,
        kg_boost_weight: float = 0.15,
    ) -> None:
        self.hop_depth = hop_depth
        self.kg_boost_weight = kg_boost_weight

        # ── Base vector retriever ──────────────────────────────
        self.base = Retriever(index_dir)

        # ── Load OWL ontology labels ───────────────────────────
        rdf = Graph()
        rdf.parse(str(ontology_path), format="turtle")
        self.label_to_uri: dict[str, str] = {}
        self.uri_to_label: dict[str, str] = {}
        for subj, _, label in rdf.triples((None, RDFS.label, None)):
            lstr = str(label).lower()
            self.label_to_uri[lstr] = str(subj)
            self.uri_to_label[str(subj)] = str(label)

        log.info("Loaded %d ontology concepts", len(self.label_to_uri))

        # ── Load NetworkX graph ────────────────────────────────
        gml_path = kg_dir / "canvas_kg.graphml"
        if gml_path.exists():
            self.nx_graph: nx.Graph | None = nx.read_graphml(str(gml_path))
            log.info(
                "KG loaded: %d nodes, %d edges",
                self.nx_graph.number_of_nodes(),
                self.nx_graph.number_of_edges(),
            )
        else:
            self.nx_graph = None
            log.warning(
                "GraphML not found at %s. Run kg_builder.py first. "
                "Falling back to vector-only retrieval.",
                gml_path,
            )

        # ── spaCy PhraseMatcher ────────────────────────────────
        self.nlp = spacy.load(spacy_model, disable=["ner", "parser", "lemmatizer"])
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(lbl) for lbl in self.label_to_uri]
        self.matcher.add("CONCEPT", patterns)

    # ── Internal helpers ───────────────────────────────────────

    def _extract_query_concepts(self, query: str) -> list[str]:
        """Return list of concept URIs found in the query string."""
        doc = self.nlp(query)
        matches = self.matcher(doc)
        seen: set[str] = set()
        uris: list[str] = []
        for _, start, end in matches:
            label = doc[start:end].text.lower()
            if label not in seen and label in self.label_to_uri:
                seen.add(label)
                uris.append(self.label_to_uri[label])
        return uris

    def _expand_via_kg(self, concept_uris: list[str]) -> list[str]:
        """
        Walk `hop_depth` hops in the NetworkX graph and return
        neighbour concept URIs (excluding seed nodes themselves).
        """
        if self.nx_graph is None or not concept_uris:
            return []

        visited: set[str] = set(concept_uris)
        frontier: set[str] = set(concept_uris)

        for _ in range(self.hop_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if self.nx_graph.has_node(node):
                    for neighbour in self.nx_graph.neighbors(node):
                        if neighbour not in visited:
                            next_frontier.add(neighbour)
            visited |= next_frontier
            frontier = next_frontier

        # Return only the neighbour URIs (not seeds)
        return [u for u in visited if u not in set(concept_uris)]

    def _build_expanded_query(self, query: str, neighbour_uris: list[str]) -> str:
        """Append neighbour concept labels to the original query."""
        extra = [
            self.uri_to_label[u]
            for u in neighbour_uris
            if u in self.uri_to_label
        ]
        if not extra:
            return query
        expanded = query + " " + " ".join(extra)
        log.debug("Expanded query: %s", expanded)
        return expanded

    def _kg_boost_score(
        self,
        chunk_text: str,
        all_concept_labels: list[str],
    ) -> float:
        """
        Return a small additive boost if the chunk contains any of the
        expanded concept labels — rewards chunks that are topically central.
        """
        if not all_concept_labels:
            return 0.0
        text_lower = chunk_text.lower()
        hits = sum(1 for lbl in all_concept_labels if lbl.lower() in text_lower)
        # Normalise: max boost = kg_boost_weight, proportional to coverage
        return self.kg_boost_weight * min(hits / max(len(all_concept_labels), 1), 1.0)

    # ── Public API ─────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[KGHit]:
        """
        Graph-enhanced top-K retrieval.

        Returns KGHit objects sorted by combined_score descending.
        Each hit carries:
          - score          : cosine similarity from FAISS
          - kg_boost       : additive bonus from KG match
          - matched_concepts: human-readable names of matched concepts
        """
        # 1. Extract concepts from query
        query_concept_uris = self._extract_query_concepts(query)
        query_concept_labels = [
            self.uri_to_label.get(u, u) for u in query_concept_uris
        ]

        # 2. KG expansion
        neighbour_uris = self._expand_via_kg(query_concept_uris)
        neighbour_labels = [
            self.uri_to_label.get(u, u) for u in neighbour_uris
        ]
        all_labels = query_concept_labels + neighbour_labels

        if all_labels:
            log.info(
                "Query concepts: %s | KG neighbours: %s",
                query_concept_labels,
                neighbour_labels[:5],
            )

        # 3. Vector search over expanded query
        expanded_query = self._build_expanded_query(query, neighbour_uris)
        # Fetch extra candidates to allow for re-ranking
        base_hits = self.base.search(expanded_query, k=min(k * 2, 20))

        # 4. Compute KG boost and build KGHit list
        kg_hits: list[KGHit] = []
        seen_ids: set[str] = set()
        for hit in base_hits:
            chunk_id = hit.chunk.get("id", "")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)

            boost = self._kg_boost_score(hit.chunk.get("text", ""), all_labels)
            kg_hits.append(KGHit(
                score=hit.score,
                chunk=hit.chunk,
                kg_boost=boost,
                matched_concepts=all_labels,
            ))

        # 5. Re-rank by combined score and return top-K
        kg_hits.sort(key=lambda h: h.combined_score, reverse=True)
        return kg_hits[:k]

    def explain(self, query: str) -> dict:
        """
        Return a diagnostic dict showing what the KG expanded the query to.
        Useful for the Gradio UI's "KG Explain" panel.
        """
        concept_uris  = self._extract_query_concepts(query)
        neighbour_uris = self._expand_via_kg(concept_uris)
        return {
            "query": query,
            "detected_concepts": [self.uri_to_label.get(u, u) for u in concept_uris],
            "kg_neighbours": [self.uri_to_label.get(u, u) for u in neighbour_uris],
            "expanded_query": self._build_expanded_query(query, neighbour_uris),
        }


# ─────────────────────────────────────────────────────────────
#  CLI test
# ─────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    parser = argparse.ArgumentParser(description="Test the KG-enhanced retriever.")
    parser.add_argument("query", nargs="?",
                        default="how does a binary search tree work")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    retriever = KGRetriever()

    print("\n── KG Expansion ─────────────────────────────────────")
    explanation = retriever.explain(args.query)
    print(f"  Query            : {explanation['query']}")
    print(f"  Detected concepts: {explanation['detected_concepts']}")
    print(f"  KG neighbours    : {explanation['kg_neighbours']}")
    print(f"  Expanded query   : {explanation['expanded_query']}")

    print(f"\n── Top {args.k} Results ──────────────────────────────────")
    hits = retriever.search(args.query, k=args.k)
    for i, hit in enumerate(hits, 1):
        print(f"\n[{i}] score={hit.score:.3f}  boost={hit.kg_boost:.3f}  "
              f"combined={hit.combined_score:.3f}")
        print(f"    source : {hit.source_label}")
        print(f"    preview: {hit.chunk.get('text', '')[:120].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
