"""kg_builder.py — Knowledge Graph construction for GraphRAG-Canvas.

Loads all indexed chunks, extracts CS concepts using a spaCy PhraseMatcher
seeded from the OWL ontology, builds RDF triples (co-occurrence + prerequisite
edges) with RDFLib, and persists:
    kg/canvas_kg_populated.ttl   — full RDF graph (Turtle)
    kg/canvas_kg.graphml         — NetworkX export for fast runtime traversal

Usage:
    python kg_builder.py                        # uses default paths
    python kg_builder.py --chunks index/chunks.jsonl --ontology ontology/canvas_kg.ttl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import spacy
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD
from rdflib.namespace import NamespaceManager
from spacy.matcher import PhraseMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kg-builder")

# ─────────────────────────────────────────────────────────────
#  Namespace
# ─────────────────────────────────────────────────────────────

CKG = Namespace("http://yogeshvarreddykallam.github.io/canvas-kg#")

RELATED_TO     = CKG.relatedTo
HAS_PREREQ     = CKG.hasPrerequisite
MENTIONED_IN   = CKG.mentionedIn
BELONGS_MODULE = CKG.belongsToModule
CO_WEIGHT      = CKG.coOccurrenceWeight
CHUNK_CLASS    = CKG.SourceChunk
CHUNK_ID_PROP  = CKG.chunkId
SRC_PATH_PROP  = CKG.sourcePath


# ─────────────────────────────────────────────────────────────
#  Step 1 – Load ontology seed concepts
# ─────────────────────────────────────────────────────────────

def load_ontology(ttl_path: Path) -> tuple[Graph, dict[str, URIRef]]:
    """Parse the Turtle ontology; return (rdflib_graph, label→uri mapping)."""
    g = Graph()
    g.parse(str(ttl_path), format="turtle")

    label_to_uri: dict[str, URIRef] = {}
    for subj, _, label in g.triples((None, RDFS.label, None)):
        label_to_uri[str(label).lower()] = subj

    log.info("Ontology loaded: %d triples, %d labelled concepts",
             len(g), len(label_to_uri))
    return g, label_to_uri


# ─────────────────────────────────────────────────────────────
#  Step 2 – Build spaCy PhraseMatcher from ontology labels
# ─────────────────────────────────────────────────────────────

def build_matcher(nlp, label_to_uri: dict[str, URIRef]) -> PhraseMatcher:
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(label) for label in label_to_uri]
    matcher.add("CONCEPT", patterns)
    log.info("PhraseMatcher built with %d patterns", len(patterns))
    return matcher


# ─────────────────────────────────────────────────────────────
#  Step 3 – Extract concepts from a chunk
# ─────────────────────────────────────────────────────────────

def extract_concepts(
    text: str,
    nlp,
    matcher: PhraseMatcher,
    label_to_uri: dict[str, URIRef],
) -> list[URIRef]:
    doc = nlp(text[:10_000])  # cap to avoid slow processing on huge chunks
    matches = matcher(doc)
    found: list[URIRef] = []
    seen: set[str] = set()
    for _, start, end in matches:
        span_text = doc[start:end].text.lower()
        if span_text not in seen:
            seen.add(span_text)
            if span_text in label_to_uri:
                found.append(label_to_uri[span_text])
    return found


# ─────────────────────────────────────────────────────────────
#  Step 4 – Build RDF triples from chunks
# ─────────────────────────────────────────────────────────────

def populate_graph(
    rdf_graph: Graph,
    chunks: list[dict],
    nlp,
    matcher: PhraseMatcher,
    label_to_uri: dict[str, URIRef],
) -> None:
    """
    For every chunk:
      1. Create a SourceChunk individual
      2. Link any matched concepts → mentionedIn → chunk
      3. For every pair of concepts in the same chunk, add/increment relatedTo weight
    """
    co_occurrence: dict[tuple[URIRef, URIRef], int] = defaultdict(int)

    for chunk in chunks:
        chunk_uri = CKG[f"chunk_{chunk['id'].replace('/', '_').replace('#', '_')}"]
        rdf_graph.add((chunk_uri, RDF.type, CHUNK_CLASS))
        rdf_graph.add((chunk_uri, CHUNK_ID_PROP, Literal(chunk["id"], datatype=XSD.string)))
        rdf_graph.add((chunk_uri, SRC_PATH_PROP, Literal(chunk["source_path"], datatype=XSD.string)))

        concepts = extract_concepts(chunk["text"], nlp, matcher, label_to_uri)

        # Link each concept → mentionedIn → chunk
        for c_uri in concepts:
            rdf_graph.add((c_uri, MENTIONED_IN, chunk_uri))

        # Tag module membership
        module_uri = CKG[f"module_{chunk.get('module', 'unknown').replace(' ', '_')}"]
        for c_uri in concepts:
            rdf_graph.add((c_uri, BELONGS_MODULE, module_uri))

        # Track co-occurrences
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                a, b = concepts[i], concepts[j]
                key = (min(a, b), max(a, b))  # canonical order
                co_occurrence[key] += 1

    # Materialise co-occurrence edges
    for (a, b), weight in co_occurrence.items():
        rdf_graph.add((a, RELATED_TO, b))
        rdf_graph.add((b, RELATED_TO, a))
        # Store weight as a reified literal on the subject
        rdf_graph.add((a, CO_WEIGHT, Literal(weight, datatype=XSD.integer)))

    log.info(
        "Populated graph: %d triples total, %d co-occurrence edges",
        len(rdf_graph), len(co_occurrence),
    )


# ─────────────────────────────────────────────────────────────
#  Step 5 – Export to NetworkX for fast runtime traversal
# ─────────────────────────────────────────────────────────────

def build_networkx(rdf_graph: Graph, label_to_uri: dict[str, URIRef]) -> nx.Graph:
    """
    Build an undirected weighted NetworkX graph from the RDF data.
    Nodes = concept URIs (labelled with human-readable name).
    Edges = relatedTo or hasPrerequisite (both treated as undirected links here).
    Edge weight = co-occurrence count.
    """
    G = nx.Graph()

    # Add all concept nodes
    uri_to_label = {v: k for k, v in label_to_uri.items()}
    for uri in label_to_uri.values():
        G.add_node(str(uri), label=uri_to_label.get(uri, str(uri)))

    # Add relatedTo edges
    weights: dict[tuple[str, str], int] = defaultdict(int)
    for subj, _, obj in rdf_graph.triples((None, RELATED_TO, None)):
        if str(subj) in {str(u) for u in label_to_uri.values()} and \
           str(obj)  in {str(u) for u in label_to_uri.values()}:
            key = (min(str(subj), str(obj)), max(str(subj), str(obj)))
            weights[key] += 1

    for (a, b), w in weights.items():
        G.add_edge(a, b, weight=w, relation="relatedTo")

    # Add hasPrerequisite edges (directed but include in undirected graph too)
    for subj, _, obj in rdf_graph.triples((None, HAS_PREREQ, None)):
        sa, so = str(subj), str(obj)
        if G.has_node(sa) and G.has_node(so):
            if not G.has_edge(sa, so):
                G.add_edge(sa, so, weight=1, relation="hasPrerequisite")

    log.info("NetworkX graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Canvas Knowledge Graph.")
    parser.add_argument("--chunks",   type=Path, default=Path("index/chunks.jsonl"))
    parser.add_argument("--ontology", type=Path, default=Path("ontology/canvas_kg.ttl"))
    parser.add_argument("--out-dir",  type=Path, default=Path("kg"))
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load chunks ────────────────────────────────────────────
    if not args.chunks.exists():
        log.error("Chunks file not found: %s. Run build_index.py first.", args.chunks)
        return

    with args.chunks.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    log.info("Loaded %d chunks", len(chunks))

    # ── Load ontology ──────────────────────────────────────────
    rdf_graph, label_to_uri = load_ontology(args.ontology)

    # ── Load spaCy ─────────────────────────────────────────────
    try:
        nlp = spacy.load(args.spacy_model, disable=["ner", "parser", "lemmatizer"])
    except OSError:
        log.error(
            "spaCy model '%s' not found. Run: python -m spacy download %s",
            args.spacy_model, args.spacy_model,
        )
        return
    matcher = build_matcher(nlp, label_to_uri)

    # ── Populate RDF graph ─────────────────────────────────────
    populate_graph(rdf_graph, chunks, nlp, matcher, label_to_uri)

    # ── Serialise Turtle ───────────────────────────────────────
    ttl_out = args.out_dir / "canvas_kg_populated.ttl"
    rdf_graph.serialize(destination=str(ttl_out), format="turtle")
    log.info("Saved populated graph → %s", ttl_out)

    # ── Export NetworkX ────────────────────────────────────────
    nx_graph = build_networkx(rdf_graph, label_to_uri)
    gml_out = args.out_dir / "canvas_kg.graphml"
    nx.write_graphml(nx_graph, str(gml_out))
    log.info("Saved NetworkX graph → %s", gml_out)

    log.info("Knowledge graph build complete.")


if __name__ == "__main__":
    main()
