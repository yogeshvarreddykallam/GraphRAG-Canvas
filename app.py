"""Gradio web UI for GraphRAG-Canvas.

Extends the original Canvas-RAG app with a Knowledge Graph retrieval layer:
  - KGRetriever expands queries via the OWL ontology + NetworkX graph
  - A dedicated "KG Expansion" panel shows which concepts were detected and
    which neighbours the graph added to the query before FAISS search
  - Sources panel now shows both vector score and KG boost per result

Launch:
    python app.py
    # then open http://127.0.0.1:7860

Environment variables (optional):
    OLLAMA_URL      default: http://localhost:11434
    OLLAMA_MODEL    default: llama3.1:8b
    TOP_K           default: 5
    MAX_HISTORY_TURNS default: 6
    OLLAMA_NUM_CTX  default: 8192
    USE_KG          set to "0" to disable KG layer (vector-only fallback)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from chat import SYSTEM_PROMPT, render_user_turn, stream_ollama
from kg_retriever import KGHit, KGRetriever

load_dotenv()

OLLAMA_URL   = os.environ.get("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
USE_KG       = os.environ.get("USE_KG", "1") != "0"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


DEFAULT_TOP_K     = _env_int("TOP_K", 5)
MAX_HISTORY_TURNS = _env_int("MAX_HISTORY_TURNS", 6)
NUM_CTX           = _env_int("OLLAMA_NUM_CTX", 8192)

INDEX_DIR    = Path("index").resolve()
KG_DIR       = Path("kg").resolve()
ONTOLOGY     = Path("ontology/canvas_kg.ttl").resolve()

RETRIEVER = KGRetriever(
    index_dir=INDEX_DIR,
    kg_dir=KG_DIR,
    ontology_path=ONTOLOGY,
)


def _format_sources_md(hits: list[KGHit]) -> str:
    """Render a markdown sources block with vector score + KG boost."""
    if not hits:
        return "_No sources retrieved for the last question._"
    lines = ["### Sources", ""]
    for i, hit in enumerate(hits, start=1):
        boost_str = f" · KG boost +{hit.kg_boost:.3f}" if hit.kg_boost > 0 else ""
        lines.append(
            f"**[{i}]** `{hit.source_label}` &nbsp;&nbsp; "
            f"_vec {hit.score:.3f}{boost_str} → combined **{hit.combined_score:.3f}**_"
        )
    return "\n".join(lines)


def _format_kg_expand_md(explanation: dict) -> str:
    """Render the KG expansion panel markdown."""
    concepts   = explanation.get("detected_concepts", [])
    neighbours = explanation.get("kg_neighbours", [])
    if not concepts and not neighbours:
        return "_No course concepts detected in your query._"
    lines = ["### Knowledge Graph Expansion", ""]
    if concepts:
        lines.append(f"**Detected concepts:** {', '.join(f'`{c}`' for c in concepts)}")
    if neighbours:
        lines.append(f"**KG neighbours added:** {', '.join(f'`{n}`' for n in neighbours[:8])}")
    lines.append(f"\n_Expanded query used for retrieval:_ _{explanation.get('expanded_query', '')[:120]}_")
    return "\n".join(lines)


def _trim_history(messages: list[dict]) -> list[dict]:
    """Keep at most MAX_HISTORY_TURNS user/assistant pairs."""
    max_msgs = MAX_HISTORY_TURNS * 2
    if len(messages) > max_msgs:
        return messages[-max_msgs:]
    return messages


def respond(
    user_message: str,
    chat_messages: list[dict],
    top_k: int,
    show_sources: bool,
    show_kg: bool,
):
    """Stream a GraphRAG response. Yields (textbox, chatbot, sources_md, kg_md)."""
    user_message = (user_message or "").strip()
    if not user_message:
        yield "", chat_messages, gr.update(), gr.update()
        return

    chat_messages = list(chat_messages) + [{"role": "user", "content": user_message}]
    yield "", chat_messages, gr.update(), gr.update()

    # ── KG expansion explanation ───────────────────────────────
    kg_md = ""
    if show_kg:
        explanation = RETRIEVER.explain(user_message)
        kg_md = _format_kg_expand_md(explanation)

    # ── Graph-enhanced retrieval ───────────────────────────────
    hits = RETRIEVER.search(user_message, k=int(top_k))

    # render_user_turn expects Hit-like objects; KGHit has .score and .chunk
    sources_md = _format_sources_md(hits) if show_sources else ""

    past = _trim_history(chat_messages[:-1])
    current_user_content = render_user_turn(user_message, hits)
    llm_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *past,
        {"role": "user", "content": current_user_content},
    ]

    chat_messages = chat_messages + [{"role": "assistant", "content": ""}]
    try:
        answer = ""
        for piece in stream_ollama(OLLAMA_URL, OLLAMA_MODEL, llm_messages, num_ctx=NUM_CTX):
            answer += piece
            chat_messages[-1]["content"] = answer
            yield "", chat_messages, sources_md, kg_md
    except RuntimeError as e:
        chat_messages[-1]["content"] = f"**[error]** {e}"
        yield "", chat_messages, sources_md, kg_md
        return

    if not chat_messages[-1]["content"].strip():
        chat_messages[-1]["content"] = "_[empty response from Ollama]_"
        yield "", chat_messages, sources_md, kg_md


def clear_chat():
    return [], "", ""


HEADER_MD = f"""
# GraphRAG-Canvas

Knowledge-Graph-enhanced retrieval over **{len(RETRIEVER.base.chunks)} chunks**
(`{RETRIEVER.base.meta['model']}`) &middot; LLM **{OLLAMA_MODEL}**
&middot; OWL ontology · {RETRIEVER.nx_graph.number_of_nodes() if RETRIEVER.nx_graph else 0} concept nodes
· {RETRIEVER.nx_graph.number_of_edges() if RETRIEVER.nx_graph else 0} KG edges
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="GraphRAG-Canvas", theme=gr.themes.Soft()) as demo:
        gr.Markdown(HEADER_MD)

        with gr.Row():
            # ── Main chat column ───────────────────────────────
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=480, label="Chat")
                with gr.Row():
                    textbox = gr.Textbox(
                        placeholder="Ask anything about CMPSC 132 — try 'explain binary search trees'",
                        show_label=False,
                        scale=8,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                sources_md = gr.Markdown(
                    value="_Retrieved sources will appear here after your first question._",
                    label="Sources",
                )

                kg_expand_md = gr.Markdown(
                    value="_KG expansion details will appear here._",
                    label="Knowledge Graph Expansion",
                )

            # ── Controls column ────────────────────────────────
            with gr.Column(scale=1, min_width=230):
                gr.Markdown("### Controls")
                top_k = gr.Slider(
                    label="Top-K chunks",
                    minimum=1, maximum=15, step=1, value=DEFAULT_TOP_K,
                )
                show_sources = gr.Checkbox(label="Show sources panel", value=True)
                show_kg      = gr.Checkbox(label="Show KG expansion panel", value=True)
                clear_btn    = gr.Button("Clear chat", variant="secondary")

                gr.Markdown("""
---
**How GraphRAG works**

1. Your query is parsed for known CS concepts
2. The KG expands them to related/prerequisite nodes
3. The expanded query hits FAISS for vector retrieval
4. Results are re-ranked with a KG relevance boost
5. Top-K chunks are injected into the LLM prompt

**Try asking:**
- "What is a linked list?"
- "How do stacks and queues differ?"
- "Explain BFS vs DFS"
- "What is dynamic programming?"
""")

        submit_inputs  = [textbox, chatbot, top_k, show_sources, show_kg]
        submit_outputs = [textbox, chatbot, sources_md, kg_expand_md]

        textbox.submit(respond, inputs=submit_inputs, outputs=submit_outputs)
        send_btn.click(respond, inputs=submit_inputs, outputs=submit_outputs)
        clear_btn.click(clear_chat, inputs=None, outputs=[chatbot, sources_md, kg_expand_md])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
    )
