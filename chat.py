"""Multi-turn RAG chat CLI backed by a local Ollama LLM.

Each turn retrieves top-K chunks for the raw question, injects them into
the current user message (not into history), calls Ollama /api/chat with
streaming enabled, and stores only the raw question and assistant reply
in history so that retrieved context does not pollute future prompts.

Slash commands at the prompt:
  /help        show commands
  /exit /quit  leave
  /reset       clear chat history
  /k N         change retrieval top-K
  /sources     toggle showing sources after each answer
  /history     dump current history
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import requests
from dotenv import load_dotenv

from retriever import Hit, Retriever


SYSTEM_PROMPT = """You are a study assistant for CMPSC 132 (Python, data structures, and algorithms) at Penn State.

Rules:
- Answer the student's question using ONLY the course material provided in the user message under "Course material".
- If the answer is not in the provided context, say so plainly ("I don't have that in the course material") — do not guess.
- EVERY factual claim (definitions, mechanisms, code behavior, complexity, examples drawn from the material) must end with a bracketed citation like [1] or [2, 3], matching the numbered sources the user provided. Uncited factual claims are not allowed.
- A single sentence may carry multiple citations when it draws on multiple sources.
- Brief connective prose ("Here's how it works:", "For example:") does not need a citation, but the claim that follows does.
- Be concise. Use fenced code blocks for Python code. Prefer examples from the course material when explaining."""


def render_user_turn(question: str, hits: list[Hit]) -> str:
    """Format the outgoing user message: context block then question."""
    if not hits:
        return f"Question: {question}\n\n(No course material matched.)"
    lines = ["Course material (top matches from the course corpus):\n"]
    for i, hit in enumerate(hits, start=1):
        lines.append(f"[{i}] {hit.source_label}")
        lines.append(hit.chunk["text"].strip())
        lines.append("")
    lines.append("---\n")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


def stream_ollama(
    base_url: str,
    model: str,
    messages: list[dict],
    num_ctx: int = 8192,
) -> Iterator[str]:
    """Stream assistant message content from Ollama's /api/chat endpoint.

    num_ctx overrides Ollama's default 4096-token context window so the
    system prompt + retrieved chunks + recent history fit comfortably.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"num_ctx": num_ctx},
    }
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            if r.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned 404. Did you `ollama pull {model}`?"
                )
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                piece = (obj.get("message") or {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break
    except requests.ConnectionError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {base_url}. Is `ollama serve` running?"
        ) from e


@dataclass
class ChatState:
    """Runtime state for the chat REPL."""

    retriever: Retriever
    ollama_url: str
    ollama_model: str
    top_k: int = 5
    max_history_turns: int = 6
    num_ctx: int = 8192
    show_sources: bool = True
    history: list[dict] = field(default_factory=list)

    def trim_history(self) -> None:
        max_msgs = self.max_history_turns * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def build_messages(self, current_user_content: str) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history,
            {"role": "user", "content": current_user_content},
        ]


HELP_TEXT = """\
Commands:
  /help                 show this help
  /exit  /quit          leave
  /reset                clear chat history
  /k N                  set retrieval top-K (currently {k})
  /sources              toggle showing sources after each answer (currently {src})
  /history              print current history

Anything else is treated as a question. Ctrl-D or /exit to leave."""


def handle_command(state: ChatState, line: str) -> bool:
    """Dispatch a slash command. Return False to exit the REPL."""
    parts = line.strip().split()
    cmd = parts[0]

    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        print(HELP_TEXT.format(k=state.top_k, src=state.show_sources))
    elif cmd == "/reset":
        state.history.clear()
        print("[history cleared]")
    elif cmd == "/k":
        if len(parts) != 2 or not parts[1].isdigit():
            print("Usage: /k N")
        else:
            state.top_k = int(parts[1])
            print(f"[top-K set to {state.top_k}]")
    elif cmd == "/sources":
        state.show_sources = not state.show_sources
        print(f"[sources display: {'on' if state.show_sources else 'off'}]")
    elif cmd == "/history":
        if not state.history:
            print("[empty]")
        else:
            for msg in state.history:
                role = msg["role"]
                text = msg["content"]
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"  {role:>9}: {text}")
    else:
        print(f"Unknown command: {cmd}. Type /help.")
    return True


def load_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def main() -> None:
    load_dotenv()
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    top_k = load_env_int("TOP_K", 5)
    max_history_turns = load_env_int("MAX_HISTORY_TURNS", 6)
    num_ctx = load_env_int("OLLAMA_NUM_CTX", 8192)

    index_dir = Path("index").resolve()
    retriever = Retriever(index_dir)
    state = ChatState(
        retriever=retriever,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        top_k=top_k,
        max_history_turns=max_history_turns,
        num_ctx=num_ctx,
    )

    print("=" * 60)
    print("CMPSC 132 RAG chat")
    print(f"  Index:  {len(retriever.chunks)} chunks "
          f"(model {retriever.meta['model']})")
    print(f"  LLM:    {state.ollama_model} @ {state.ollama_url}")
    print(f"  Top-K:  {state.top_k}    History: up to "
          f"{state.max_history_turns} turns    num_ctx: {state.num_ctx}")
    print("  Type /help for commands, /exit to leave.")
    print("=" * 60)

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return

        if not question:
            continue
        if question.startswith("/"):
            if not handle_command(state, question):
                return
            continue

        hits = retriever.search(question, k=state.top_k)
        current_user_content = render_user_turn(question, hits)
        messages = state.build_messages(current_user_content)

        print()
        try:
            answer_parts: list[str] = []
            for piece in stream_ollama(
                state.ollama_url,
                state.ollama_model,
                messages,
                num_ctx=state.num_ctx,
            ):
                print(piece, end="", flush=True)
                answer_parts.append(piece)
            print()
        except RuntimeError as e:
            print(f"\n[error] {e}")
            continue

        answer = "".join(answer_parts).strip()
        if not answer:
            print("[empty response from Ollama]")
            continue

        if state.show_sources and hits:
            print("\nSources:")
            for i, hit in enumerate(hits, start=1):
                print(f"  [{i}] {hit.source_label}  (score={hit.score:.3f})")

        state.history.append({"role": "user", "content": question})
        state.history.append({"role": "assistant", "content": answer})
        state.trim_history()


if __name__ == "__main__":
    main()
