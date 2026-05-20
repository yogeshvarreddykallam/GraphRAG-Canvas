"""
Canvas course scraper for the RAG project (Part 0).

Walks every module of a Canvas course via the REST API and saves each item
locally in a form the downstream RAG pipeline can chunk and embed:

  * Pages        -> HTML rendered to PDF (weasyprint)
  * Files        -> downloaded verbatim (PDFs stay PDFs, .py stays .py)
  * External URLs-> recorded in module_index.md (no external-site scraping)
  * Assignments  -> description rendered to PDF + JSON sidecar with dates/points
  * Quizzes      -> description rendered to PDF + JSON sidecar
  * Discussions  -> message rendered to PDF + JSON sidecar
  * SubHeaders   -> section dividers in module_index.md

Layout:
  canvas_data/
    00_module_name/
      01_item_title.pdf
      02_item_title.py        (raw downloads keep original extension)
      assignments/
        03_lab_0.pdf
        03_lab_0.json
      module_index.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from canvasapi import Canvas
from canvasapi.exceptions import CanvasException, ResourceDoesNotExist
from dotenv import load_dotenv
from tqdm import tqdm
from weasyprint import CSS, HTML

# ---------------------------------------------------------------------------
# Config + setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("canvas-scrape")

_slug_re = re.compile(r"[^a-zA-Z0-9]+")


def slugify(text: str, max_len: int = 80) -> str:
    s = _slug_re.sub("_", text or "").strip("_").lower()
    return (s[:max_len] or "untitled").rstrip("_")


PAGE_CSS = CSS(
    string="""
    @page { size: Letter; margin: 0.75in; }
    body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
           font-size: 11pt; line-height: 1.45; color: #111; }
    h1, h2, h3 { color: #1a2a44; }
    pre, code { font-family: ui-monospace, "SF Mono", Menlo, monospace;
                font-size: 10pt; background: #f4f4f4; }
    pre { padding: 8px; border-radius: 4px; white-space: pre-wrap; }
    img { max-width: 100%; height: auto; }
    table { border-collapse: collapse; }
    th, td { border: 1px solid #bbb; padding: 4px 8px; }
    a { color: #1a56b0; }
    """
)


@dataclass
class Config:
    canvas_domain: str
    canvas_token: str
    course_id: int
    output_dir: Path


_URL_FETCHER: Optional[Callable[..., dict]] = None


def make_url_fetcher(cfg: Config) -> Callable[..., dict]:
    """Build a requests-backed URL fetcher for WeasyPrint.

    Uses the certifi CA bundle so HTTPS works on macOS and forwards the
    Canvas bearer token for same-host URLs so authenticated media embeds.
    """
    canvas_host = urllib.parse.urlparse(cfg.canvas_domain).netloc

    def fetcher(url: str, timeout: int = 30, ssl_context=None) -> dict:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            try:
                from weasyprint import default_url_fetcher  # type: ignore
            except ImportError:
                from weasyprint.urls import default_url_fetcher  # type: ignore
            return default_url_fetcher(url, timeout=timeout,
                                       ssl_context=ssl_context)

        headers: dict[str, str] = {}
        if parsed.netloc == canvas_host:
            headers["Authorization"] = f"Bearer {cfg.canvas_token}"

        resp = requests.get(url, headers=headers, timeout=timeout,
                            allow_redirects=True)
        resp.raise_for_status()
        return {
            "mime_type": (resp.headers.get("content-type") or "").split(";")[0].strip(),
            "string": resp.content,
            "redirected_url": resp.url,
            "filename": parsed.path.rsplit("/", 1)[-1] or "file",
        }

    return fetcher


def load_config() -> Config:
    load_dotenv()
    domain = os.environ.get("CANVAS_DOMAIN", "").rstrip("/")
    token = os.environ.get("CANVAS_TOKEN", "")
    course_id = os.environ.get("COURSE_ID", "")
    output_dir = os.environ.get("OUTPUT_DIR", "canvas_data")

    missing = [n for n, v in [("CANVAS_DOMAIN", domain), ("CANVAS_TOKEN", token),
                              ("COURSE_ID", course_id)] if not v]
    if missing:
        log.error("Missing env vars: %s. Copy .env.example to .env and fill them in.",
                  ", ".join(missing))
        sys.exit(1)
    try:
        course_id_int = int(course_id)
    except ValueError:
        log.error("COURSE_ID must be an integer (got %r).", course_id)
        sys.exit(1)

    return Config(
        canvas_domain=domain,
        canvas_token=token,
        course_id=course_id_int,
        output_dir=Path(output_dir).expanduser().resolve(),
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def html_to_pdf(html_body: str, title: str, out_path: Path) -> None:
    """Render a Canvas HTML blob to PDF using weasyprint."""
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_html_escape(title)}</title></head>
<body>
<h1>{_html_escape(title)}</h1>
{html_body or "<p><em>(no content)</em></p>"}
</body></html>"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"string": doc, "base_url": "."}
    if _URL_FETCHER is not None:
        kwargs["url_fetcher"] = _URL_FETCHER
    HTML(**kwargs).write_pdf(str(out_path), stylesheets=[PAGE_CSS])


def _html_escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def download_file(url: str, token: str, out_path: Path) -> None:
    """Stream-download a Canvas file. Canvas file URLs need the bearer token."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    f.write(chunk)


def write_sidecar(obj: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-item handlers
# ---------------------------------------------------------------------------

def handle_page(cfg: Config, course, item, module_dir: Path, prefix: str) -> dict:
    """Fetch Page body -> PDF."""
    page_url = getattr(item, "page_url", None)
    if not page_url:
        return {"type": "Page", "title": item.title, "status": "no page_url"}
    page = course.get_page(page_url)
    body_html = getattr(page, "body", "") or ""
    out = module_dir / f"{prefix}_{slugify(item.title)}.pdf"
    html_to_pdf(body_html, item.title, out)
    return {"type": "Page", "title": item.title, "path": str(out.relative_to(module_dir.parent))}


def handle_file(cfg: Config, course, item, module_dir: Path, prefix: str) -> dict:
    """Download a File item. Preserves original extension."""
    content_id = getattr(item, "content_id", None)
    if not content_id:
        return {"type": "File", "title": item.title, "status": "no content_id"}
    f = course.get_file(content_id)
    orig_name = getattr(f, "display_name", None) or getattr(f, "filename", None) or item.title
    p = Path(orig_name)
    stem = slugify(p.stem)
    ext = p.suffix.lower() or ""
    out = module_dir / f"{prefix}_{stem}{ext}"
    download_file(f.url, cfg.canvas_token, out)
    return {"type": "File", "title": item.title, "path": str(out.relative_to(module_dir.parent))}


def handle_external_url(cfg: Config, course, item, module_dir: Path, prefix: str) -> dict:
    """Record an external URL in metadata; do not scrape external sites."""
    return {
        "type": "ExternalUrl",
        "title": item.title,
        "url": getattr(item, "external_url", ""),
    }


def handle_assignment(cfg: Config, course, item, module_dir: Path, prefix: str) -> dict:
    content_id = getattr(item, "content_id", None)
    if not content_id:
        return {"type": "Assignment", "title": item.title, "status": "no content_id"}
    a = course.get_assignment(content_id)
    sub_dir = module_dir / "assignments"
    pdf_out = sub_dir / f"{prefix}_{slugify(item.title)}.pdf"
    html_to_pdf(getattr(a, "description", "") or "", item.title, pdf_out)
    meta = {
        "type": "Assignment",
        "id": a.id,
        "name": getattr(a, "name", item.title),
        "due_at": getattr(a, "due_at", None),
        "points_possible": getattr(a, "points_possible", None),
        "submission_types": getattr(a, "submission_types", None),
        "html_url": getattr(a, "html_url", None),
    }
    write_sidecar(meta, pdf_out.with_suffix(".json"))
    return {"type": "Assignment", "title": item.title,
            "path": str(pdf_out.relative_to(module_dir.parent))}


def handle_quiz(cfg: Config, course, item, module_dir: Path, prefix: str) -> dict:
    content_id = getattr(item, "content_id", None)
    if not content_id:
        return {"type": "Quiz", "title": item.title, "status": "no content_id"}
    try:
        q = course.get_quiz(content_id)
    except ResourceDoesNotExist:
        return {"type": "Quiz", "title": item.title, "status": "not accessible"}
    sub_dir = module_dir / "quizzes"
    pdf_out = sub_dir / f"{prefix}_{slugify(item.title)}.pdf"
    html_to_pdf(getattr(q, "description", "") or "", item.title, pdf_out)
    meta = {
        "type": "Quiz",
        "id": q.id,
        "title": getattr(q, "title", item.title),
        "due_at": getattr(q, "due_at", None),
        "points_possible": getattr(q, "points_possible", None),
        "question_count": getattr(q, "question_count", None),
        "html_url": getattr(q, "html_url", None),
    }
    write_sidecar(meta, pdf_out.with_suffix(".json"))
    return {"type": "Quiz", "title": item.title,
            "path": str(pdf_out.relative_to(module_dir.parent))}


def handle_discussion(cfg: Config, course, item, module_dir: Path, prefix: str) -> dict:
    content_id = getattr(item, "content_id", None)
    if not content_id:
        return {"type": "Discussion", "title": item.title, "status": "no content_id"}
    try:
        d = course.get_discussion_topic(content_id)
    except ResourceDoesNotExist:
        return {"type": "Discussion", "title": item.title, "status": "not accessible"}
    sub_dir = module_dir / "discussions"
    pdf_out = sub_dir / f"{prefix}_{slugify(item.title)}.pdf"
    html_to_pdf(getattr(d, "message", "") or "", item.title, pdf_out)
    meta = {
        "type": "Discussion",
        "id": d.id,
        "title": getattr(d, "title", item.title),
        "posted_at": getattr(d, "posted_at", None),
        "html_url": getattr(d, "html_url", None),
    }
    write_sidecar(meta, pdf_out.with_suffix(".json"))
    return {"type": "Discussion", "title": item.title,
            "path": str(pdf_out.relative_to(module_dir.parent))}


def handle_subheader(cfg: Config, course, item, module_dir: Path, prefix: str) -> dict:
    return {"type": "SubHeader", "title": item.title}


HANDLERS = {
    "Page": handle_page,
    "File": handle_file,
    "ExternalUrl": handle_external_url,
    "ExternalTool": handle_external_url,
    "Assignment": handle_assignment,
    "Quiz": handle_quiz,
    "Discussion": handle_discussion,
    "SubHeader": handle_subheader,
}


# ---------------------------------------------------------------------------
# Module walker
# ---------------------------------------------------------------------------

def process_module(cfg: Config, course, module, module_idx: int,
                   skip_existing: bool) -> None:
    module_dir = cfg.output_dir / f"{module_idx:02d}_{slugify(module.name)}"
    module_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    items = list(module.get_module_items())

    for i, item in enumerate(tqdm(items, desc=f"  {module.name[:50]}", leave=False), start=1):
        prefix = f"{i:02d}"
        itype = getattr(item, "type", "Unknown")
        handler = HANDLERS.get(itype)
        rec: dict[str, Any] = {"position": i, "type": itype, "title": item.title}
        if handler is None:
            rec["status"] = "unknown type"
            records.append(rec)
            continue
        try:
            rec.update(handler(cfg, course, item, module_dir, prefix))
        except CanvasException as e:
            log.warning("Canvas error on %s/%s: %s", module.name, item.title, e)
            rec["status"] = f"canvas error: {e}"
        except Exception as e:
            log.exception("Failed on %s/%s: %s", module.name, item.title, e)
            rec["status"] = f"error: {e}"
        records.append(rec)

    write_module_index(module_dir, module.name, records)


def write_module_index(module_dir: Path, module_name: str, records: list[dict]) -> None:
    lines = [f"# {module_name}", ""]
    for r in records:
        bullet = f"- **[{r['type']}]** {r['title']}"
        if "path" in r:
            bullet += f" — `{Path(r['path']).name}`"
        if "url" in r:
            bullet += f" — <{r['url']}>"
        if "status" in r:
            bullet += f" _({r['status']})_"
        lines.append(bullet)
    (module_dir / "module_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (module_dir / "module_items.json").write_text(
        json.dumps(records, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape a Canvas course into a RAG-friendly tree.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N modules (smoke test).")
    parser.add_argument("--module-id", type=int, default=None,
                        help="Only process this specific module ID.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip items whose output file already exists.")
    args = parser.parse_args()

    cfg = load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", cfg.output_dir)

    global _URL_FETCHER
    _URL_FETCHER = make_url_fetcher(cfg)

    canvas = Canvas(cfg.canvas_domain, cfg.canvas_token)
    course = canvas.get_course(cfg.course_id)
    log.info("Course: %s (%s)", course.name, course.course_code if hasattr(course, "course_code") else "")

    modules = list(course.get_modules())
    if args.module_id:
        modules = [m for m in modules if m.id == args.module_id]
    if args.limit:
        modules = modules[: args.limit]

    log.info("Processing %d module(s).", len(modules))
    for idx, module in enumerate(modules):
        log.info("[%d/%d] %s", idx + 1, len(modules), module.name)
        process_module(cfg, course, module, idx, args.skip_existing)

    log.info("Done. %d module(s) written to %s", len(modules), cfg.output_dir)


if __name__ == "__main__":
    main()
