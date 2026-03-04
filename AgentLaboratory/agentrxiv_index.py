"""Low-RAM AgentRxiv local index (no Flask server, no local embedding model).

Previous design in this repo started a Flask server and loaded a SentenceTransformer
model globally. That is convenient but memory-heavy for multi-agent runs.

This module replaces the server+SentenceTransformer flow with:
- persistent SQLite database (metadata + summaries + FTS index)
- optional remote embeddings (OpenAI) for reranking (no local model in RAM)
- full paper text stored as disk blobs (txt extracted from pdf)

Search is "hybrid-ready":
- primary recall = SQLite FTS5 (bm25 ranking)
- optional semantic rerank = embeddings (if configured)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pypdf import PdfReader  # already a dependency in requirements

from inference import query_model
from rag_memory import OpenAIEmbedder, SQLiteRAGStore


def _now_ts() -> int:
    return int(time.time())


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_pdf_text(pdf_path: Path, *, max_pages: Optional[int] = None) -> str:
    # Read incrementally page-by-page; still returns a string, but avoids holding
    # unnecessary intermediate objects.
    reader = PdfReader(str(pdf_path))
    out = []
    pages = reader.pages if max_pages is None else reader.pages[: max_pages]
    for page in pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            out.append(t)
    return "\n".join(out)


@dataclass
class AgentRxivResult:
    paper_id: int
    filename: str
    summary: str
    score: Optional[float] = None


class AgentRxivIndex:
    """Persistent local index for PDFs in an uploads folder."""

    def __init__(
        self,
        *,
        uploads_dir: Path,
        db_path: Path,
        blob_dir: Path,
        openai_api_key: Optional[str] = None,
        summary_llm: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        enable_embeddings: bool = False,
        max_index_chars: int = 40000,
    ):
        self.uploads_dir = Path(uploads_dir)
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir)
        self.summary_llm = summary_llm
        self.max_index_chars = int(max_index_chars)

        _ensure_dir(self.uploads_dir)
        _ensure_dir(self.db_path.parent)
        _ensure_dir(self.blob_dir)

        self._embedder = None
        if enable_embeddings and openai_api_key:
            try:
                self._embedder = OpenAIEmbedder(api_key=openai_api_key, model=embed_model)
            except Exception:
                self._embedder = None

        # Use shared RAG store implementation so search is consistent
        self._store = SQLiteRAGStore(
            db_path=self.db_path,
            blob_dir=self.blob_dir,
            embedder=self._embedder,
            max_inline_chars=20000,
            max_snippet_chars=2500,
        )
        self._ns = "agentrxiv_papers"

    def sync_from_uploads(self) -> int:
        """Ingest new PDFs from uploads_dir. Returns number of newly ingested PDFs."""
        n_new = 0
        for pdf in sorted(self.uploads_dir.glob("*.pdf")):
            try:
                sha = _sha256_file(pdf)
            except Exception:
                continue
            key = sha  # stable key prevents duplicates even if filename changes
            # Check if already exists
            existing = self._store.search(namespace=self._ns, query=f'key:"{key}"', k=1, candidate_k=1, use_embeddings=False)
            # FTS query above might be empty if tokenization differs; do a direct DB check:
            if self._exists_by_key(key):
                continue

            text = _read_pdf_text(pdf)
            if not text:
                continue
            # Store full extracted text on disk blob (low RAM afterwards); SQLite indexes only snippet/summary.
            meta = {"filename": pdf.name, "sha256": sha, "source_path": str(pdf)}
            self._store.add(
                namespace=self._ns,
                key=key,
                text=text,
                summary="",  # summary generated lazily
                meta=meta,
                created_ts=_now_ts(),
                embed_text_for_rerank=pdf.name,
            )
            n_new += 1
        return n_new

    def _exists_by_key(self, key: str) -> bool:
        con = sqlite3.connect(str(self.db_path))
        try:
            row = con.execute("SELECT 1 FROM rag_items WHERE namespace=? AND key=? LIMIT 1", (self._ns, key)).fetchone()
            return row is not None
        finally:
            con.close()

    def _ensure_summary(self, item_id: int, key: str, filename: str, openai_api_key: Optional[str]) -> str:
        # Load current summary
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        try:
            row = con.execute("SELECT summary, text_inline, text_path, meta_json FROM rag_items WHERE id=?", (item_id,)).fetchone()
            if not row:
                return ""
            cur_summary = (row["summary"] or "").strip()
            if cur_summary:
                return cur_summary

            # Read only a chunk for summarization
            text = row["text_inline"] or ""
            if not text:
                tp = row["text_path"] or ""
                if tp and Path(tp).exists():
                    text = Path(tp).read_text(encoding="utf-8", errors="ignore")
            text = (text or "")[:20000]

            if not text.strip():
                return ""

            summ = query_model(
                model_str=self.summary_llm,
                system_prompt="Please provide a 5 sentence summary of this paper.",
                prompt=text,
                openai_api_key=openai_api_key,
            ).strip()

            # Update in DB and FTS
            con.execute("UPDATE rag_items SET summary=? WHERE id=?", (summ, item_id))
            con.execute("UPDATE rag_items_fts SET summary=? WHERE rowid=?", (summ, item_id))
            con.commit()
            return summ
        finally:
            con.close()

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        openai_api_key: Optional[str] = None,
        ensure_summaries: bool = True,
    ) -> List[AgentRxivResult]:
        self.sync_from_uploads()
        items = self._store.search(namespace=self._ns, query=query, k=k, candidate_k=max(25, k * 5), use_embeddings=True)
        out: List[AgentRxivResult] = []
        for it in items:
            filename = (it.meta or {}).get("filename", it.key)
            summ = it.summary
            if ensure_summaries and not summ:
                summ = self._ensure_summary(it.id, it.key, filename, openai_api_key)
            out.append(AgentRxivResult(paper_id=it.id, filename=filename, summary=summ, score=it.score))
        return out

    def get_full_text(self, paper_id: int) -> str:
        con = sqlite3.connect(str(self.db_path))
        con.row_factory = sqlite3.Row
        try:
            row = con.execute("SELECT text_inline, text_path FROM rag_items WHERE id=?", (paper_id,)).fetchone()
            if not row:
                return "Paper ID not found?"
            if row["text_inline"]:
                return row["text_inline"]
            tp = row["text_path"] or ""
            if tp and Path(tp).exists():
                return Path(tp).read_text(encoding="utf-8", errors="ignore")
            return ""
        finally:
            con.close()
