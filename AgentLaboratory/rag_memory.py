"""Disk-backed RAG-style memory for AgentLaboratory (RAM-friendly).

Key goals
---------
- Keep *long-term* memory on disk (SQLite + optional blob files).
- Keep *short-term* working set in RAM only (last N turns + small retrieved set).
- Support keyword retrieval via SQLite FTS5 (fast, low-RAM).
- Optionally add vector reranking using external embedding APIs (no local embedding model in RAM).

This module is dependency-light (stdlib + numpy if available).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import math

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _now_ts() -> int:
    return int(time.time())


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return "{}"


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 12] + "…[truncated]"


def _cosine(a: "np.ndarray", b: "np.ndarray") -> float:
    # Minimal allocation cosine
    denom = (math.sqrt(float((a * a).sum())) * math.sqrt(float((b * b).sum())))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum()) / denom


class Embedder:
    """Pluggable embedder interface (so we can avoid local models in RAM)."""

    def embed(self, texts: Sequence[str]) -> "np.ndarray":
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    """OpenAI embeddings via the official SDK (keeps RAM low; model runs remotely)."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        if np is None:
            raise RuntimeError("numpy is required for embeddings storage/reranking.")
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"openai SDK missing: {e}")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> "np.ndarray":
        # Small batches: keep memory minimal
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        vecs = [d.embedding for d in resp.data]
        return np.asarray(vecs, dtype="float32")


@dataclass
class RAGItem:
    id: int
    namespace: str
    key: str
    summary: str
    text_snippet: str
    meta: Dict[str, Any]
    created_ts: int
    score: Optional[float] = None


class SQLiteRAGStore:
    """SQLite-backed store with FTS5 keyword search + optional embedding rerank.

    Storage strategy (RAM-friendly)
    -------------------------------
    - For *large* payloads, store full text on disk (text_path) and only index a
      bounded snippet + summary in SQLite.
    - Use FTS5 + bm25() for cheap candidate recall, then (optionally) rerank only
      the small candidate set with vectors.

    Notes
    -----
    - Requires SQLite built with FTS5 (common in Python builds).
    - Uses WAL mode for better concurrent reads/writes (multi-agent runs).
    """

    def __init__(
        self,
        db_path: Path,
        blob_dir: Optional[Path] = None,
        embedder: Optional[Embedder] = None,
        *,
        max_inline_chars: int = 20000,
        max_snippet_chars: int = 3000,
    ):
        self.db_path = Path(db_path)
        self.blob_dir = Path(blob_dir) if blob_dir else self.db_path.parent / "rag_blobs"
        self.embedder = embedder
        self.max_inline_chars = int(max_inline_chars)
        self.max_snippet_chars = int(max_snippet_chars)

        _ensure_dir(self.db_path.parent)
        _ensure_dir(self.blob_dir)

        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.db_path), timeout=30.0)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        con = self._connect()
        try:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    text_inline TEXT DEFAULT '',
                    text_path TEXT DEFAULT '',
                    text_snippet TEXT DEFAULT '',
                    meta_json TEXT DEFAULT '{}',
                    created_ts INTEGER NOT NULL,
                    emb_model TEXT DEFAULT '',
                    emb_dim INTEGER DEFAULT 0,
                    emb BLOB
                );

                CREATE INDEX IF NOT EXISTS idx_rag_ns_ts ON rag_items(namespace, created_ts);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_ns_key ON rag_items(namespace, key);

                CREATE VIRTUAL TABLE IF NOT EXISTS rag_items_fts USING fts5(
                    namespace,
                    key,
                    summary,
                    text_snippet,
                    content=''
                );
                """
            )
            con.commit()
        finally:
            con.close()

    def _write_blob(self, namespace: str, key: str, text: str) -> str:
        safe_ns = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in namespace)[:80]
        safe_key = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)[:120]
        p = self.blob_dir / safe_ns
        _ensure_dir(p)
        fp = p / f"{safe_key}.txt"
        fp.write_text(text, encoding="utf-8", errors="ignore")
        return str(fp)

    def add(
        self,
        *,
        namespace: str,
        key: str,
        text: str,
        summary: str = "",
        meta: Optional[Dict[str, Any]] = None,
        created_ts: Optional[int] = None,
        embed_text_for_rerank: Optional[str] = None,
    ) -> int:
        """Add or replace an item.

        If an item with (namespace,key) exists, it is replaced (so keys are stable).
        """
        meta = meta or {}
        created_ts = int(created_ts or _now_ts())
        text = text or ""
        summary = summary or ""

        # Decide storage: inline vs blob file
        if len(text) > self.max_inline_chars:
            text_path = self._write_blob(namespace, key, text)
            text_inline = ""
        else:
            text_path = ""
            text_inline = text

        snippet_src = text_inline if text_inline else text
        text_snippet = _truncate(snippet_src, self.max_snippet_chars)

        emb = None
        emb_dim = 0
        emb_model = ""
        if self.embedder is not None and np is not None:
            try:
                to_embed = embed_text_for_rerank if embed_text_for_rerank is not None else (summary or text_snippet)
                vec = self.embedder.embed([to_embed])[0]
                vec = np.asarray(vec, dtype="float32")
                emb = vec.tobytes()
                emb_dim = int(vec.shape[0])
                emb_model = getattr(self.embedder, "model", "")
            except Exception:
                emb = None
                emb_dim = 0
                emb_model = ""

        con = self._connect()
        try:
            # Upsert via delete+insert to keep FTS in sync without triggers.
            cur = con.execute(
                "SELECT id FROM rag_items WHERE namespace=? AND key=?",
                (namespace, key),
            )
            row = cur.fetchone()
            if row:
                old_id = int(row["id"])
                con.execute("DELETE FROM rag_items WHERE id=?", (old_id,))
                con.execute("DELETE FROM rag_items_fts WHERE rowid=?", (old_id,))

            con.execute(
                """
                INSERT INTO rag_items(namespace,key,summary,text_inline,text_path,text_snippet,meta_json,created_ts,emb_model,emb_dim,emb)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    namespace,
                    key,
                    summary,
                    text_inline,
                    text_path,
                    text_snippet,
                    _safe_json(meta),
                    created_ts,
                    emb_model,
                    emb_dim,
                    emb,
                ),
            )
            new_id = con.execute("SELECT last_insert_rowid()").fetchone()[0]
            con.execute(
                "INSERT INTO rag_items_fts(rowid,namespace,key,summary,text_snippet) VALUES(?,?,?,?,?)",
                (new_id, namespace, key, summary, text_snippet),
            )
            con.commit()
            return int(new_id)
        finally:
            con.close()

    def _load_full_text(self, row: sqlite3.Row) -> str:
        inline = row["text_inline"] or ""
        if inline:
            return inline
        path = row["text_path"] or ""
        if not path:
            return ""
        try:
            return Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def search(
        self,
        *,
        namespace: str,
        query: str,
        k: int = 6,
        candidate_k: int = 30,
        use_embeddings: bool = True,
    ) -> List[RAGItem]:
        query = (query or "").strip()
        if not query:
            return []

        candidate_k = max(candidate_k, k)

        con = self._connect()
        try:
            # Keyword recall (cheap)
            rows = con.execute(
                """
                SELECT i.*, bm25(rag_items_fts) AS bm25_score
                FROM rag_items_fts
                JOIN rag_items i ON i.id = rag_items_fts.rowid
                WHERE rag_items_fts MATCH ? AND i.namespace=?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (query, namespace, int(candidate_k)),
            ).fetchall()

            items: List[RAGItem] = []
            for r in rows:
                try:
                    meta = json.loads(r["meta_json"] or "{}")
                except Exception:
                    meta = {}
                items.append(
                    RAGItem(
                        id=int(r["id"]),
                        namespace=r["namespace"],
                        key=r["key"],
                        summary=r["summary"] or "",
                        text_snippet=r["text_snippet"] or "",
                        meta=meta,
                        created_ts=int(r["created_ts"]),
                        score=float(r["bm25_score"]) if r["bm25_score"] is not None else None,
                    )
                )

            # Optional vector rerank (only on small candidate set)
            if (
                use_embeddings
                and self.embedder is not None
                and np is not None
                and items
            ):
                try:
                    qv = self.embedder.embed([query])[0]
                    qv = np.asarray(qv, dtype="float32")
                    # Fetch embeddings for candidates in one round trip
                    ids = [it.id for it in items]
                    q_marks = ",".join(["?"] * len(ids))
                    emb_rows = con.execute(
                        f"SELECT id, emb_dim, emb FROM rag_items WHERE id IN ({q_marks})",
                        ids,
                    ).fetchall()
                    emb_map: Dict[int, "np.ndarray"] = {}
                    for er in emb_rows:
                        if er["emb"] is None:
                            continue
                        dim = int(er["emb_dim"] or 0)
                        if dim <= 0:
                            continue
                        vec = np.frombuffer(er["emb"], dtype="float32", count=dim)
                        emb_map[int(er["id"])] = vec
                    # Compute cosine for items that have vectors
                    scored: List[Tuple[float, RAGItem]] = []
                    for it in items:
                        v = emb_map.get(it.id)
                        if v is None:
                            continue
                        scored.append((_cosine(qv, v), it))
                    if scored:
                        scored.sort(key=lambda x: x[0], reverse=True)
                        # Keep original items without vectors after vector-ranked ones
                        top = [it for _s, it in scored[:k]]
                        return top
                except Exception:
                    pass

            return items[:k]
        finally:
            con.close()

    def recent(
        self,
        *,
        namespace: str,
        k: int = 10,
    ) -> List[RAGItem]:
        con = self._connect()
        try:
            rows = con.execute(
                """
                SELECT *
                FROM rag_items
                WHERE namespace=?
                ORDER BY created_ts DESC
                LIMIT ?
                """,
                (namespace, int(k)),
            ).fetchall()
            out: List[RAGItem] = []
            for r in rows:
                try:
                    meta = json.loads(r["meta_json"] or "{}")
                except Exception:
                    meta = {}
                out.append(
                    RAGItem(
                        id=int(r["id"]),
                        namespace=r["namespace"],
                        key=r["key"],
                        summary=r["summary"] or "",
                        text_snippet=r["text_snippet"] or "",
                        meta=meta,
                        created_ts=int(r["created_ts"]),
                    )
                )
            return out
        finally:
            con.close()


class AgentRAGMemory:
    """A thin convenience wrapper around SQLiteRAGStore for agent-turn memory."""

    def __init__(
        self,
        *,
        db_path: Path,
        blob_dir: Optional[Path] = None,
        agent_name: str,
        openai_api_key: Optional[str] = None,
        embed_model: str = "text-embedding-3-small",
        enable_embeddings: bool = True,
    ):
        embedder: Optional[Embedder] = None
        if enable_embeddings and openai_api_key:
            try:
                embedder = OpenAIEmbedder(api_key=openai_api_key, model=embed_model)
            except Exception:
                embedder = None

        self.store = SQLiteRAGStore(db_path=db_path, blob_dir=blob_dir, embedder=embedder)
        self.namespace = f"agent_turns::{agent_name}"

    def remember_turn(
        self,
        *,
        key: str,
        phase: str,
        step: int,
        research_topic: str,
        feedback: str,
        response: str,
        keep_full_text: bool = False,
    ) -> int:
        # Keep a compact but information-dense record.
        text = (
            f"Phase: {phase}\n"
            f"Step: {step}\n"
            f"Research topic: {research_topic}\n"
            f"Feedback:\n{feedback}\n\n"
            f"Response:\n{response}\n"
        )
        summary = f"{phase} step {step}: {_truncate(response, 800)}"
        meta = {"phase": phase, "step": step}
        # For pure RAM reduction, we can avoid writing full text inline by forcing blob.
        if keep_full_text:
            max_inline_chars = self.store.max_inline_chars
        else:
            # Make sure large items go to blob even if store max_inline is big
            max_inline_chars = min(self.store.max_inline_chars, 6000)
        old = self.store.max_inline_chars
        self.store.max_inline_chars = max_inline_chars
        try:
            return self.store.add(
                namespace=self.namespace,
                key=key,
                text=text,
                summary=summary,
                meta=meta,
                embed_text_for_rerank=summary,
            )
        finally:
            self.store.max_inline_chars = old

    def recall(
        self,
        *,
        query: str,
        k: int = 6,
        max_chars: int = 3500,
    ) -> str:
        items = self.store.search(namespace=self.namespace, query=query, k=k)
        if not items:
            return ""
        lines = []
        for it in items:
            # Prefer summary; keep snippet as fallback
            s = (it.summary or it.text_snippet or "").strip()
            if not s:
                continue
            lines.append(f"- {s}")
        out = "\n".join(lines)
        return _truncate(out, max_chars)
