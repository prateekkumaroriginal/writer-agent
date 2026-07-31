"""Durable user memory backed by PostgreSQL and ChromaDB."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from os import getenv
from pathlib import Path
from typing import Literal
from uuid import uuid4

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.messages import HumanMessage, SystemMessage
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from writer_agent.model import llm
from writer_agent.prompts import MEMORY_MANAGEMENT_SYSTEM_PROMPT
from writer_agent.state import WorkflowEvent
from writer_agent.workflow_events import workflow_event

MemoryKind = Literal["core", "contextual"]
MemoryAction = Literal["add", "edit", "delete"]
MAX_MEMORY_LENGTH = 500
MAX_CORE_MEMORIES = 8
MAX_CONTEXTUAL_MEMORIES = 5
MAX_MEMORY_CONTEXT_CHARS = 4_000
MAX_CONTEXTUAL_DISTANCE = 0.65
CONTEXTUAL_EMBEDDING_MODEL = "chroma-default/all-MiniLM-L6-v2"
INDEX_BATCH_SIZE = 100
DEFAULT_CHROMA_PATH = ".writer_agent/chroma"

MEMORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS writer_memories (
    id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('core', 'contextual')),
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    source_task_id UUID,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT writer_memories_user_content
        UNIQUE (user_id, normalized_content)
);

CREATE INDEX IF NOT EXISTS writer_memories_user_kind_idx
ON writer_memories (user_id, kind, updated_at DESC);
"""


class MemoryView(BaseModel):
    """One user-visible durable memory."""

    model_config = ConfigDict(frozen=True)

    id: str
    kind: MemoryKind
    content: str
    source_task_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ProposedMemoryOperation(BaseModel):
    """One model-proposed mutation grounded in the current user message."""

    model_config = ConfigDict(extra="forbid")

    action: MemoryAction
    memory_id: str | None = None
    kind: MemoryKind | None = None
    content: str | None = Field(
        default=None,
        min_length=3,
        max_length=MAX_MEMORY_LENGTH,
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "ProposedMemoryOperation":
        if self.action == "add":
            if self.memory_id is not None or not self.kind or not self.content:
                raise ValueError(
                    "Add requires kind and content, without memory_id."
                )
        elif self.action == "edit":
            if not self.memory_id or not self.kind or not self.content:
                raise ValueError(
                    "Edit requires memory_id, kind, and content."
                )
        elif (
            not self.memory_id
            or self.kind is not None
            or self.content is not None
        ):
            raise ValueError(
                "Delete requires only memory_id."
            )
        return self


class MemoryDecisionSchema(BaseModel):
    """Bounded mutation decisions for one explicit user message."""

    model_config = ConfigDict(extra="forbid")

    operations: list[ProposedMemoryOperation] = Field(
        default_factory=list,
        max_length=5,
    )


class MemoryMutation(BaseModel):
    """One successfully applied mutation suitable for user-facing audit."""

    model_config = ConfigDict(frozen=True)

    action: MemoryAction
    previous: MemoryView | None = None
    current: MemoryView | None = None


def _normalize_content(content: str) -> str:
    return " ".join(content.casefold().split())


def _validate_content(content: str) -> str:
    normalized = " ".join(content.split())
    if len(normalized) < 3:
        raise ValueError("Memory content must contain at least 3 characters.")
    if len(normalized) > MAX_MEMORY_LENGTH:
        raise ValueError(
            f"Memories are limited to {MAX_MEMORY_LENGTH} characters."
        )
    return normalized


def _content_fingerprint(content: str) -> str:
    """Identify the exact content represented by one vector record."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class MemoryStore:
    """Structured memory index with contextual vectors stored in ChromaDB."""

    def __init__(
        self,
        database_url: str,
        *,
        chroma_path: str | None = None,
        min_size: int = 1,
        max_size: int = 5,
    ) -> None:
        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            kwargs={"autocommit": True, "row_factory": dict_row},
        )
        self._pool.open(wait=True)
        configured_path = chroma_path or getenv(
            "WRITER_AGENT_CHROMA_PATH",
            DEFAULT_CHROMA_PATH,
        )
        path = Path(configured_path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=str(path))
        self._embedding_function = DefaultEmbeddingFunction()
        self._collection = self._chroma.get_or_create_collection(
            name="writer_contextual_memories",
            embedding_function=self._embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def setup(self) -> None:
        with self._pool.connection() as connection:
            connection.execute(MEMORY_TABLE_SQL)
        self.rebuild_contextual_index()

    def close(self) -> None:
        self._pool.close()

    def add(
        self,
        *,
        user_id: str,
        kind: MemoryKind,
        content: str,
        source_task_id: str | None = None,
    ) -> tuple[MemoryView, bool]:
        clean_content = _validate_content(content)
        now = datetime.now(UTC)
        memory_id = str(uuid4())
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                INSERT INTO writer_memories (
                    id, user_id, kind, content, normalized_content,
                    source_task_id, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, normalized_content) DO NOTHING
                RETURNING *
                """,
                (
                    memory_id,
                    user_id,
                    kind,
                    clean_content,
                    _normalize_content(clean_content),
                    source_task_id,
                    now,
                    now,
                ),
            ).fetchone()
            created = row is not None
            if row is None:
                row = connection.execute(
                    """
                    SELECT * FROM writer_memories
                    WHERE user_id = %s AND normalized_content = %s
                    """,
                    (user_id, _normalize_content(clean_content)),
                ).fetchone()
        if row is None:
            raise RuntimeError("Memory lookup failed after conflict.")
        memory = _memory_from_row(row)
        if created and memory.kind == "contextual":
            self._upsert_contextual(user_id, memory)
        return memory, created

    def list(self, user_id: str, *, limit: int = 100) -> list[MemoryView]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM writer_memories
                WHERE user_id = %s
                ORDER BY kind, updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return [_memory_from_row(row) for row in rows]

    def update(
        self,
        memory_id: str,
        *,
        user_id: str,
        kind: MemoryKind,
        content: str,
        source_task_id: str | None = None,
    ) -> MemoryView:
        clean_content = _validate_content(content)
        with self._pool.connection() as connection:
            previous = connection.execute(
                """
                SELECT * FROM writer_memories
                WHERE id = %s AND user_id = %s
                """,
                (memory_id, user_id),
            ).fetchone()
            if previous is None:
                raise KeyError(f"No memory found for {memory_id!r}.")
            row = connection.execute(
                """
                UPDATE writer_memories
                SET kind = %s, content = %s, normalized_content = %s,
                    source_task_id = COALESCE(%s, source_task_id),
                    updated_at = %s
                WHERE id = %s AND user_id = %s
                RETURNING *
                """,
                (
                    kind,
                    clean_content,
                    _normalize_content(clean_content),
                    source_task_id,
                    datetime.now(UTC),
                    memory_id,
                    user_id,
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("Memory update did not return a row.")
        if previous["kind"] == "contextual" and kind != "contextual":
            self._collection.delete(ids=[memory_id])
        memory = _memory_from_row(row)
        if kind == "contextual":
            self._upsert_contextual(user_id, memory)
        return memory

    def delete(self, memory_id: str, *, user_id: str) -> None:
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                DELETE FROM writer_memories
                WHERE id = %s AND user_id = %s
                RETURNING kind
                """,
                (memory_id, user_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"No memory found for {memory_id!r}.")
        if row["kind"] == "contextual":
            self._collection.delete(ids=[memory_id])

    def retrieve(self, *, user_id: str, query: str) -> list[MemoryView]:
        with self._pool.connection() as connection:
            core_rows = connection.execute(
                """
                SELECT * FROM writer_memories
                WHERE user_id = %s AND kind = 'core'
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, MAX_CORE_MEMORIES),
            ).fetchall()
        memories = [_memory_from_row(row) for row in core_rows]
        if query.strip():
            result = self._collection.query(
                query_texts=[query],
                n_results=MAX_CONTEXTUAL_MEMORIES,
                where={"user_id": user_id},
                include=["metadatas", "distances"],
            )
            ids = (result.get("ids") or [[]])[0]
            distances = (result.get("distances") or [[]])[0]
            relevant_ids = [
                memory_id
                for memory_id, distance in zip(ids, distances, strict=False)
                if distance <= MAX_CONTEXTUAL_DISTANCE
            ]
            if relevant_ids:
                memories.extend(self._get_many(user_id, relevant_ids))
        return memories

    def format_context(self, *, user_id: str, query: str) -> str:
        memories = self.retrieve(user_id=user_id, query=query)
        if not memories:
            return "No relevant saved memories."
        lines: list[str] = []
        total = 0
        for memory in memories:
            line = f"- [{memory.kind}] {memory.content}"
            if total + len(line) + 1 > MAX_MEMORY_CONTEXT_CHARS:
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines) or "No relevant saved memories."

    def rebuild_contextual_index(self) -> None:
        """Reconcile Chroma with PostgreSQL without re-embedding unchanged rows."""
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM writer_memories
                WHERE kind = 'contextual'
                """
            ).fetchall()
        expected = {
            str(row["id"]): (row["user_id"], _memory_from_row(row))
            for row in rows
        }
        indexed = self._collection.get(include=["metadatas"])
        indexed_ids = indexed.get("ids") or []
        indexed_metadata = indexed.get("metadatas") or []
        metadata_by_id = dict(
            zip(indexed_ids, indexed_metadata, strict=False)
        )

        orphaned_ids = [
            memory_id
            for memory_id in indexed_ids
            if memory_id not in expected
        ]
        if orphaned_ids:
            self._collection.delete(ids=orphaned_ids)

        pending: list[tuple[str, MemoryView]] = []
        for memory_id, (user_id, memory) in expected.items():
            metadata = metadata_by_id.get(memory_id) or {}
            if (
                metadata.get("user_id") != user_id
                or metadata.get("embedding_model")
                != CONTEXTUAL_EMBEDDING_MODEL
                or metadata.get("content_fingerprint")
                != _content_fingerprint(memory.content)
            ):
                pending.append((user_id, memory))

        for offset in range(0, len(pending), INDEX_BATCH_SIZE):
            self._upsert_contextual_batch(
                pending[offset : offset + INDEX_BATCH_SIZE]
            )

    def _get_many(
        self,
        user_id: str,
        memory_ids: list[str],
    ) -> list[MemoryView]:
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM writer_memories
                WHERE user_id = %s AND id = ANY(%s::uuid[])
                """,
                (user_id, memory_ids),
            ).fetchall()
        by_id = {str(row["id"]): _memory_from_row(row) for row in rows}
        return [by_id[memory_id] for memory_id in memory_ids if memory_id in by_id]

    def _upsert_contextual(self, user_id: str, memory: MemoryView) -> None:
        self._upsert_contextual_batch([(user_id, memory)])

    def _upsert_contextual_batch(
        self,
        records: list[tuple[str, MemoryView]],
    ) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[memory.id for _, memory in records],
            documents=[memory.content for _, memory in records],
            metadatas=[
                {
                    "user_id": user_id,
                    "kind": memory.kind,
                    "embedding_model": CONTEXTUAL_EMBEDDING_MODEL,
                    "content_fingerprint": _content_fingerprint(
                        memory.content
                    ),
                }
                for user_id, memory in records
            ],
        )


def manage_durable_memories(
    store: MemoryStore,
    *,
    user_id: str,
    user_message: str,
    source_task_id: str | None,
) -> list[MemoryMutation]:
    """Apply agent-decided durable-memory mutations from one user message."""
    try:
        candidates = store.retrieve(user_id=user_id, query=user_message)
    except Exception:
        return []
    inventory = (
        "\n".join(
            (
                f"- id={memory.id} | kind={memory.kind} | "
                f"content={memory.content}"
            )
            for memory in candidates
        )
        or "None"
    )
    extractor = llm.with_structured_output(MemoryDecisionSchema)
    try:
        result = extractor.invoke(
            [
                SystemMessage(content=MEMORY_MANAGEMENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"""
Existing relevant saved memories:
{inventory}

New user message:
{user_message}
""".strip()
                ),
            ]
        )
    except Exception:
        return []

    existing_by_id = {memory.id: memory for memory in candidates}
    touched_ids: set[str] = set()
    mutations: list[MemoryMutation] = []
    for operation in result.operations:
        try:
            if operation.action == "add":
                memory, created = store.add(
                    user_id=user_id,
                    kind=operation.kind,
                    content=operation.content,
                    source_task_id=source_task_id,
                )
                if created:
                    mutations.append(
                        MemoryMutation(action="add", current=memory)
                    )
                continue

            memory_id = operation.memory_id
            previous = existing_by_id.get(memory_id)
            if previous is None or memory_id in touched_ids:
                continue
            touched_ids.add(memory_id)
            if operation.action == "edit":
                if (
                    previous.kind == operation.kind
                    and _normalize_content(previous.content)
                    == _normalize_content(operation.content)
                ):
                    continue
                current = store.update(
                    memory_id,
                    user_id=user_id,
                    kind=operation.kind,
                    content=operation.content,
                    source_task_id=source_task_id,
                )
                mutations.append(
                    MemoryMutation(
                        action="edit",
                        previous=previous,
                        current=current,
                    )
                )
            else:
                store.delete(memory_id, user_id=user_id)
                mutations.append(
                    MemoryMutation(
                        action="delete",
                        previous=previous,
                    )
                )
        except Exception:
            continue
    return mutations


def memory_events_for_mutations(
    mutations: list[MemoryMutation],
) -> list[WorkflowEvent]:
    """Translate applied mutations into transparent workflow events."""
    events: list[WorkflowEvent] = []
    for mutation in mutations:
        if mutation.action == "add" and mutation.current is not None:
            events.append(
                workflow_event(
                    "memory",
                    "Memory added",
                    content=mutation.current.content,
                    details=[f"Type: {mutation.current.kind}"],
                    decision="add",
                )
            )
        elif (
            mutation.action == "edit"
            and mutation.previous is not None
            and mutation.current is not None
        ):
            events.append(
                workflow_event(
                    "memory",
                    "Memory updated",
                    content=mutation.current.content,
                    details=[
                        f"Previous: {mutation.previous.content}",
                        f"Type: {mutation.current.kind}",
                    ],
                    decision="edit",
                )
            )
        elif mutation.action == "delete" and mutation.previous is not None:
            events.append(
                workflow_event(
                    "memory",
                    "Memory deleted",
                    content=mutation.previous.content,
                    details=[f"Type: {mutation.previous.kind}"],
                    decision="delete",
                )
            )
    return events


def _memory_from_row(row: dict) -> MemoryView:
    return MemoryView(
        id=str(row["id"]),
        kind=row["kind"],
        content=row["content"],
        source_task_id=(
            str(row["source_task_id"]) if row.get("source_task_id") else None
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
