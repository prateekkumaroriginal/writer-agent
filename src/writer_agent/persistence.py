"""Durable workflow runtimes and checkpoint inspection helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from os import getenv
from typing import Any

import psycopg
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import StateSnapshot
from psycopg.rows import dict_row

from writer_agent.graph import build_supervisor_graph
from writer_agent.memory import (
    MemoryStore,
    manage_durable_memories,
    memory_events_for_mutations,
)
from writer_agent.state import SupervisorState


def thread_config(
    thread_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> RunnableConfig:
    """Build the LangGraph configuration used to identify a durable thread."""
    if not thread_id or not thread_id.strip():
        raise ValueError("thread_id must be a non-empty string")

    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id.strip()},
    }
    if metadata:
        config["metadata"] = dict(metadata)
    return config


class PersistentWriterAgent:
    """Run the writer graph with Postgres checkpoint persistence."""

    def __init__(
        self,
        database_url: str | None = None,
        *,
        memory_store: MemoryStore | None = None,
        interrupt_before: Sequence[str] | None = None,
        interrupt_after: Sequence[str] | None = None,
    ) -> None:
        """Connect to Postgres, migrate checkpoint tables, and compile the graph."""
        resolved_url = database_url or getenv("DATABASE_URL")
        if not resolved_url:
            raise ValueError(
                "DATABASE_URL is required for Postgres checkpointing."
            )

        self._database_url = resolved_url
        self._memory_store = memory_store
        self._owns_memory_store = False
        self._connection = psycopg.connect(
            resolved_url,
            autocommit=True,
            row_factory=dict_row,
        )
        self.checkpointer = PostgresSaver(self._connection)
        self.checkpointer.setup()
        self.graph = build_supervisor_graph(
            checkpointer=self.checkpointer,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )
        self._closed = False

    def start(
        self,
        thread_id: str,
        state: SupervisorState,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SupervisorState:
        """Start a new durable workflow thread.

        Existing thread IDs are rejected so a completed or interrupted run is
        not accidentally overwritten. Use ``resume`` for an existing thread.
        """
        config = thread_config(thread_id, metadata)
        if self.graph.get_state(config).values:
            raise ValueError(
                f"Thread {thread_id!r} already exists; use resume() or a new ID."
            )

        run_metadata = {
            "started_at": datetime.now(UTC).isoformat(),
            **dict(metadata or {}),
        }
        prepared_state = self._with_memory_context(state)
        initial_state: SupervisorState = {
            **prepared_state,
            "thread_id": thread_id.strip(),
            "run_metadata": run_metadata,
        }
        return self.graph.invoke(initial_state, config=config)

    def start_stream(
        self,
        thread_id: str,
        state: SupervisorState,
        *,
        on_state: Callable[[SupervisorState], object],
        metadata: Mapping[str, Any] | None = None,
    ) -> SupervisorState:
        """Start a workflow and report checkpointed states as they are produced."""
        config = thread_config(thread_id, metadata)
        if self.graph.get_state(config).values:
            raise ValueError(
                f"Thread {thread_id!r} already exists; use resume() or a new ID."
            )

        run_metadata = {
            "started_at": datetime.now(UTC).isoformat(),
            **dict(metadata or {}),
        }
        prepared_state = self._with_memory_context(state)
        initial_state: SupervisorState = {
            **prepared_state,
            "thread_id": thread_id.strip(),
            "run_metadata": run_metadata,
        }
        return self._stream(
            initial_state,
            config=config,
            on_state=on_state,
        )

    def resume(
        self,
        thread_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> SupervisorState:
        """Continue an interrupted workflow from its latest checkpoint."""
        config = thread_config(thread_id, metadata)
        snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise KeyError(f"No checkpoints found for thread {thread_id!r}.")
        return self.graph.invoke(None, config=config)

    def resume_stream(
        self,
        thread_id: str,
        *,
        on_state: Callable[[SupervisorState], object],
        metadata: Mapping[str, Any] | None = None,
    ) -> SupervisorState:
        """Resume a workflow and report checkpointed states as they are produced."""
        config = thread_config(thread_id, metadata)
        snapshot = self.graph.get_state(config)
        if not snapshot.values:
            raise KeyError(f"No checkpoints found for thread {thread_id!r}.")
        return self._stream(None, config=config, on_state=on_state)

    def get_state(self, thread_id: str) -> StateSnapshot:
        """Return the latest checkpoint snapshot for a thread."""
        return self.graph.get_state(thread_config(thread_id))

    def get_history(self, thread_id: str) -> list[StateSnapshot]:
        """Return newest-first checkpoint history for a thread."""
        return list(self.graph.get_state_history(thread_config(thread_id)))

    def _stream(
        self,
        input_state: SupervisorState | None,
        *,
        config: RunnableConfig,
        on_state: Callable[[SupervisorState], object],
    ) -> SupervisorState:
        """Consume root and specialist state updates, then return durable state."""
        for _, streamed_state in self.graph.stream(
            input_state,
            config=config,
            stream_mode="values",
            subgraphs=True,
        ):
            if isinstance(streamed_state, dict):
                on_state(streamed_state)

        snapshot = self.graph.get_state(config)
        final_state = dict(snapshot.values)
        on_state(final_state)
        return final_state

    def _with_memory_context(
        self,
        state: SupervisorState,
    ) -> SupervisorState:
        """Lazily activate memory for direct Python API callers."""
        if state.get("memory_context"):
            return state
        user_id = state.get("user_id")
        user_request = state.get("user_request")
        if not user_id or not user_request:
            return {
                **state,
                "memory_context": "No relevant saved memories.",
            }
        try:
            if self._memory_store is None:
                self._memory_store = MemoryStore(self._database_url)
                self._owns_memory_store = True
                self._memory_store.setup()
            mutations = manage_durable_memories(
                self._memory_store,
                user_id=user_id,
                user_message=user_request,
                source_task_id=None,
            )
            memory_events = memory_events_for_mutations(mutations)
            memory_context = self._memory_store.format_context(
                user_id=user_id,
                query=user_request,
            )
        except Exception:
            memory_context = "No relevant saved memories."
            memory_events = []
        return {
            **state,
            "memory_context": memory_context,
            "workflow_events": [
                *state.get("workflow_events", []),
                *memory_events,
            ],
        }

    def close(self) -> None:
        """Close the Postgres connection owned by this runtime."""
        if not self._closed:
            if self._owns_memory_store and self._memory_store is not None:
                self._memory_store.close()
            self._connection.close()
            self._closed = True

    def __enter__(self) -> PersistentWriterAgent:
        """Return this runtime when entering a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the owned Postgres connection when leaving the context."""
        self.close()
