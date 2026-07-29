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
        interrupt_before: Sequence[str] | None = None,
        interrupt_after: Sequence[str] | None = None,
    ) -> None:
        """Connect to Postgres, migrate checkpoint tables, and compile the graph."""
        resolved_url = database_url or getenv("DATABASE_URL")
        if not resolved_url:
            raise ValueError(
                "DATABASE_URL is required for Postgres checkpointing."
            )

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
        initial_state: SupervisorState = {
            **state,
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
        initial_state: SupervisorState = {
            **state,
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

    def close(self) -> None:
        """Close the Postgres connection owned by this runtime."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> PersistentWriterAgent:
        """Return this runtime when entering a context manager."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the owned Postgres connection when leaving the context."""
        self.close()
