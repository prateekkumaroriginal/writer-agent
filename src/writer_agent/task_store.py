"""PostgreSQL task index used by the Streamlit product UI."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from writer_agent.state import SupervisorState
from writer_agent.ui_models import (
    ReviewView,
    SourceView,
    StepView,
    TaskProjection,
    TaskStage,
    TaskStatus,
    TaskSummary,
    TaskView,
    WorkflowEventView,
    jsonable_models,
    projection_from_state,
    steps_for_stage,
    title_from_request,
)

TASK_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS writer_tasks (
    id UUID PRIMARY KEY,
    thread_id TEXT NOT NULL UNIQUE,
    user_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    request TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    status_message TEXT NOT NULL,
    progress_current INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER NOT NULL DEFAULT 5,
    steps JSONB NOT NULL DEFAULT '[]'::jsonb,
    final_answer TEXT,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    plan TEXT,
    plan_confidence DOUBLE PRECISION,
    review JSONB,
    workflow_events JSONB NOT NULL DEFAULT '[]'::jsonb,
    can_resume BOOLEAN NOT NULL DEFAULT FALSE,
    internal_error TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CONSTRAINT writer_tasks_user_idempotency UNIQUE (user_id, idempotency_key)
);

CREATE INDEX IF NOT EXISTS writer_tasks_user_recent_idx
ON writer_tasks (user_id, updated_at DESC);
"""

TASK_MIGRATION_SQL = """
ALTER TABLE writer_tasks
ADD COLUMN IF NOT EXISTS workflow_events JSONB NOT NULL DEFAULT '[]'::jsonb;
"""


class PostgresTaskStore:
    """Durable, thread-safe task projection store."""

    def __init__(
        self,
        database_url: str,
        *,
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

    def setup(self) -> None:
        """Create the application-level task index."""
        with self._pool.connection() as connection:
            connection.execute(TASK_TABLE_SQL)
            connection.execute(TASK_MIGRATION_SQL)

    def create(
        self,
        *,
        user_id: str,
        request: str,
        idempotency_key: str,
    ) -> tuple[TaskView, bool]:
        """Create a queued task or return its idempotent predecessor."""
        now = datetime.now(UTC)
        task_id = str(uuid4())
        thread_id = f"writer-{task_id}"
        steps = steps_for_stage("queued")
        with self._pool.connection() as connection:
            created = connection.execute(
                """
                INSERT INTO writer_tasks (
                    id, thread_id, user_id, idempotency_key, request, title,
                    status, stage, status_message, progress_current,
                    progress_total, steps, created_at, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    'queued', 'queued', 'Waiting to start…', 0,
                    5, %s, %s, %s
                )
                ON CONFLICT (user_id, idempotency_key) DO NOTHING
                RETURNING *
                """,
                (
                    task_id,
                    thread_id,
                    user_id,
                    idempotency_key,
                    request,
                    title_from_request(request),
                    Jsonb(jsonable_models(steps)),
                    now,
                    now,
                ),
            ).fetchone()
            if created is not None:
                return _task_from_row(created), True
            existing = connection.execute(
                """
                SELECT * FROM writer_tasks
                WHERE user_id = %s AND idempotency_key = %s
                """,
                (user_id, idempotency_key),
            ).fetchone()
        if existing is None:
            raise RuntimeError("Idempotent task lookup failed after conflict.")
        return _task_from_row(existing), False

    def get(self, task_id: str, *, user_id: str | None = None) -> TaskView:
        """Return one task, optionally enforcing its owner."""
        query = "SELECT * FROM writer_tasks WHERE id = %s"
        parameters: tuple[Any, ...] = (task_id,)
        if user_id is not None:
            query += " AND user_id = %s"
            parameters += (user_id,)
        with self._pool.connection() as connection:
            row = connection.execute(query, parameters).fetchone()
        if row is None:
            raise KeyError(f"No writing task found for {task_id!r}.")
        return _task_from_row(row)

    def list_recent(
        self,
        user_id: str,
        *,
        limit: int = 20,
    ) -> list[TaskSummary]:
        """List the user's newest tasks for the sidebar."""
        with self._pool.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, title, status, stage, updated_at
                FROM writer_tasks
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return [
            TaskSummary(
                id=str(row["id"]),
                title=row["title"],
                status=row["status"],
                stage=row["stage"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def claim(self, task_id: str, *, resume: bool) -> bool:
        """Atomically claim a queued or interrupted task for one worker."""
        expected_status = "interrupted" if resume else "queued"
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                UPDATE writer_tasks
                SET status = 'running',
                    status_message = %s,
                    can_resume = FALSE,
                    internal_error = NULL,
                    updated_at = %s
                WHERE id = %s AND status = %s
                RETURNING id
                """,
                (
                    "Resuming your workflow…"
                    if resume
                    else "Starting your workflow…",
                    datetime.now(UTC),
                    task_id,
                    expected_status,
                ),
            ).fetchone()
        return row is not None

    def update_from_state(
        self,
        task_id: str,
        state: SupervisorState,
    ) -> TaskView:
        """Persist a sanitized graph-state projection."""
        projection = projection_from_state(state)
        now = datetime.now(UTC)
        with self._pool.connection() as connection:
            row = connection.execute(
                """
                UPDATE writer_tasks
                SET status = %s,
                    stage = %s,
                    status_message = %s,
                    progress_current = %s,
                    steps = %s,
                    final_answer = %s,
                    sources = %s,
                    plan = %s,
                    plan_confidence = %s,
                    review = %s,
                    workflow_events = %s,
                    can_resume = %s,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
                """,
                (
                    projection.status,
                    projection.stage,
                    projection.status_message,
                    projection.progress_current,
                    Jsonb(jsonable_models(projection.steps)),
                    projection.final_answer,
                    Jsonb(jsonable_models(projection.sources)),
                    projection.plan,
                    projection.plan_confidence,
                    Jsonb(projection.review.model_dump(mode="json"))
                    if projection.review
                    else None,
                    Jsonb(jsonable_models(projection.workflow_events)),
                    projection.can_resume,
                    now,
                    task_id,
                ),
            ).fetchone()
        if row is None:
            raise KeyError(f"No writing task found for {task_id!r}.")
        return _task_from_row(row)

    def mark_interrupted(
        self,
        task_id: str,
        *,
        message: str | None = None,
        internal_error: str | None = None,
    ) -> None:
        """Make an interrupted checkpoint resumable."""
        self._set_terminal(
            task_id,
            status="interrupted",
            stage="attention",
            message=message
            or (
                "This task stopped before completion. You can resume it "
                "from its latest checkpoint."
            ),
            can_resume=True,
            internal_error=internal_error,
        )

    def mark_failed(
        self,
        task_id: str,
        *,
        message: str = "The workflow could not complete. Your request was saved.",
        internal_error: str | None = None,
    ) -> None:
        """Record a terminal failure without exposing technical detail."""
        self._set_terminal(
            task_id,
            status="failed",
            stage="failed",
            message=message,
            can_resume=False,
            internal_error=internal_error,
        )

    def recover_orphaned(self) -> None:
        """Reclassify work left behind by a previous UI process."""
        now = datetime.now(UTC)
        with self._pool.connection() as connection:
            connection.execute(
                """
                UPDATE writer_tasks
                SET status = 'interrupted',
                    stage = 'attention',
                    status_message = (
                        'This task stopped before completion. You can resume it '
                        'from its latest checkpoint.'
                    ),
                    can_resume = TRUE,
                    updated_at = %s
                WHERE status = 'running'
                """,
                (now,),
            )
            connection.execute(
                """
                UPDATE writer_tasks
                SET status = 'failed',
                    stage = 'failed',
                    status_message = (
                        'The app stopped before this task began. Start a new task '
                        'to try again.'
                    ),
                    can_resume = FALSE,
                    updated_at = %s
                WHERE status = 'queued'
                """,
                (now,),
            )

    def close(self) -> None:
        self._pool.close()

    def _set_terminal(
        self,
        task_id: str,
        *,
        status: TaskStatus,
        stage: TaskStage,
        message: str,
        can_resume: bool,
        internal_error: str | None,
    ) -> None:
        steps = steps_for_stage(stage)
        progress = sum(step.status in {"done", "error"} for step in steps)
        with self._pool.connection() as connection:
            connection.execute(
                """
                UPDATE writer_tasks
                SET status = %s,
                    stage = %s,
                    status_message = %s,
                    progress_current = %s,
                    steps = %s,
                    can_resume = %s,
                    internal_error = %s,
                    updated_at = %s
                WHERE id = %s
                """,
                (
                    status,
                    stage,
                    message,
                    progress,
                    Jsonb(jsonable_models(steps)),
                    can_resume,
                    internal_error,
                    datetime.now(UTC),
                    task_id,
                ),
            )


def _task_from_row(row: Mapping[str, Any]) -> TaskView:
    """Validate a database row at the UI boundary."""
    review_data = row.get("review")
    return TaskView(
        id=str(row["id"]),
        thread_id=row["thread_id"],
        user_id=row["user_id"],
        title=row["title"],
        request=row["request"],
        status=row["status"],
        stage=row["stage"],
        status_message=row["status_message"],
        progress_current=row["progress_current"],
        progress_total=row["progress_total"],
        steps=[StepView.model_validate(item) for item in row["steps"]],
        final_answer=row.get("final_answer"),
        sources=[
            SourceView.model_validate(item) for item in row.get("sources", [])
        ],
        plan=row.get("plan"),
        plan_confidence=row.get("plan_confidence"),
        review=(
            ReviewView.model_validate(review_data) if review_data else None
        ),
        workflow_events=[
            WorkflowEventView.model_validate(item)
            for item in row.get("workflow_events", [])
        ],
        can_resume=row.get("can_resume", False),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
