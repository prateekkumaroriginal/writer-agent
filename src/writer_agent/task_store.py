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
    TaskVersionSummary,
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
    conversation_id UUID NOT NULL,
    parent_task_id UUID,
    turn_number INTEGER NOT NULL DEFAULT 1,
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

ALTER TABLE writer_tasks
ADD COLUMN IF NOT EXISTS conversation_id UUID;

ALTER TABLE writer_tasks
ADD COLUMN IF NOT EXISTS parent_task_id UUID;

ALTER TABLE writer_tasks
ADD COLUMN IF NOT EXISTS turn_number INTEGER NOT NULL DEFAULT 1;

UPDATE writer_tasks
SET conversation_id = id
WHERE conversation_id IS NULL;

ALTER TABLE writer_tasks
ALTER COLUMN conversation_id SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS writer_tasks_conversation_turn_idx
ON writer_tasks (conversation_id, turn_number);

CREATE INDEX IF NOT EXISTS writer_tasks_conversation_idx
ON writer_tasks (conversation_id, turn_number);
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
        conversation_id = task_id
        steps = steps_for_stage("queued")
        with self._pool.connection() as connection:
            created = connection.execute(
                """
                INSERT INTO writer_tasks (
                    id, thread_id, user_id, conversation_id, turn_number,
                    idempotency_key, request, title,
                    status, stage, status_message, progress_current,
                    progress_total, steps, created_at, updated_at
                )
                VALUES (
                    %s, %s, %s, %s, 1, %s, %s, %s,
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
                    conversation_id,
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
        if existing.get("parent_task_id") is not None:
            raise ValueError(
                "This idempotency key belongs to a conversation follow-up."
            )
        return _task_from_row(existing), False

    def create_follow_up(
        self,
        *,
        parent_task_id: str,
        user_id: str,
        request: str,
        idempotency_key: str,
    ) -> tuple[TaskView, bool]:
        """Append one queued run to the latest completed conversation turn."""
        now = datetime.now(UTC)
        task_id = str(uuid4())
        thread_id = f"writer-{task_id}"
        with self._pool.connection() as connection:
            with connection.transaction():
                parent = connection.execute(
                    """
                    SELECT * FROM writer_tasks
                    WHERE id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (parent_task_id, user_id),
                ).fetchone()
                if parent is None:
                    raise KeyError(
                        f"No writing task found for {parent_task_id!r}."
                    )
                conversation_id = parent.get("conversation_id") or parent["id"]
                latest = connection.execute(
                    """
                    SELECT * FROM writer_tasks
                    WHERE conversation_id = %s AND user_id = %s
                    ORDER BY turn_number DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (conversation_id, user_id),
                ).fetchone()
                if latest is None:
                    raise RuntimeError("Conversation has no latest run.")

                existing = connection.execute(
                    """
                    SELECT * FROM writer_tasks
                    WHERE user_id = %s AND idempotency_key = %s
                    """,
                    (user_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if str(existing.get("parent_task_id")) != str(parent["id"]):
                        raise ValueError(
                            "This idempotency key belongs to another task."
                        )
                    return _task_from_row(existing), False

                if str(latest["id"]) != str(parent["id"]):
                    raise ValueError(
                        "Follow-ups can only be added to the latest version."
                    )
                if latest["status"] != "completed":
                    raise ValueError(
                        "Wait for the current version to complete before "
                        "adding a follow-up."
                    )

                created = connection.execute(
                    """
                    INSERT INTO writer_tasks (
                        id, thread_id, user_id, conversation_id,
                        parent_task_id, turn_number, idempotency_key,
                        request, title, status, stage, status_message,
                        progress_current, progress_total, steps,
                        created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, 'queued', 'queued', 'Waiting to start…',
                        0, 5, %s, %s, %s
                    )
                    RETURNING *
                    """,
                    (
                        task_id,
                        thread_id,
                        user_id,
                        conversation_id,
                        parent["id"],
                        int(parent.get("turn_number") or 1) + 1,
                        idempotency_key,
                        request,
                        parent["title"],
                        Jsonb(jsonable_models(steps_for_stage("queued"))),
                        now,
                        now,
                    ),
                ).fetchone()
        if created is None:
            raise RuntimeError("Follow-up creation did not return a task.")
        return _task_from_row(created), True

    def create_retry(
        self,
        *,
        task_id: str,
        user_id: str,
        idempotency_key: str,
    ) -> tuple[TaskView, bool]:
        """Append a fresh run for the latest failed conversation turn."""
        now = datetime.now(UTC)
        retry_id = str(uuid4())
        thread_id = f"writer-{retry_id}"
        with self._pool.connection() as connection:
            with connection.transaction():
                failed_task = connection.execute(
                    """
                    SELECT * FROM writer_tasks
                    WHERE id = %s AND user_id = %s
                    FOR UPDATE
                    """,
                    (task_id, user_id),
                ).fetchone()
                if failed_task is None:
                    raise KeyError(f"No writing task found for {task_id!r}.")

                conversation_id = (
                    failed_task.get("conversation_id") or failed_task["id"]
                )
                latest = connection.execute(
                    """
                    SELECT * FROM writer_tasks
                    WHERE conversation_id = %s AND user_id = %s
                    ORDER BY turn_number DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    (conversation_id, user_id),
                ).fetchone()
                if latest is None:
                    raise RuntimeError("Conversation has no latest run.")

                existing = connection.execute(
                    """
                    SELECT * FROM writer_tasks
                    WHERE user_id = %s AND idempotency_key = %s
                    """,
                    (user_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if str(existing.get("parent_task_id")) != str(
                        failed_task["id"]
                    ):
                        raise ValueError(
                            "This idempotency key belongs to another task."
                        )
                    return _task_from_row(existing), False

                if str(latest["id"]) != str(failed_task["id"]):
                    raise ValueError(
                        "Only the latest conversation version can be retried."
                    )
                if failed_task["status"] not in {"escalated", "failed"}:
                    raise ValueError(
                        "Only a failed task can be tried again."
                    )

                created = connection.execute(
                    """
                    INSERT INTO writer_tasks (
                        id, thread_id, user_id, conversation_id,
                        parent_task_id, turn_number, idempotency_key,
                        request, title, status, stage, status_message,
                        progress_current, progress_total, steps,
                        created_at, updated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, 'queued', 'queued', 'Waiting to start…',
                        0, 5, %s, %s, %s
                    )
                    RETURNING *
                    """,
                    (
                        retry_id,
                        thread_id,
                        user_id,
                        conversation_id,
                        failed_task["id"],
                        int(failed_task.get("turn_number") or 1) + 1,
                        idempotency_key,
                        failed_task["request"],
                        failed_task["title"],
                        Jsonb(jsonable_models(steps_for_stage("queued"))),
                        now,
                        now,
                    ),
                ).fetchone()
        if created is None:
            raise RuntimeError("Retry creation did not return a task.")
        return _task_from_row(created), True

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
                SELECT id, conversation_id, turn_number, title, status, stage,
                       updated_at
                FROM (
                    SELECT DISTINCT ON (conversation_id)
                        id, conversation_id, turn_number, title, status, stage,
                        updated_at
                    FROM writer_tasks
                    WHERE user_id = %s
                    ORDER BY conversation_id, turn_number DESC
                ) latest
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            ).fetchall()
        return [
            TaskSummary(
                id=str(row["id"]),
                conversation_id=str(row["conversation_id"]),
                turn_number=row["turn_number"],
                title=row["title"],
                status=row["status"],
                stage=row["stage"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def list_versions(
        self,
        task_id: str,
        *,
        user_id: str,
    ) -> list[TaskVersionSummary]:
        """List all persisted turns for the task's conversation."""
        with self._pool.connection() as connection:
            task = connection.execute(
                """
                SELECT conversation_id FROM writer_tasks
                WHERE id = %s AND user_id = %s
                """,
                (task_id, user_id),
            ).fetchone()
            if task is None:
                raise KeyError(f"No writing task found for {task_id!r}.")
            rows = connection.execute(
                """
                SELECT id, turn_number, request, status, updated_at
                FROM writer_tasks
                WHERE conversation_id = %s AND user_id = %s
                ORDER BY turn_number
                """,
                (task["conversation_id"], user_id),
            ).fetchall()
        return [
            TaskVersionSummary(
                id=str(row["id"]),
                turn_number=row["turn_number"],
                request=row["request"],
                status=row["status"],
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
        conversation_id=str(row.get("conversation_id") or row["id"]),
        parent_task_id=(
            str(row["parent_task_id"]) if row.get("parent_task_id") else None
        ),
        turn_number=int(row.get("turn_number") or 1),
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
