"""Application service and background runner for the Streamlit UI."""

from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from os import getenv
from threading import Lock
from uuid import uuid4

from writer_agent.persistence import PersistentWriterAgent
from writer_agent.provider_errors import RetryableProviderError
from writer_agent.run_records import record_run
from writer_agent.task_store import PostgresTaskStore
from writer_agent.ui_models import TaskSummary, TaskVersionSummary, TaskView

DEFAULT_USER_ID = "local-user"
MAX_REQUEST_LENGTH = 8_000
DEFAULT_LOCAL_DATABASE_URL = (
    "postgresql://writer_agent:writer_agent_dev@localhost:5432/"
    "writer_agent?sslmode=disable"
)


class TaskRunner:
    """Run durable workflows outside Streamlit's rerun lifecycle."""

    def __init__(
        self,
        store: PostgresTaskStore,
        database_url: str,
        *,
        max_workers: int = 1,
    ) -> None:
        self._store = store
        self._database_url = database_url
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="writer-agent",
        )
        self._futures: dict[str, Future[None]] = {}
        self._lock = Lock()

    def submit(self, task_id: str, *, resume: bool = False) -> None:
        """Schedule one task if it is not already active in this process."""
        with self._lock:
            future = self._futures.get(task_id)
            if future is not None and not future.done():
                return
            future = self._executor.submit(self._run, task_id, resume)
            self._futures[task_id] = future
            future.add_done_callback(
                lambda _future, current_id=task_id: self._forget(current_id)
            )

    def is_active(self, task_id: str) -> bool:
        with self._lock:
            future = self._futures.get(task_id)
            return future is not None and not future.done()

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _forget(self, task_id: str) -> None:
        with self._lock:
            self._futures.pop(task_id, None)

    def _run(self, task_id: str, resume: bool) -> None:
        if not self._store.claim(task_id, resume=resume):
            return
        task = self._store.get(task_id)
        try:
            with PersistentWriterAgent(self._database_url) as runtime:
                def callback(state):
                    self._store.update_from_state(task_id, state)
                    try:
                        record_run(task.thread_id, state)
                    except (OSError, TypeError, ValueError):
                        pass

                metadata = {
                    "request_source": "streamlit",
                    "application_task_id": task_id,
                    "conversation_id": task.conversation_id,
                    "turn_number": task.turn_number,
                }
                if resume:
                    runtime.resume_stream(
                        task.thread_id,
                        on_state=callback,
                        metadata=metadata,
                    )
                else:
                    initial_state = {
                        "user_id": task.user_id,
                        "user_request": task.request,
                        "conversation_id": task.conversation_id,
                        "turn_number": task.turn_number,
                        "planning_mode": "initial",
                    }
                    revision_base = self._find_revision_base(task)
                    if revision_base is not None:
                        parent_snapshot = runtime.get_state(
                            revision_base.thread_id
                        )
                        parent_state = dict(parent_snapshot.values)
                        initial_state.update(
                            {
                                "planning_mode": "revision",
                                "previous_effective_request": (
                                    parent_state.get("effective_request")
                                    or revision_base.request
                                ),
                                "previous_final_answer": (
                                    revision_base.final_answer or ""
                                ),
                                "previous_subtasks": list(
                                    parent_state.get("subtasks", [])
                                ),
                                "previous_subtask_results": dict(
                                    parent_state.get("subtask_results", {})
                                ),
                            }
                        )
                    runtime.start_stream(
                        task.thread_id,
                        initial_state,
                        on_state=callback,
                        metadata=metadata,
                    )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            retryable_provider_failure = (
                isinstance(exc, RetryableProviderError)
                or "temporarily unavailable or rate-limited"
                in str(exc).casefold()
            )
            try:
                with PersistentWriterAgent(self._database_url) as runtime:
                    checkpoint_state = dict(
                        runtime.get_state(task.thread_id).values
                    )
                    has_checkpoint = bool(checkpoint_state)
                    if has_checkpoint:
                        try:
                            record_run(task.thread_id, checkpoint_state)
                        except (OSError, TypeError, ValueError):
                            pass
            except Exception:
                has_checkpoint = False
            if has_checkpoint:
                self._store.mark_interrupted(
                    task_id,
                    message=(
                        "The AI provider is temporarily rate-limited. Resume "
                        "this task after the provider limit resets."
                        if retryable_provider_failure
                        else None
                    ),
                    internal_error=error_text,
                )
            else:
                self._store.mark_failed(
                    task_id,
                    internal_error=error_text,
                )

    def _find_revision_base(self, task: TaskView) -> TaskView | None:
        """Find the nearest successful answer behind a conversation turn."""
        parent_task_id = task.parent_task_id
        visited: set[str] = set()
        while parent_task_id is not None:
            if parent_task_id in visited:
                raise RuntimeError("Conversation history contains a cycle.")
            visited.add(parent_task_id)
            parent = self._store.get(parent_task_id, user_id=task.user_id)
            if parent.status == "completed" and parent.final_answer is not None:
                return parent
            parent_task_id = parent.parent_task_id
        return None


class WriterAgentService:
    """Stable interface consumed by Streamlit."""

    def __init__(
        self,
        database_url: str,
        *,
        user_id: str = DEFAULT_USER_ID,
        max_workers: int = 1,
    ) -> None:
        self.user_id = user_id
        self._store = PostgresTaskStore(database_url)
        self._store.setup()
        self._store.recover_orphaned()
        self._runner = TaskRunner(
            self._store,
            database_url,
            max_workers=max_workers,
        )
        self._closed = False
        atexit.register(self.close)

    @classmethod
    def from_environment(cls) -> "WriterAgentService":
        database_url = getenv("DATABASE_URL", DEFAULT_LOCAL_DATABASE_URL)
        return cls(
            database_url,
            user_id=getenv("WRITER_AGENT_USER_ID", DEFAULT_USER_ID),
        )

    def create_task(
        self,
        request: str,
        *,
        idempotency_key: str | None = None,
    ) -> TaskView:
        """Validate, persist, and asynchronously start one writing task."""
        normalized = " ".join(request.split())
        if len(normalized) < 10:
            raise ValueError(
                "Describe what you want written in at least 10 characters."
            )
        if len(request) > MAX_REQUEST_LENGTH:
            raise ValueError(
                f"Writing briefs are limited to {MAX_REQUEST_LENGTH:,} characters."
            )
        task, created = self._store.create(
            user_id=self.user_id,
            request=request.strip(),
            idempotency_key=idempotency_key or str(uuid4()),
        )
        if created or (task.status == "queued" and not self._runner.is_active(task.id)):
            self._runner.submit(task.id)
        return task

    def create_follow_up(
        self,
        task_id: str,
        request: str,
        *,
        idempotency_key: str | None = None,
    ) -> TaskView:
        """Append and asynchronously execute a revision turn."""
        normalized = " ".join(request.split())
        if len(normalized) < 3:
            raise ValueError(
                "Describe the change you want in at least 3 characters."
            )
        if len(request) > MAX_REQUEST_LENGTH:
            raise ValueError(
                f"Follow-ups are limited to {MAX_REQUEST_LENGTH:,} characters."
            )
        task, created = self._store.create_follow_up(
            parent_task_id=task_id,
            user_id=self.user_id,
            request=request.strip(),
            idempotency_key=idempotency_key or str(uuid4()),
        )
        if created or (
            task.status == "queued" and not self._runner.is_active(task.id)
        ):
            self._runner.submit(task.id)
        return task

    def get_task(self, task_id: str) -> TaskView:
        return self._store.get(task_id, user_id=self.user_id)

    def list_recent_tasks(self, *, limit: int = 20) -> list[TaskSummary]:
        return self._store.list_recent(self.user_id, limit=limit)

    def list_task_versions(self, task_id: str) -> list[TaskVersionSummary]:
        return self._store.list_versions(task_id, user_id=self.user_id)

    def resume_task(self, task_id: str) -> TaskView:
        task = self.get_task(task_id)
        if task.status != "interrupted" or not task.can_resume:
            raise ValueError("This task is not waiting to be resumed.")
        self._runner.submit(task.id, resume=True)
        return task

    def retry_task(
        self,
        task_id: str,
        *,
        idempotency_key: str | None = None,
    ) -> TaskView:
        """Create and run a fresh version of a failed conversation turn."""
        task, created = self._store.create_retry(
            task_id=task_id,
            user_id=self.user_id,
            idempotency_key=idempotency_key or str(uuid4()),
        )
        if created or (
            task.status == "queued" and not self._runner.is_active(task.id)
        ):
            self._runner.submit(task.id)
        return task

    def close(self) -> None:
        if self._closed:
            return
        self._runner.close()
        self._store.close()
        self._closed = True
