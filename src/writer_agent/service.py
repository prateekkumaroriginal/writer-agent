"""Application service and background runner for the Streamlit UI."""

from __future__ import annotations

import atexit
from concurrent.futures import Future, ThreadPoolExecutor
from os import getenv
from threading import Lock
from uuid import uuid4

from writer_agent.persistence import PersistentWriterAgent
from writer_agent.provider_errors import RetryableProviderError
from writer_agent.task_store import PostgresTaskStore
from writer_agent.ui_models import TaskSummary, TaskView

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
                callback = lambda state: self._store.update_from_state(
                    task_id, state
                )
                metadata = {
                    "request_source": "streamlit",
                    "application_task_id": task_id,
                }
                if resume:
                    runtime.resume_stream(
                        task.thread_id,
                        on_state=callback,
                        metadata=metadata,
                    )
                else:
                    runtime.start_stream(
                        task.thread_id,
                        {
                            "user_id": task.user_id,
                            "user_request": task.request,
                        },
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
                    has_checkpoint = bool(
                        runtime.get_state(task.thread_id).values
                    )
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

    def get_task(self, task_id: str) -> TaskView:
        return self._store.get(task_id, user_id=self.user_id)

    def list_recent_tasks(self, *, limit: int = 20) -> list[TaskSummary]:
        return self._store.list_recent(self.user_id, limit=limit)

    def resume_task(self, task_id: str) -> TaskView:
        task = self.get_task(task_id)
        if task.status != "interrupted" or not task.can_resume:
            raise ValueError("This task is not waiting to be resumed.")
        self._runner.submit(task.id, resume=True)
        return task

    def close(self) -> None:
        if self._closed:
            return
        self._runner.close()
        self._store.close()
        self._closed = True
