"""Tests for background execution outside Streamlit reruns."""

from datetime import UTC, datetime
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from writer_agent.service import TaskRunner, WriterAgentService
from writer_agent.provider_errors import RetryableProviderError
from writer_agent.ui_models import TaskView, steps_for_stage


def queued_task() -> TaskView:
    now = datetime.now(UTC)
    return TaskView(
        id="00000000-0000-0000-0000-000000000001",
        thread_id="thread-1",
        user_id="local-user",
        title="Write a useful report",
        request="Write a useful report about testing.",
        status="queued",
        stage="queued",
        status_message="Waiting to start…",
        progress_current=0,
        steps=steps_for_stage("queued"),
        created_at=now,
        updated_at=now,
    )


class FakeStore:
    def __init__(self):
        self.task = queued_task()
        self.states = []
        self.failed = []
        self.interrupted = []

    def claim(self, task_id, *, resume):
        return task_id == self.task.id

    def get(self, task_id, *, user_id=None):
        if task_id != self.task.id:
            raise KeyError(task_id)
        if user_id is not None and user_id != self.task.user_id:
            raise KeyError(task_id)
        return self.task

    def update_from_state(self, task_id, state):
        self.states.append((task_id, state))

    def mark_failed(self, task_id, **kwargs):
        self.failed.append((task_id, kwargs))

    def mark_interrupted(self, task_id, **kwargs):
        self.interrupted.append((task_id, kwargs))


class SuccessfulRuntime:
    def __init__(self, _database_url):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def start_stream(self, _thread_id, _state, *, on_state, metadata):
        final = {
            "status": "completed",
            "final_answer": "Finished",
            "subtasks": [],
            "subtask_results": {},
        }
        on_state(final)
        return final


class RevisionRuntime(SuccessfulRuntime):
    started_state = None

    def get_state(self, thread_id):
        if thread_id != "thread-parent":
            return SimpleNamespace(values={})
        return SimpleNamespace(
            values={
                "effective_request": "Explain React.",
                "subtasks": [{"id": "s1", "status": "passed"}],
                "subtask_results": {"s1": {"output": {"summary": "React"}}},
            }
        )

    def start_stream(self, _thread_id, state, *, on_state, metadata):
        type(self).started_state = state
        return super().start_stream(
            _thread_id,
            state,
            on_state=on_state,
            metadata=metadata,
        )


class FailingRuntime:
    opens = 0

    def __init__(self, _database_url):
        type(self).opens += 1

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def start_stream(self, *_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    def get_state(self, _thread_id):
        return SimpleNamespace(values={})


class RateLimitedRuntime(FailingRuntime):
    def start_stream(self, *_args, **_kwargs):
        raise RetryableProviderError("Provider rate-limited.")

    def get_state(self, _thread_id):
        return SimpleNamespace(values={"status": "planning"})


class TaskRunnerTests(unittest.TestCase):
    def setUp(self):
        self.run_record_patcher = patch("writer_agent.service.record_run")
        self.run_record_patcher.start()
        self.addCleanup(self.run_record_patcher.stop)

    def test_runner_records_streamed_state(self):
        store = FakeStore()
        runner = TaskRunner(store, "postgresql://test")

        with patch(
            "writer_agent.service.PersistentWriterAgent",
            SuccessfulRuntime,
        ):
            runner._run(store.task.id, False)
        runner.close()

        self.assertEqual(len(store.states), 1)
        self.assertEqual(store.states[0][1]["status"], "completed")
        self.assertFalse(store.failed)

    def test_revision_runner_loads_parent_checkpoint_artifacts(self):
        now = datetime.now(UTC)
        parent = TaskView(
            id="00000000-0000-0000-0000-000000000010",
            thread_id="thread-parent",
            user_id="local-user",
            conversation_id="00000000-0000-0000-0000-000000000100",
            turn_number=1,
            title="Explain React",
            request="Explain React.",
            status="completed",
            stage="completed",
            status_message="Your reviewed document is ready.",
            progress_current=5,
            steps=steps_for_stage("completed"),
            final_answer="React is a UI library.",
            created_at=now,
            updated_at=now,
        )
        child = TaskView(
            id="00000000-0000-0000-0000-000000000011",
            thread_id="thread-child",
            user_id="local-user",
            conversation_id=parent.conversation_id,
            parent_task_id=parent.id,
            turn_number=2,
            title="Explain React",
            request="Also include an example.",
            status="queued",
            stage="queued",
            status_message="Waiting to start…",
            progress_current=0,
            steps=steps_for_stage("queued"),
            created_at=now,
            updated_at=now,
        )

        class RevisionStore(FakeStore):
            def __init__(self):
                super().__init__()
                self.task = child

            def get(self, task_id, *, user_id=None):
                if task_id == child.id:
                    task = child
                elif task_id == parent.id:
                    task = parent
                else:
                    raise KeyError(task_id)
                if user_id is not None and user_id != task.user_id:
                    raise KeyError(task_id)
                return task

        store = RevisionStore()
        runner = TaskRunner(store, "postgresql://test")
        RevisionRuntime.started_state = None

        with patch(
            "writer_agent.service.PersistentWriterAgent",
            RevisionRuntime,
        ):
            runner._run(child.id, False)
        runner.close()

        state = RevisionRuntime.started_state
        self.assertEqual(state["planning_mode"], "revision")
        self.assertEqual(state["previous_effective_request"], "Explain React.")
        self.assertEqual(
            state["previous_final_answer"],
            "React is a UI library.",
        )
        self.assertEqual(state["previous_subtasks"][0]["id"], "s1")

    def test_revision_retry_uses_nearest_completed_ancestor(self):
        now = datetime.now(UTC)
        completed = queued_task().model_copy(
            update={
                "id": "00000000-0000-0000-0000-000000000020",
                "thread_id": "thread-parent",
                "conversation_id": (
                    "00000000-0000-0000-0000-000000000100"
                ),
                "status": "completed",
                "stage": "completed",
                "status_message": "Your reviewed document is ready.",
                "progress_current": 5,
                "steps": steps_for_stage("completed"),
                "final_answer": "React is a UI library.",
                "created_at": now,
                "updated_at": now,
            }
        )
        failed_revision = queued_task().model_copy(
            update={
                "id": "00000000-0000-0000-0000-000000000021",
                "thread_id": "thread-failed-revision",
                "conversation_id": completed.conversation_id,
                "parent_task_id": completed.id,
                "turn_number": 2,
                "request": "Remove the word count.",
                "status": "escalated",
                "stage": "attention",
                "status_message": "Revision escalated.",
                "created_at": now,
                "updated_at": now,
            }
        )
        retry = queued_task().model_copy(
            update={
                "id": "00000000-0000-0000-0000-000000000022",
                "thread_id": "thread-retry",
                "conversation_id": completed.conversation_id,
                "parent_task_id": failed_revision.id,
                "turn_number": 3,
                "request": failed_revision.request,
                "created_at": now,
                "updated_at": now,
            }
        )

        class RetryStore(FakeStore):
            def __init__(self):
                super().__init__()
                self.task = retry
                self.tasks = {
                    completed.id: completed,
                    failed_revision.id: failed_revision,
                    retry.id: retry,
                }

            def get(self, task_id, *, user_id=None):
                task = self.tasks[task_id]
                if user_id is not None and user_id != task.user_id:
                    raise KeyError(task_id)
                return task

        store = RetryStore()
        runner = TaskRunner(store, "postgresql://test")
        RevisionRuntime.started_state = None

        with patch(
            "writer_agent.service.PersistentWriterAgent",
            RevisionRuntime,
        ):
            runner._run(retry.id, False)
        runner.close()

        state = RevisionRuntime.started_state
        self.assertEqual(state["planning_mode"], "revision")
        self.assertEqual(state["user_request"], "Remove the word count.")
        self.assertEqual(
            state["previous_final_answer"],
            "React is a UI library.",
        )

    def test_initial_task_retry_without_successful_ancestor_stays_initial(self):
        now = datetime.now(UTC)
        failed_initial = queued_task().model_copy(
            update={
                "id": "00000000-0000-0000-0000-000000000030",
                "thread_id": "thread-failed-initial",
                "conversation_id": (
                    "00000000-0000-0000-0000-000000000030"
                ),
                "status": "failed",
                "stage": "failed",
                "status_message": "Failed.",
                "created_at": now,
                "updated_at": now,
            }
        )
        retry = queued_task().model_copy(
            update={
                "id": "00000000-0000-0000-0000-000000000031",
                "thread_id": "thread-initial-retry",
                "conversation_id": failed_initial.conversation_id,
                "parent_task_id": failed_initial.id,
                "turn_number": 2,
                "created_at": now,
                "updated_at": now,
            }
        )

        class RetryStore(FakeStore):
            def __init__(self):
                super().__init__()
                self.task = retry
                self.tasks = {
                    failed_initial.id: failed_initial,
                    retry.id: retry,
                }

            def get(self, task_id, *, user_id=None):
                task = self.tasks[task_id]
                if user_id is not None and user_id != task.user_id:
                    raise KeyError(task_id)
                return task

        class CapturingRuntime(SuccessfulRuntime):
            started_state = None

            def start_stream(self, thread_id, state, *, on_state, metadata):
                type(self).started_state = state
                return super().start_stream(
                    thread_id,
                    state,
                    on_state=on_state,
                    metadata=metadata,
                )

        store = RetryStore()
        runner = TaskRunner(store, "postgresql://test")
        with patch(
            "writer_agent.service.PersistentWriterAgent",
            CapturingRuntime,
        ):
            runner._run(retry.id, False)
        runner.close()

        self.assertEqual(
            CapturingRuntime.started_state["planning_mode"],
            "initial",
        )
        self.assertNotIn(
            "previous_final_answer",
            CapturingRuntime.started_state,
        )

    def test_uncheckpointed_exception_becomes_safe_failure(self):
        store = FakeStore()
        runner = TaskRunner(store, "postgresql://test")
        FailingRuntime.opens = 0

        with patch(
            "writer_agent.service.PersistentWriterAgent",
            FailingRuntime,
        ):
            runner._run(store.task.id, False)
        runner.close()

        self.assertEqual(len(store.failed), 1)
        self.assertIn(
            "RuntimeError",
            store.failed[0][1]["internal_error"],
        )
        self.assertFalse(store.interrupted)

    def test_rate_limit_with_checkpoint_becomes_resumable(self):
        store = FakeStore()
        runner = TaskRunner(store, "postgresql://test")

        with patch(
            "writer_agent.service.PersistentWriterAgent",
            RateLimitedRuntime,
        ):
            runner._run(store.task.id, False)
        runner.close()

        self.assertEqual(len(store.interrupted), 1)
        self.assertIn(
            "rate-limited",
            store.interrupted[0][1]["message"],
        )
        self.assertFalse(store.failed)


class ServiceValidationTests(unittest.TestCase):
    def setUp(self):
        self.service = WriterAgentService.__new__(WriterAgentService)
        self.service.user_id = "local-user"

    def test_short_brief_is_rejected_before_storage(self):
        with self.assertRaisesRegex(ValueError, "at least 10"):
            self.service.create_task("Too short")

    def test_oversized_brief_is_rejected_before_storage(self):
        with self.assertRaisesRegex(ValueError, "8,000"):
            self.service.create_task("x" * 8_001)

    def test_short_follow_up_is_rejected_before_storage(self):
        with self.assertRaisesRegex(ValueError, "at least 3"):
            self.service.create_follow_up("task-1", "x")

    def test_follow_up_is_persisted_and_scheduled(self):
        task = queued_task().model_copy(
            update={
                "parent_task_id": "parent-task",
                "turn_number": 2,
            }
        )

        class Store:
            def create_follow_up(self, **kwargs):
                self.arguments = kwargs
                return task, True

        class Runner:
            def __init__(self):
                self.submitted = []

            def submit(self, task_id):
                self.submitted.append(task_id)

        self.service._store = Store()
        self.service._runner = Runner()

        result = self.service.create_follow_up(
            "parent-task",
            "Also include an example.",
            idempotency_key="follow-up-key",
        )

        self.assertEqual(result.id, task.id)
        self.assertEqual(
            self.service._store.arguments["parent_task_id"],
            "parent-task",
        )
        self.assertEqual(self.service._runner.submitted, [task.id])

    def test_retry_is_persisted_and_scheduled(self):
        task = queued_task().model_copy(
            update={
                "parent_task_id": "failed-task",
                "turn_number": 3,
            }
        )

        class Store:
            def create_retry(self, **kwargs):
                self.arguments = kwargs
                return task, True

        class Runner:
            def __init__(self):
                self.submitted = []

            def submit(self, task_id):
                self.submitted.append(task_id)

        self.service._store = Store()
        self.service._runner = Runner()

        result = self.service.retry_task(
            "failed-task",
            idempotency_key="retry-key",
        )

        self.assertEqual(result.id, task.id)
        self.assertEqual(
            self.service._store.arguments["task_id"],
            "failed-task",
        )
        self.assertEqual(self.service._runner.submitted, [task.id])


if __name__ == "__main__":
    unittest.main()
