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

    def get(self, task_id):
        if task_id != self.task.id:
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


if __name__ == "__main__":
    unittest.main()
