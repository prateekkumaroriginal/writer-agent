"""Deterministic tests for checkpoint thread configuration."""

import unittest
from types import SimpleNamespace

from writer_agent.persistence import PersistentWriterAgent, thread_config


class ThreadConfigTests(unittest.TestCase):
    def test_thread_config_requires_non_empty_id(self):
        """Reject thread identifiers that contain no usable characters."""
        with self.assertRaises(ValueError):
            thread_config("  ")

    def test_thread_config_includes_metadata(self):
        """Place the thread ID and caller metadata in RunnableConfig."""
        config = thread_config("run-1", {"request_source": "test"})

        self.assertEqual(config["configurable"]["thread_id"], "run-1")
        self.assertEqual(config["metadata"]["request_source"], "test")


class StreamingRuntimeTests(unittest.TestCase):
    def test_stream_reports_updates_and_authoritative_final_state(self):
        reported = []

        class FakeGraph:
            def stream(self, *_args, **_kwargs):
                yield ((), {"status": "planning"})
                yield (("specialist",), {"status": "executing"})

            def get_state(self, _config):
                return SimpleNamespace(
                    values={
                        "status": "completed",
                        "final_answer": "Finished",
                    }
                )

        runtime = PersistentWriterAgent.__new__(PersistentWriterAgent)
        runtime.graph = FakeGraph()

        result = runtime._stream(
            {"user_request": "Write"},
            config=thread_config("stream-test"),
            on_state=reported.append,
        )

        self.assertEqual(
            [state["status"] for state in reported],
            ["planning", "executing", "completed"],
        )
        self.assertEqual(result["final_answer"], "Finished")
