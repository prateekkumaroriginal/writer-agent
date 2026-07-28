"""Deterministic tests for checkpoint thread configuration."""

import unittest

from writer_agent.persistence import thread_config


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
