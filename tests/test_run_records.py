"""Tests for durable JSON run recording."""

import json
import tempfile
import unittest
from pathlib import Path

from writer_agent.run_records import record_run


class RunRecordTests(unittest.TestCase):
    def test_record_is_named_for_thread_and_replaced_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = record_run(
                "writer-thread-1",
                {"status": "planning", "thread_id": "stale"},
                output_dir=directory,
            )
            record_run(
                "writer-thread-1",
                {"status": "completed"},
                output_dir=directory,
            )

            self.assertEqual(path.name, "writer-thread-1.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["thread_id"], "writer-thread-1")
            self.assertEqual(payload["status"], "completed")
            self.assertFalse(
                (Path(directory) / ".writer-thread-1.tmp").exists()
            )

    def test_unsafe_thread_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                record_run("../escape", {}, output_dir=directory)


if __name__ == "__main__":
    unittest.main()
