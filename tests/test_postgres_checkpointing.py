"""Opt-in integration test for the Postgres checkpoint backend."""

import os
import unittest
from unittest.mock import patch
from uuid import uuid4

from writer_agent.persistence import PersistentWriterAgent


def low_confidence_plan(_state):
    """Return a deterministic plan that routes the workflow to escalation."""
    return {
        "status": "planning",
        "plan": "Insufficient-confidence integration-test plan.",
        "plan_confidence": 0.4,
        "subtasks": [],
        "final_answer": None,
    }


@unittest.skipUnless(
    os.getenv("RUN_POSTGRES_TESTS") == "1",
    "Set RUN_POSTGRES_TESTS=1 to run the Postgres integration test.",
)
class PostgresCheckpointTests(unittest.TestCase):
    def test_thread_resumes_across_postgres_connections(self):
        """Resume one thread through a new Postgres connection without data loss."""
        database_url = os.environ["DATABASE_URL"]
        thread_id = f"postgres-integration-{uuid4()}"

        with patch("writer_agent.graph.supervisor_plan", low_confidence_plan):
            with PersistentWriterAgent(
                database_url,
                interrupt_after=["initialise_task"],
            ) as runtime:
                interrupted = runtime.start(
                    thread_id,
                    {"user_request": "Write a Postgres integration test report."},
                    metadata={"request_source": "postgres-integration-test"},
                )
                task_id = interrupted["task_id"]
                with self.assertRaisesRegex(ValueError, "already exists"):
                    runtime.start(
                        thread_id,
                        {"user_request": "A different request."},
                    )

            with PersistentWriterAgent(database_url) as resumed:
                completed = resumed.resume(thread_id)
                self.assertEqual(completed["task_id"], task_id)
                self.assertEqual(completed["status"], "escalated")
                self.assertGreaterEqual(len(resumed.get_history(thread_id)), 3)
