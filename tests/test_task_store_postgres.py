"""Opt-in integration tests for the Streamlit task index."""

import os
import unittest
from uuid import uuid4

import psycopg

from writer_agent.task_store import PostgresTaskStore


@unittest.skipUnless(
    os.getenv("RUN_POSTGRES_TESTS") == "1",
    "Set RUN_POSTGRES_TESTS=1 to run the task-store integration test.",
)
class TaskStorePostgresTests(unittest.TestCase):
    def test_task_is_idempotent_persisted_and_projected(self):
        database_url = os.environ["DATABASE_URL"]
        store = PostgresTaskStore(database_url)
        store.setup()
        key = f"task-store-integration-{uuid4()}"
        task_id = None
        try:
            task, created = store.create(
                user_id="task-store-integration",
                request="Write a durable task-store integration report.",
                idempotency_key=key,
            )
            task_id = task.id
            duplicate, duplicate_created = store.create(
                user_id="task-store-integration",
                request="This duplicate must not create another task.",
                idempotency_key=key,
            )

            self.assertTrue(created)
            self.assertFalse(duplicate_created)
            self.assertEqual(duplicate.id, task.id)

            completed = store.update_from_state(
                task.id,
                {
                    "status": "completed",
                    "final_answer": "Durable final document.",
                    "plan": "Write and review.",
                    "plan_confidence": 0.9,
                    "subtasks": [],
                    "subtask_results": {},
                    "workflow_events": [
                        {
                            "id": "event-1",
                            "kind": "plan",
                            "title": "Initial plan",
                            "content": "Write and review.",
                        }
                    ],
                    "final_review": {
                        "passed": True,
                        "score": 0.9,
                        "issues": [],
                        "action": "return",
                    },
                },
            )

            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.final_answer, "Durable final document.")
            self.assertEqual(
                completed.workflow_events[0].title,
                "Initial plan",
            )
            self.assertEqual(
                store.list_recent("task-store-integration", limit=1)[0].id,
                task.id,
            )
        finally:
            store.close()
            if task_id is not None:
                with psycopg.connect(
                    database_url,
                    autocommit=True,
                ) as connection:
                    connection.execute(
                        "DELETE FROM writer_tasks WHERE id = %s",
                        (task_id,),
                    )


if __name__ == "__main__":
    unittest.main()
