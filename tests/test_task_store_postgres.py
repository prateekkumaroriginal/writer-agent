"""Opt-in integration tests for the Streamlit task index."""

import os
import unittest
from datetime import UTC, datetime
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
                            "created_at": datetime.now(UTC).isoformat(),
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

            follow_up, follow_up_created = store.create_follow_up(
                parent_task_id=task.id,
                user_id="task-store-integration",
                request="Also include a concrete example.",
                idempotency_key=f"{key}-follow-up",
            )
            self.assertTrue(follow_up_created)
            self.assertEqual(follow_up.parent_task_id, task.id)
            self.assertEqual(follow_up.conversation_id, task.conversation_id)
            self.assertEqual(follow_up.turn_number, 2)
            self.assertEqual(
                [version.turn_number for version in store.list_versions(
                    follow_up.id,
                    user_id="task-store-integration",
                )],
                [1, 2],
            )
            self.assertEqual(
                store.list_recent("task-store-integration", limit=1)[0].id,
                follow_up.id,
            )

            store.mark_failed(follow_up.id)
            retry, retry_created = store.create_retry(
                task_id=follow_up.id,
                user_id="task-store-integration",
                idempotency_key=f"{key}-retry",
            )
            self.assertTrue(retry_created)
            self.assertEqual(retry.parent_task_id, follow_up.id)
            self.assertEqual(retry.request, follow_up.request)
            self.assertEqual(retry.conversation_id, task.conversation_id)
            self.assertEqual(retry.turn_number, 3)
            self.assertEqual(
                [
                    version.turn_number
                    for version in store.list_versions(
                        retry.id,
                        user_id="task-store-integration",
                    )
                ],
                [1, 2, 3],
            )
        finally:
            store.close()
            if task_id is not None:
                with psycopg.connect(
                    database_url,
                    autocommit=True,
                ) as connection:
                    connection.execute(
                        "DELETE FROM writer_tasks WHERE conversation_id = %s",
                        (task_id,),
                    )


if __name__ == "__main__":
    unittest.main()
