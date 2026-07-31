"""Opt-in PostgreSQL and ChromaDB memory integration test."""

from __future__ import annotations

from os import getenv
from tempfile import TemporaryDirectory
import unittest
from uuid import uuid4

from writer_agent.memory import MemoryStore

RUN_POSTGRES_TESTS = getenv("RUN_POSTGRES_TESTS") == "1"
DATABASE_URL = getenv(
    "DATABASE_URL",
    (
        "postgresql://writer_agent:writer_agent_dev@localhost:5432/"
        "writer_agent?sslmode=disable"
    ),
)


@unittest.skipUnless(
    RUN_POSTGRES_TESTS,
    "Set RUN_POSTGRES_TESTS=1 to run the memory integration test.",
)
class MemoryPostgresTests(unittest.TestCase):
    def test_crud_and_user_scoped_contextual_retrieval(self):
        user_id = f"memory-test-{uuid4()}"
        with TemporaryDirectory() as directory:
            store = MemoryStore(DATABASE_URL, chroma_path=directory)
            try:
                store.setup()

                core, created = store.add(
                    user_id=user_id,
                    kind="core",
                    content="The user prefers British English.",
                )
                contextual, _ = store.add(
                    user_id=user_id,
                    kind="contextual",
                    content="Acme launch articles target product leaders.",
                )
                other, _ = store.add(
                    user_id=f"other-{user_id}",
                    kind="contextual",
                    content="Acme private information for another user.",
                )

                self.assertTrue(created)
                self.assertEqual(len(store.list(user_id)), 2)
                retrieved = store.retrieve(
                    user_id=user_id,
                    query="Write the Acme product launch.",
                )
                self.assertEqual(retrieved[0].id, core.id)
                self.assertIn(contextual.id, [item.id for item in retrieved])
                self.assertTrue(
                    all(
                        "another user" not in item.content
                        for item in retrieved
                    )
                )

                updated = store.update(
                    contextual.id,
                    user_id=user_id,
                    kind="contextual",
                    content=(
                        "Acme launch articles target engineering leaders."
                    ),
                )
                self.assertIn("engineering", updated.content)
                store.delete(core.id, user_id=user_id)
                self.assertEqual(len(store.list(user_id)), 1)
                store.delete(contextual.id, user_id=user_id)
                store.delete(other.id, user_id=f"other-{user_id}")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
