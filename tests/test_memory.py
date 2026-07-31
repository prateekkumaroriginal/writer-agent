"""Deterministic tests for long-term memory behavior."""

from __future__ import annotations

from datetime import UTC, datetime
import unittest
from unittest.mock import patch

from writer_agent.memory import (
    MAX_MEMORY_CONTEXT_CHARS,
    MemoryDecisionSchema,
    MemoryMutation,
    MemoryStore,
    MemoryView,
    ProposedMemoryOperation,
    _content_fingerprint,
    manage_durable_memories,
    memory_events_for_mutations,
)
from writer_agent.parent_nodes import _initial_planning_request, initialise_task
from writer_agent.persistence import PersistentWriterAgent


class FakeExtractor:
    def __init__(self, result):
        self.result = result
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return self.result


class FakeLLM:
    def __init__(self, extractor):
        self.extractor = extractor
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.extractor


class FakeMemoryStore:
    def __init__(self, memories=None):
        self.memories = list(memories or [])
        self.added = []
        self.updated = []
        self.deleted = []

    def retrieve(self, **kwargs):
        self.retrieval = kwargs
        return list(self.memories)

    def add(self, **kwargs):
        self.added.append(kwargs)
        now = datetime.now(UTC)
        memory = MemoryView(
            id=f"added-{len(self.added)}",
            kind=kwargs["kind"],
            content=kwargs["content"],
            source_task_id=kwargs["source_task_id"],
            created_at=now,
            updated_at=now,
        )
        self.memories.append(memory)
        return memory, True

    def update(self, memory_id, **kwargs):
        self.updated.append((memory_id, kwargs))
        previous = next(item for item in self.memories if item.id == memory_id)
        current = previous.model_copy(
            update={
                "kind": kwargs["kind"],
                "content": kwargs["content"],
                "source_task_id": kwargs["source_task_id"],
                "updated_at": datetime.now(UTC),
            }
        )
        self.memories = [
            current if item.id == memory_id else item
            for item in self.memories
        ]
        return current

    def delete(self, memory_id, **kwargs):
        self.deleted.append((memory_id, kwargs))
        self.memories = [
            item for item in self.memories if item.id != memory_id
        ]


class MemoryExtractionTests(unittest.TestCase):
    def test_operation_schema_rejects_unsafe_field_combinations(self):
        with self.assertRaises(ValueError):
            ProposedMemoryOperation(
                action="delete",
                memory_id="memory-1",
                content="Replace this while deleting it.",
            )
        with self.assertRaises(ValueError):
            ProposedMemoryOperation(
                action="edit",
                memory_id="memory-1",
                kind="core",
            )

    def test_agent_adds_edits_and_deletes_existing_memories(self):
        now = datetime.now(UTC)
        british = MemoryView(
            id="memory-british",
            kind="core",
            content="The user prefers British English.",
            created_at=now,
            updated_at=now,
        )
        old_audience = MemoryView(
            id="memory-audience",
            kind="contextual",
            content="Acme writes for product leaders.",
            created_at=now,
            updated_at=now,
        )
        extractor = FakeExtractor(
            MemoryDecisionSchema(
                operations=[
                    {
                        "action": "add",
                        "kind": "core",
                        "content": "The user prefers concise writing.",
                    },
                    {
                        "action": "edit",
                        "memory_id": "memory-audience",
                        "kind": "contextual",
                        "content": "Acme writes for engineering leaders.",
                    },
                    {
                        "action": "delete",
                        "memory_id": "memory-british",
                    },
                ]
            )
        )
        store = FakeMemoryStore([british, old_audience])
        fake_llm = FakeLLM(extractor)

        with patch("writer_agent.memory.llm", fake_llm):
            mutations = manage_durable_memories(
                store,
                user_id="user-1",
                user_message=(
                    "Forget British English. Acme now writes for engineering "
                    "leaders, and always keep my writing concise."
                ),
                source_task_id="task-1",
            )

        self.assertEqual(
            [item.action for item in mutations],
            ["add", "edit", "delete"],
        )
        self.assertEqual(store.added[0]["source_task_id"], "task-1")
        self.assertEqual(store.updated[0][0], "memory-audience")
        self.assertEqual(store.deleted[0][0], "memory-british")
        self.assertIs(fake_llm.schema, MemoryDecisionSchema)
        self.assertIn(
            "id=memory-audience",
            extractor.messages[1].content,
        )

    def test_unknown_memory_id_is_not_mutated(self):
        extractor = FakeExtractor(
            MemoryDecisionSchema(
                operations=[
                    {
                        "action": "delete",
                        "memory_id": "invented-id",
                    }
                ]
            )
        )
        store = FakeMemoryStore()

        with patch("writer_agent.memory.llm", FakeLLM(extractor)):
            mutations = manage_durable_memories(
                store,
                user_id="user-1",
                user_message="Forget that preference.",
                source_task_id="task-1",
            )

        self.assertEqual(mutations, [])
        self.assertEqual(store.deleted, [])

    def test_extraction_failure_does_not_block_the_task(self):
        class FailingExtractor:
            def invoke(self, _messages):
                raise RuntimeError("provider unavailable")

        store = FakeMemoryStore()
        with patch("writer_agent.memory.llm", FakeLLM(FailingExtractor())):
            mutations = manage_durable_memories(
                store,
                user_id="user-1",
                user_message="Write a report.",
                source_task_id="task-1",
            )

        self.assertEqual(mutations, [])
        self.assertEqual(store.added, [])

    def test_successful_mutations_become_transparent_events(self):
        now = datetime.now(UTC)
        previous = MemoryView(
            id="memory-1",
            kind="core",
            content="Use British English.",
            created_at=now,
            updated_at=now,
        )
        current = previous.model_copy(
            update={"content": "Use US English."}
        )
        events = memory_events_for_mutations(
            [
                MemoryMutation(action="add", current=current),
                MemoryMutation(
                    action="edit",
                    previous=previous,
                    current=current,
                ),
                MemoryMutation(action="delete", previous=previous),
            ]
        )

        self.assertEqual(
            [event["title"] for event in events],
            ["Memory added", "Memory updated", "Memory deleted"],
        )
        self.assertEqual(events[0]["content"], "Use US English.")
        self.assertIn(
            "Previous: Use British English.",
            events[1]["details"],
        )


class MemoryContextTests(unittest.TestCase):
    def test_content_fingerprint_is_deterministic_and_content_sensitive(self):
        first = _content_fingerprint("Product strategy for Acme")
        second = _content_fingerprint("Product strategy for Acme")
        changed = _content_fingerprint("Product strategy for Beta")

        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_context_is_bounded(self):
        now = datetime.now(UTC)
        memories = [
            MemoryView(
                id=f"memory-{index}",
                kind="core",
                content="x" * 500,
                created_at=now,
                updated_at=now,
            )
            for index in range(20)
        ]
        store = MemoryStore.__new__(MemoryStore)
        store.retrieve = lambda **_kwargs: memories

        context = store.format_context(user_id="user-1", query="brief")

        self.assertLessEqual(len(context), MAX_MEMORY_CONTEXT_CHARS)
        self.assertGreater(len(context), 0)

    def test_planning_request_receives_memory_snapshot(self):
        request = _initial_planning_request(
            {
                "memory_context": (
                    "- [core] The user prefers concise British English."
                )
            },
            "Write a launch announcement.",
        )

        self.assertIn("Relevant saved memories", request)
        self.assertIn("concise British English", request)

    def test_initialisation_preserves_memory_events(self):
        event = {
            "id": "memory-event-1",
            "kind": "memory",
            "title": "Memory deleted",
            "created_at": datetime.now(UTC).isoformat(),
            "content": "Use British English.",
        }

        state = initialise_task(
            {
                "user_request": "Forget my British English preference.",
                "workflow_events": [event],
            }
        )

        self.assertEqual(state["workflow_events"], [event])

    def test_direct_runtime_activates_memory_for_python_api(self):
        class Store:
            def format_context(self, **kwargs):
                self.arguments = kwargs
                return "- [contextual] Acme writes for product leaders."

        store = Store()
        runtime = PersistentWriterAgent.__new__(PersistentWriterAgent)
        runtime._memory_store = store
        runtime._owns_memory_store = False
        runtime._database_url = "postgresql://test"

        now = datetime.now(UTC)
        mutation = MemoryMutation(
            action="add",
            current=MemoryView(
                id="memory-1",
                kind="contextual",
                content="Acme writes for product leaders.",
                created_at=now,
                updated_at=now,
            ),
        )
        with patch(
            "writer_agent.persistence.manage_durable_memories",
            return_value=[mutation],
        ) as capture:
            state = runtime._with_memory_context(
                {
                    "user_id": "user-1",
                    "user_request": "Write an Acme launch article.",
                }
            )

        capture.assert_called_once()
        self.assertIn("product leaders", state["memory_context"])
        self.assertEqual(store.arguments["user_id"], "user-1")
        self.assertEqual(
            state["workflow_events"][0]["title"],
            "Memory added",
        )


if __name__ == "__main__":
    unittest.main()
