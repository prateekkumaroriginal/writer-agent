"""Tests for the sanitized Streamlit view-model projection."""

import unittest
from datetime import UTC, datetime

from writer_agent.ui_models import (
    projection_from_state,
    sources_from_state,
    stage_from_state,
    steps_for_stage,
    title_from_request,
    workflow_events_from_state,
)


def subtask(subtask_id: str, agent_type: str, status: str = "running"):
    return {
        "id": subtask_id,
        "agent_type": agent_type,
        "objective": "Objective",
        "expected_output": "Output",
        "tools_allowed": [],
        "review_criteria": [],
        "status": status,
        "retry_count": 0,
    }


class TaskProjectionTests(unittest.TestCase):
    def test_current_specialist_maps_to_plain_language_stage(self):
        for agent_type, expected in (
            ("research", "researching"),
            ("data", "analysing"),
            ("writing", "writing"),
        ):
            with self.subTest(agent_type=agent_type):
                state = {
                    "status": "executing",
                    "current_subtask_id": "s1",
                    "subtasks": [subtask("s1", agent_type)],
                }
                self.assertEqual(stage_from_state(state), expected)

    def test_completed_projection_exposes_reviewed_answer(self):
        state = {
            "status": "completed",
            "final_answer": "The reviewed document.",
            "plan": "Research, analyse, and write.",
            "plan_confidence": 0.9,
            "subtasks": [],
            "subtask_results": {},
            "final_review": {
                "passed": True,
                "score": 0.92,
                "issues": [],
                "action": "return",
            },
        }

        projection = projection_from_state(state)

        self.assertEqual(projection.status, "completed")
        self.assertEqual(projection.stage, "completed")
        self.assertEqual(projection.progress_current, 5)
        self.assertEqual(projection.final_answer, "The reviewed document.")
        self.assertTrue(projection.review.passed)
        self.assertTrue(all(step.status == "done" for step in projection.steps))

    def test_escalation_never_exposes_stale_final_answer(self):
        projection = projection_from_state(
            {
                "status": "escalated",
                "final_answer": "Stale content",
                "escalation_reason": "The plan was not confident enough.",
                "subtasks": [],
                "subtask_results": {},
            }
        )

        self.assertEqual(projection.status, "escalated")
        self.assertEqual(projection.stage, "attention")
        self.assertIsNone(projection.final_answer)
        self.assertIn("not confident", projection.status_message)

    def test_sources_are_http_only_and_deduplicated(self):
        state = {
            "subtask_results": {
                "s1": {
                    "sources": [
                        {
                            "title": "First",
                            "url": "https://example.com/a",
                            "snippet": "Evidence",
                        },
                        {
                            "title": "Duplicate",
                            "url": "https://example.com/a",
                        },
                        {
                            "title": "Unsafe",
                            "url": "javascript:alert(1)",
                        },
                    ]
                }
            }
        }

        sources = sources_from_state(state)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].title, "First")

    def test_stage_steps_mark_only_one_current_step(self):
        steps = steps_for_stage("analysing")

        self.assertEqual([item.status for item in steps], [
            "done",
            "done",
            "current",
            "upcoming",
            "upcoming",
        ])

    def test_long_title_is_bounded(self):
        title = title_from_request("word " * 30)

        self.assertLessEqual(len(title), 58)
        self.assertTrue(title.endswith("…"))

    def test_workflow_events_are_validated_and_deduplicated(self):
        events = workflow_events_from_state(
            {
                "workflow_events": [
                    {
                        "id": "event-1",
                        "kind": "search",
                        "title": "Web search",
                        "created_at": "2026-07-31T08:30:00+00:00",
                        "content": "relevant query",
                    },
                    {
                        "id": "event-1",
                        "kind": "search",
                        "title": "Duplicate",
                        "created_at": "2026-07-31T08:31:00+00:00",
                    },
                    {
                        "id": "event-2",
                        "kind": "hidden_reasoning",
                        "title": "Unsafe internal event",
                        "created_at": "2026-07-31T08:32:00+00:00",
                    },
                    {
                        "id": "event-3",
                        "kind": "plan",
                        "title": "Event without required timestamp",
                    },
                ]
            }
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].content, "relevant query")

    def test_memory_mutation_event_is_exposed(self):
        events = workflow_events_from_state(
            {
                "workflow_events": [
                    {
                        "id": "memory-event-1",
                        "kind": "memory",
                        "title": "Memory updated",
                        "created_at": "2026-07-31T08:30:00+00:00",
                        "content": "Use US English.",
                        "details": ["Previous: Use British English."],
                        "decision": "edit",
                    }
                ]
            }
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "memory")
        self.assertIn("US English", events[0].content)

    def test_workflow_event_exposes_safe_execution_metadata(self):
        timestamp = datetime(2026, 7, 31, 8, 30, tzinfo=UTC)
        events = workflow_events_from_state(
            {
                "workflow_events": [
                    {
                        "id": "research-event-1",
                        "kind": "research",
                        "title": "Research response",
                        "created_at": timestamp.isoformat(),
                        "subtask_name": "Research task",
                        "objective": "Find supporting evidence.",
                        "agent": "Research agent",
                        "review_criteria": ["Uses authoritative sources"],
                        "attempt": 2,
                        "retry_count": 1,
                        "sources": [
                            {
                                "title": "Evidence",
                                "url": "https://example.com/evidence",
                                "snippet": "Relevant evidence.",
                            },
                            {
                                "title": "Unsafe",
                                "url": "javascript:alert(1)",
                            },
                        ],
                    }
                ]
            }
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.created_at, timestamp)
        self.assertEqual(event.subtask_name, "Research task")
        self.assertEqual(event.objective, "Find supporting evidence.")
        self.assertEqual(event.agent, "Research agent")
        self.assertEqual(event.review_criteria, ["Uses authoritative sources"])
        self.assertEqual(event.attempt, 2)
        self.assertEqual(event.retry_count, 1)
        self.assertEqual(len(event.sources), 1)
        self.assertEqual(event.sources[0].title, "Evidence")


if __name__ == "__main__":
    unittest.main()
