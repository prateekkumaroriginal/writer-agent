"""Tests for the sanitized Streamlit view-model projection."""

import unittest

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
                        "content": "relevant query",
                    },
                    {
                        "id": "event-1",
                        "kind": "search",
                        "title": "Duplicate",
                    },
                    {
                        "id": "event-2",
                        "kind": "hidden_reasoning",
                        "title": "Unsafe internal event",
                    },
                ]
            }
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].content, "relevant query")


if __name__ == "__main__":
    unittest.main()
