"""Deterministic tests for Phase 3A multi-turn revision behavior."""

import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from writer_agent.graph import build_supervisor_graph
from writer_agent.parent_nodes import supervisor_plan
from writer_agent.schemas import (
    FinalReviewSchema,
    PlannedSubtaskSchema,
    ReviewDecisionSchema,
    SupervisorPlanSchema,
    SupervisorRevisionPlanSchema,
    WritingResponseSchema,
)


class FakeStructuredRunner:
    def __init__(self, response, captured_messages):
        self.response = response
        self.captured_messages = captured_messages

    def invoke(self, messages):
        self.captured_messages.extend(messages)
        if isinstance(self.response, list):
            return self.response.pop(0)
        return self.response


class FakeLLM:
    def __init__(self, responses):
        self.responses = responses
        self.captured_messages = []

    def with_structured_output(self, schema):
        if schema not in self.responses:
            raise AssertionError(f"No fake response configured for {schema}.")
        return FakeStructuredRunner(
            self.responses[schema],
            self.captured_messages,
        )


def research_subtask():
    return {
        "id": "s1",
        "agent_type": "research",
        "objective": "Explain React",
        "expected_output": "Research notes",
        "tools_allowed": ["web_search"],
        "review_criteria": ["Accurate"],
        "status": "passed",
        "retry_count": 0,
    }


def research_result():
    return {
        "subtask_id": "s1",
        "agent_type": "research",
        "output": {
            "summary": "React is a component-based UI library.",
            "findings": ["Components compose user interfaces."],
            "uncertainties": [],
        },
        "confidence": 0.9,
        "sources": [
            {
                "title": "React",
                "url": "https://react.dev/",
                "snippet": "The library for web and native user interfaces.",
            }
        ],
        "errors": [],
    }


def writing_only_revision(*, confidence: float = 0.9):
    return SupervisorRevisionPlanSchema(
        plan="Reuse the existing explanation and add a practical example.",
        plan_confidence=confidence,
        intent="extend",
        effective_request=(
            "Explain React and include a practical component example."
        ),
        reuse_research=True,
        reuse_data=False,
        reuse_previous_answer=True,
        subtasks=[
            PlannedSubtaskSchema(
                agent_type="writing",
                objective="Revise the explanation to add an example.",
                expected_output="A complete revised explanation.",
                review_criteria=["Includes a correct practical example"],
            )
        ],
    )


class RevisionSchemaTests(unittest.TestCase):
    def test_revision_plan_must_end_with_writing(self):
        with self.assertRaises(ValidationError):
            SupervisorRevisionPlanSchema(
                plan="Research only.",
                plan_confidence=0.9,
                intent="verify",
                effective_request="Verify the current React information.",
                reuse_research=False,
                reuse_data=False,
                reuse_previous_answer=False,
                subtasks=[
                    PlannedSubtaskSchema(
                        agent_type="research",
                        objective="Verify React",
                        expected_output="Research",
                        review_criteria=[],
                    )
                ],
            )


class SupervisorRevisionTests(unittest.TestCase):
    def revision_state(self, user_request="Also include an example."):
        return {
            "planning_mode": "revision",
            "user_request": user_request,
            "previous_effective_request": "Explain React.",
            "previous_final_answer": "React is a UI library.",
            "previous_subtasks": [research_subtask()],
            "previous_subtask_results": {"s1": research_result()},
            "replan_feedback": [],
        }

    def test_supervisor_reuses_research_and_schedules_only_writing(self):
        fake_llm = FakeLLM(
            {SupervisorRevisionPlanSchema: writing_only_revision()}
        )

        with patch("writer_agent.parent_nodes.llm", fake_llm):
            planned = supervisor_plan(self.revision_state())

        self.assertEqual(
            [item["agent_type"] for item in planned["subtasks"]],
            ["research", "writing"],
        )
        self.assertEqual(planned["subtasks"][0]["status"], "passed")
        self.assertTrue(planned["subtasks"][0]["id"].startswith("reused-"))
        self.assertEqual(planned["subtasks"][1]["status"], "pending")
        self.assertEqual(planned["reused_agent_types"], ["research"])
        self.assertTrue(planned["reuse_previous_answer"])
        self.assertEqual(
            planned["effective_request"],
            "Explain React and include a practical component example.",
        )

    def test_freshness_guard_forces_research(self):
        fake_llm = FakeLLM(
            {SupervisorRevisionPlanSchema: writing_only_revision()}
        )

        with patch("writer_agent.parent_nodes.llm", fake_llm):
            planned = supervisor_plan(
                self.revision_state("Use the latest React information.")
            )

        new_types = [
            item["agent_type"]
            for item in planned["subtasks"]
            if not item["id"].startswith("reused-")
        ]
        self.assertEqual(new_types, ["research", "writing"])
        self.assertEqual(planned["reused_agent_types"], [])

    def test_current_paragraph_edit_does_not_force_research(self):
        fake_llm = FakeLLM(
            {SupervisorRevisionPlanSchema: writing_only_revision()}
        )

        with patch("writer_agent.parent_nodes.llm", fake_llm):
            planned = supervisor_plan(
                self.revision_state("Shorten the current paragraph.")
            )

        new_types = [
            item["agent_type"]
            for item in planned["subtasks"]
            if not item["id"].startswith("reused-")
        ]
        self.assertEqual(new_types, ["writing"])

    def test_low_confidence_revision_falls_back_to_fresh_plan(self):
        fallback = SupervisorPlanSchema(
            plan="Research and rewrite from a fresh brief.",
            plan_confidence=0.85,
            subtasks=[
                PlannedSubtaskSchema(
                    agent_type="research",
                    objective="Research the replacement request.",
                    expected_output="Current research.",
                    review_criteria=["Relevant"],
                ),
                PlannedSubtaskSchema(
                    agent_type="writing",
                    objective="Write the replacement answer.",
                    expected_output="Complete answer.",
                    review_criteria=["Complete"],
                ),
            ],
        )
        fake_llm = FakeLLM(
            {
                SupervisorRevisionPlanSchema: writing_only_revision(
                    confidence=0.4
                ),
                SupervisorPlanSchema: fallback,
            }
        )

        with patch("writer_agent.parent_nodes.llm", fake_llm):
            planned = supervisor_plan(self.revision_state())

        self.assertEqual(planned["revision_intent"], "replace")
        self.assertEqual(planned["reused_agent_types"], [])
        self.assertFalse(planned["reuse_previous_answer"])
        self.assertEqual(
            [item["agent_type"] for item in planned["subtasks"]],
            ["research", "writing"],
        )


class FullRevisionGraphTests(unittest.TestCase):
    def test_revision_reuses_parent_research_and_returns_complete_answer(self):
        fake_llm = FakeLLM(
            {
                SupervisorRevisionPlanSchema: writing_only_revision(),
                WritingResponseSchema: WritingResponseSchema(
                    content=(
                        "React is a component-based UI library. "
                        "For example: function Hello() { return <h1>Hello</h1>; }"
                    ),
                    confidence=0.95,
                ),
                ReviewDecisionSchema: ReviewDecisionSchema(
                    passed=True,
                    score=0.95,
                    issues=[],
                    action="pass",
                ),
                FinalReviewSchema: FinalReviewSchema(
                    passed=True,
                    score=0.95,
                    issues=[],
                    action="return",
                ),
            }
        )

        with (
            patch("writer_agent.parent_nodes.llm", fake_llm),
            patch("writer_agent.specialist_nodes.llm", fake_llm),
        ):
            result = build_supervisor_graph().invoke(
                {
                    "planning_mode": "revision",
                    "conversation_id": "conversation-1",
                    "turn_number": 2,
                    "user_request": "Also include an example.",
                    "previous_effective_request": "Explain React.",
                    "previous_final_answer": "React is a UI library.",
                    "previous_subtasks": [research_subtask()],
                    "previous_subtask_results": {"s1": research_result()},
                }
            )

        self.assertEqual(result["status"], "completed")
        self.assertIn("function Hello", result["final_answer"])
        self.assertEqual(result["reused_agent_types"], ["research"])
        writing_prompt = next(
            message.content
            for message in fake_llm.captured_messages
            if isinstance(message, HumanMessage)
            and "Previous reviewed answer:" in message.content
        )
        self.assertIn("React is a UI library.", writing_prompt)
        self.assertIn("React is a component-based UI library.", writing_prompt)


if __name__ == "__main__":
    unittest.main()
