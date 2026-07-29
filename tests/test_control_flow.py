"""Deterministic tests for retry and replanning state transitions."""

import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from writer_agent.graph import build_specialist_graph, build_supervisor_graph
from writer_agent.parent_nodes import (
    final_review,
    get_latest_writing_content,
    prepare_final_retry,
    prepare_replan,
    route_after_final_review,
    route_after_specialists,
    supervisor_plan,
)
from writer_agent.schemas import (
    DataResponseSchema,
    FinalReviewSchema,
    PlannedSubtaskSchema,
    ReviewDecisionSchema,
    SupervisorPlanSchema,
    WritingResponseSchema,
)
from writer_agent.specialist_nodes import request_replan, writing_agent


class FakeStructuredRunner:
    def __init__(self, response, captured_messages=None):
        """Store a deterministic structured response and optional message sink."""
        self.response = response
        self.captured_messages = captured_messages

    def invoke(self, messages):
        """Capture invocation messages and return the next fake response."""
        if self.captured_messages is not None:
            self.captured_messages.extend(messages)
        if isinstance(self.response, list):
            if not self.response:
                raise AssertionError("No fake response remains for this schema.")
            return self.response.pop(0)
        return self.response


class FakeLLM:
    def __init__(self, responses, captured_messages=None):
        """Store fake responses keyed by structured-output schema."""
        self.responses = responses
        self.captured_messages = captured_messages

    def with_structured_output(self, schema):
        """Create a fake runner for the requested structured-output schema."""
        return FakeStructuredRunner(
            self.responses[schema],
            self.captured_messages,
        )


def writing_result(subtask_id: str, content: str = "Original writing output"):
    """Build a successful writing result for control-flow tests."""
    return {
        "subtask_id": subtask_id,
        "agent_type": "writing",
        "output": {"content": content},
        "confidence": 0.8,
        "sources": [],
        "errors": [],
    }


class FinalRetryTests(unittest.TestCase):
    def test_final_retry_reopens_only_final_writing_subtask(self):
        """Verify final retry reopens writing while preserving passed research."""
        state = {
            "subtasks": [
                {
                    "id": "s1",
                    "agent_type": "research",
                    "objective": "Research",
                    "expected_output": "Notes",
                    "tools_allowed": ["web_search"],
                    "review_criteria": [],
                    "status": "passed",
                    "retry_count": 0,
                },
                {
                    "id": "s2",
                    "agent_type": "writing",
                    "objective": "Write",
                    "expected_output": "Article",
                    "tools_allowed": [],
                    "review_criteria": [],
                    "status": "passed",
                    "retry_count": 0,
                },
            ],
            "subtask_results": {
                "s1": {
                    "subtask_id": "s1",
                    "agent_type": "research",
                    "output": {"summary": "Research"},
                    "confidence": 0.9,
                    "sources": [],
                    "errors": [],
                },
                "s2": writing_result("s2"),
            },
            "final_review": {
                "passed": False,
                "score": 0.4,
                "issues": ["Expand the conclusion."],
                "action": "retry",
            },
            "final_retry_count": 0,
        }

        updated = prepare_final_retry(state)

        self.assertEqual(updated["subtasks"][0]["status"], "passed")
        self.assertEqual(updated["subtasks"][1]["status"], "pending")
        self.assertNotIn("s2", updated["subtask_results"])
        self.assertEqual(updated["revision_feedback"], ["Expand the conclusion."])
        self.assertEqual(updated["final_retry_count"], 1)
        self.assertEqual(updated["status"], "executing")

    def test_final_writing_selection_uses_subtask_order_not_dict_order(self):
        """Verify final writing selection follows planned subtask order."""
        state = {
            "subtasks": [
                {
                    "id": "s1",
                    "agent_type": "writing",
                    "objective": "Draft",
                    "expected_output": "Draft",
                    "tools_allowed": [],
                    "review_criteria": [],
                    "status": "passed",
                    "retry_count": 0,
                },
                {
                    "id": "s2",
                    "agent_type": "writing",
                    "objective": "Final",
                    "expected_output": "Final",
                    "tools_allowed": [],
                    "review_criteria": [],
                    "status": "passed",
                    "retry_count": 0,
                },
            ],
            "subtask_results": {
                "s2": writing_result("s2", "Semantically final result"),
                "s1": writing_result("s1", "Inserted last but not final"),
            },
        }

        self.assertEqual(
            get_latest_writing_content(state),
            "Semantically final result",
        )

    def test_final_retry_limit_converts_retry_to_escalation(self):
        """Verify an exhausted final retry request becomes escalation."""
        state = {
            "subtasks": [
                {
                    "id": "s1",
                    "agent_type": "writing",
                    "objective": "Write",
                    "expected_output": "Article",
                    "tools_allowed": [],
                    "review_criteria": [],
                    "status": "passed",
                    "retry_count": 0,
                }
            ],
            "subtask_results": {"s1": writing_result("s1")},
            "final_retry_count": 2,
            "max_final_retries": 2,
            "replan_count": 0,
            "max_replans": 1,
        }
        response = FinalReviewSchema(
            passed=False,
            score=0.4,
            issues=["Still incomplete."],
            action="retry",
        )

        with patch(
            "writer_agent.parent_nodes.llm",
            FakeLLM({FinalReviewSchema: response}),
        ):
            updated = final_review(state)

        self.assertEqual(updated["final_review"]["action"], "escalate")
        self.assertEqual(
            updated["escalation_reason"],
            "Final writing failed after retries.",
        )
        self.assertEqual(route_after_final_review(updated), "escalate")

    def test_writing_agent_receives_final_review_feedback(self):
        """Verify final reviewer issues are included in the writing prompt."""
        captured_messages = []
        response = WritingResponseSchema(
            content="Revised output that addresses the conclusion feedback.",
            confidence=0.9,
        )
        subtask = {
            "id": "s1",
            "agent_type": "writing",
            "objective": "Revise",
            "expected_output": "Article",
            "tools_allowed": [],
            "review_criteria": [],
            "status": "running",
            "retry_count": 0,
        }
        state = {
            "user_request": "Write an article.",
            "subtasks": [subtask],
            "current_subtask_id": "s1",
            "subtask_results": {},
            "revision_feedback": ["Expand the conclusion."],
        }

        with patch(
            "writer_agent.specialist_nodes.llm",
            FakeLLM(
                {WritingResponseSchema: response},
                captured_messages,
            ),
        ):
            writing_agent(state)

        human_message = next(
            message
            for message in captured_messages
            if isinstance(message, HumanMessage)
        )
        self.assertIn("Expand the conclusion.", human_message.content)


class SupervisorReplanTests(unittest.TestCase):
    def test_subtask_replan_returns_feedback_to_supervisor(self):
        """Verify specialist issues guide a clean replacement plan."""
        subtask = {
            "id": "s1",
            "agent_type": "data",
            "objective": "Analyse",
            "expected_output": "Analysis",
            "tools_allowed": [],
            "review_criteria": [],
            "status": "running",
            "retry_count": 0,
        }
        state = {
            "subtasks": [subtask],
            "current_subtask_id": "s1",
            "review_reports": [
                {
                    "subtask_id": "s1",
                    "passed": False,
                    "score": 0.2,
                    "issues": ["Research evidence is required first."],
                    "action": "replan",
                }
            ],
            "replan_count": 0,
            "max_replans": 1,
        }

        requested = request_replan(state)
        self.assertTrue(requested["replan_requested"])
        self.assertEqual(
            requested["replan_feedback"],
            ["Research evidence is required first."],
        )
        requested_state = {**state, **requested}
        self.assertEqual(route_after_specialists(requested_state), "replan")

        prepared = prepare_replan(requested_state)
        self.assertEqual(prepared["replan_count"], 1)

        captured_messages = []
        plan = SupervisorPlanSchema(
            plan="Research before analysis.",
            plan_confidence=0.9,
            subtasks=[
                PlannedSubtaskSchema(
                    agent_type="research",
                    objective="Gather evidence",
                    expected_output="Research notes",
                    review_criteria=["Source grounded"],
                ),
                PlannedSubtaskSchema(
                    agent_type="writing",
                    objective="Write the answer",
                    expected_output="Final answer",
                    review_criteria=["Uses the evidence"],
                ),
            ],
        )
        planning_state = {
            **state,
            **requested_state,
            **prepared,
            "user_request": "Analyse the topic.",
            "subtask_results": {"s1": writing_result("s1", "Stale output")},
        }

        with patch(
            "writer_agent.parent_nodes.llm",
            FakeLLM(
                {SupervisorPlanSchema: plan},
                captured_messages,
            ),
        ):
            replanned = supervisor_plan(planning_state)

        human_message = next(
            message
            for message in captured_messages
            if isinstance(message, HumanMessage)
        )
        self.assertIn("Research evidence is required first.", human_message.content)
        self.assertEqual(replanned["subtask_results"], {})
        self.assertFalse(replanned["replan_requested"])
        self.assertEqual(replanned["replan_feedback"], [])
        self.assertEqual(
            [subtask["agent_type"] for subtask in replanned["subtasks"]],
            ["research", "writing"],
        )

    def test_invalid_replan_without_current_subtask_escalates(self):
        """Verify malformed specialist replanning escalates safely."""
        state = {
            "subtasks": [],
            "current_subtask_id": None,
            "review_reports": [
                {
                    "subtask_id": None,
                    "passed": False,
                    "score": 0.0,
                    "issues": ["No current subtask."],
                    "action": "replan",
                }
            ],
            "replan_count": 0,
            "max_replans": 1,
        }

        requested = {**state, **request_replan(state)}

        self.assertEqual(requested["status"], "escalated")
        self.assertEqual(route_after_specialists(requested), "escalate")

    def test_graphs_expose_retry_and_replan_nodes(self):
        """Verify both compiled graphs contain the new recovery nodes."""
        specialist_nodes = build_specialist_graph().get_graph().nodes
        supervisor_nodes = build_supervisor_graph().get_graph().nodes

        self.assertIn("request_replan", specialist_nodes)
        self.assertIn("prepare_final_retry", supervisor_nodes)
        self.assertIn("prepare_replan", supervisor_nodes)


class FullGraphControlFlowTests(unittest.TestCase):
    def test_final_review_retry_revises_writing_and_completes(self):
        """Verify a full final-revision loop completes with revised writing."""
        plan = SupervisorPlanSchema(
            plan="Write the answer.",
            plan_confidence=0.9,
            subtasks=[
                PlannedSubtaskSchema(
                    agent_type="writing",
                    objective="Write the answer",
                    expected_output="A complete answer",
                    review_criteria=["Clear and complete"],
                )
            ],
        )
        fake_llm = FakeLLM(
            {
                SupervisorPlanSchema: plan,
                WritingResponseSchema: [
                    WritingResponseSchema(
                        content="First draft with enough content to validate.",
                        confidence=0.8,
                    ),
                    WritingResponseSchema(
                        content="Revised draft with a stronger and complete conclusion.",
                        confidence=0.95,
                    ),
                ],
                ReviewDecisionSchema: [
                    ReviewDecisionSchema(
                        passed=True,
                        score=0.8,
                        issues=[],
                        action="pass",
                    ),
                    ReviewDecisionSchema(
                        passed=True,
                        score=0.95,
                        issues=[],
                        action="pass",
                    ),
                ],
                FinalReviewSchema: [
                    FinalReviewSchema(
                        passed=False,
                        score=0.6,
                        issues=["Strengthen the conclusion."],
                        action="retry",
                    ),
                    FinalReviewSchema(
                        passed=True,
                        score=0.95,
                        issues=[],
                        action="return",
                    ),
                ],
            }
        )

        with (
            patch("writer_agent.parent_nodes.llm", fake_llm),
            patch("writer_agent.specialist_nodes.llm", fake_llm),
        ):
            result = build_supervisor_graph().invoke(
                {"user_request": "Write an answer."}
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_retry_count"], 1)
        self.assertEqual(
            result["final_answer"],
            "Revised draft with a stronger and complete conclusion.",
        )
        titles = [event["title"] for event in result["workflow_events"]]
        self.assertEqual(titles.count("Initial plan"), 1)
        self.assertEqual(titles.count("Draft response"), 1)
        self.assertEqual(titles.count("Revised draft"), 1)
        self.assertEqual(
            titles.count("Final review requested a rewrite"),
            1,
        )
        self.assertEqual(titles.count("Final review passed"), 1)

    def test_specialist_replan_builds_corrected_plan_and_completes(self):
        """Verify specialist feedback triggers a successful supervisor replan."""
        initial_plan = SupervisorPlanSchema(
            plan="Analyse the topic.",
            plan_confidence=0.9,
            subtasks=[
                PlannedSubtaskSchema(
                    agent_type="data",
                    objective="Analyse the topic",
                    expected_output="Analysis",
                    review_criteria=["Evidence based"],
                )
            ],
        )
        corrected_plan = SupervisorPlanSchema(
            plan="Write a supported answer.",
            plan_confidence=0.9,
            subtasks=[
                PlannedSubtaskSchema(
                    agent_type="writing",
                    objective="Write the supported answer",
                    expected_output="Final answer",
                    review_criteria=["Clear"],
                )
            ],
        )
        fake_llm = FakeLLM(
            {
                SupervisorPlanSchema: [initial_plan, corrected_plan],
                DataResponseSchema: DataResponseSchema(
                    content="Insufficient analysis because evidence is missing.",
                    confidence=0.3,
                ),
                WritingResponseSchema: WritingResponseSchema(
                    content="Corrected final response after supervisor replanning.",
                    confidence=0.9,
                ),
                ReviewDecisionSchema: [
                    ReviewDecisionSchema(
                        passed=False,
                        score=0.3,
                        issues=["Use a corrected workflow."],
                        action="replan",
                    ),
                    ReviewDecisionSchema(
                        passed=True,
                        score=0.9,
                        issues=[],
                        action="pass",
                    ),
                ],
                FinalReviewSchema: FinalReviewSchema(
                    passed=True,
                    score=0.9,
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
                {"user_request": "Analyse the topic."}
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["replan_count"], 1)
        self.assertFalse(result["replan_requested"])
        self.assertEqual(
            result["final_answer"],
            "Corrected final response after supervisor replanning.",
        )
        titles = [event["title"] for event in result["workflow_events"]]
        self.assertEqual(titles.count("Initial plan"), 1)
        self.assertEqual(titles.count("Revised plan"), 1)

    def test_final_reviewer_replan_builds_corrected_plan_and_completes(self):
        """Verify final-review feedback triggers a successful supervisor replan."""
        initial_plan = SupervisorPlanSchema(
            plan="Write an initial answer.",
            plan_confidence=0.9,
            subtasks=[
                PlannedSubtaskSchema(
                    agent_type="writing",
                    objective="Write the answer",
                    expected_output="Final answer",
                    review_criteria=["Clear"],
                )
            ],
        )
        corrected_plan = SupervisorPlanSchema(
            plan="Write a corrected answer.",
            plan_confidence=0.95,
            subtasks=[
                PlannedSubtaskSchema(
                    agent_type="writing",
                    objective="Correct the misunderstood answer",
                    expected_output="Corrected final answer",
                    review_criteria=["Matches the request"],
                )
            ],
        )
        fake_llm = FakeLLM(
            {
                SupervisorPlanSchema: [initial_plan, corrected_plan],
                WritingResponseSchema: [
                    WritingResponseSchema(
                        content="A coherent response that misunderstands the request.",
                        confidence=0.8,
                    ),
                    WritingResponseSchema(
                        content="A corrected response based on the replanned workflow.",
                        confidence=0.95,
                    ),
                ],
                ReviewDecisionSchema: [
                    ReviewDecisionSchema(
                        passed=True,
                        score=0.8,
                        issues=[],
                        action="pass",
                    ),
                    ReviewDecisionSchema(
                        passed=True,
                        score=0.95,
                        issues=[],
                        action="pass",
                    ),
                ],
                FinalReviewSchema: [
                    FinalReviewSchema(
                        passed=False,
                        score=0.4,
                        issues=["The workflow misunderstood the request."],
                        action="replan",
                    ),
                    FinalReviewSchema(
                        passed=True,
                        score=0.95,
                        issues=[],
                        action="return",
                    ),
                ],
            }
        )

        with (
            patch("writer_agent.parent_nodes.llm", fake_llm),
            patch("writer_agent.specialist_nodes.llm", fake_llm),
        ):
            result = build_supervisor_graph().invoke(
                {"user_request": "Write the requested answer."}
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(
            result["final_answer"],
            "A corrected response based on the replanned workflow.",
        )
        titles = [event["title"] for event in result["workflow_events"]]
        self.assertEqual(titles.count("Initial plan"), 1)
        self.assertEqual(titles.count("Revised plan"), 1)


if __name__ == "__main__":
    unittest.main()
