"""Regression tests for research retries and exhausted specialist routing."""

import os
import unittest
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage

from writer_agent.parent_nodes import route_after_specialists
from writer_agent.prompts import REVIEWER_SYSTEM_PROMPT
from writer_agent.schemas import (
    ResearchResponseSchema,
    ReviewDecisionSchema,
    SearchQuerySchema,
)
from writer_agent.search import search_web
from writer_agent.specialist_nodes import (
    build_search_query,
    mark_subtask_failed,
    research_agent,
    review_agent,
    route_after_subtask_review,
)


def research_subtask(*, retry_count=0):
    return {
        "id": "s1",
        "agent_type": "research",
        "objective": "Research a balanced report.",
        "expected_output": "Credible sources from multiple perspectives.",
        "tools_allowed": ["web_search"],
        "review_criteria": ["Include official and protester viewpoints."],
        "status": "running",
        "retry_count": retry_count,
    }


class FakeRunner:
    def __init__(self, captured):
        self.captured = captured

    def invoke(self, messages):
        self.captured.extend(messages)
        return SearchQuerySchema(
            query="Jantar Mantar student protest official response sources"
        )


class FakeLLM:
    def __init__(self, captured):
        self.captured = captured

    def with_structured_output(self, schema):
        if schema is not SearchQuerySchema:
            raise AssertionError(schema)
        return FakeRunner(self.captured)


class ResearchRetryTests(unittest.TestCase):
    def test_research_at_60_percent_confidence_must_pass(self):
        state = {
            "current_subtask_id": "s1",
            "subtasks": [research_subtask()],
            "subtask_results": {
                "s1": {
                    "subtask_id": "s1",
                    "agent_type": "research",
                    "output": {"summary": "Limited research."},
                    "confidence": 0.60,
                    "sources": [],
                    "errors": [],
                }
            },
        }

        with patch("writer_agent.specialist_nodes.llm", object()):
            result = review_agent(state)

        report = result["review_reports"][0]
        self.assertTrue(report["passed"])
        self.assertEqual(report["score"], 0.60)
        self.assertEqual(report["action"], "pass")
        self.assertEqual(report["issues"], [])
        event = result["workflow_events"][0]
        self.assertEqual(event["subtask_name"], "Research task")
        self.assertEqual(event["agent"], "Research agent")
        self.assertEqual(
            event["review_criteria"],
            ["Include official and protester viewpoints."],
        )
        self.assertEqual(event["attempt"], 1)
        self.assertEqual(event["retry_count"], 0)

    def test_research_event_attaches_its_sources_and_attempt(self):
        subtask = research_subtask(retry_count=1)
        state = {
            "user_request": "Create a balanced report.",
            "current_subtask_id": "s1",
            "subtasks": [subtask],
            "review_reports": [],
        }
        query_runner = Mock()
        query_runner.invoke.return_value = SearchQuerySchema(
            query="balanced report official sources"
        )
        research_runner = Mock()
        research_runner.invoke.return_value = ResearchResponseSchema(
            summary="A grounded summary.",
            findings=["A supported finding."],
            uncertainties=[],
            confidence=0.8,
        )
        fake_llm = Mock()
        fake_llm.with_structured_output.side_effect = lambda schema: {
            SearchQuerySchema: query_runner,
            ResearchResponseSchema: research_runner,
        }[schema]
        source = {
            "title": "Official evidence",
            "url": "https://example.com/evidence",
            "snippet": "Supporting context.",
        }

        with (
            patch("writer_agent.specialist_nodes.llm", fake_llm),
            patch(
                "writer_agent.specialist_nodes.search_web",
                return_value=[source],
            ),
        ):
            result = research_agent(state)

        event = next(
            item
            for item in result["workflow_events"]
            if item["kind"] == "research"
        )
        self.assertEqual(event["sources"], [source])
        self.assertEqual(event["attempt"], 2)
        self.assertEqual(event["retry_count"], 1)

    def test_reviewer_prompt_documents_research_threshold(self):
        self.assertIn("greater", REVIEWER_SYSTEM_PROMPT)
        self.assertIn("0.60", REVIEWER_SYSTEM_PROMPT)
        self.assertIn("overrides", REVIEWER_SYSTEM_PROMPT)

    def test_research_below_60_percent_still_uses_reviewer(self):
        state = {
            "current_subtask_id": "s1",
            "subtasks": [research_subtask()],
            "subtask_results": {
                "s1": {
                    "subtask_id": "s1",
                    "agent_type": "research",
                    "output": {"summary": "Insufficient research."},
                    "confidence": 0.599,
                    "sources": [],
                    "errors": [],
                }
            },
        }
        fake_llm = Mock()
        fake_llm.with_structured_output.return_value.invoke.return_value = (
            ReviewDecisionSchema(
                passed=False,
                score=0.599,
                issues=["Evidence is incomplete."],
                action="retry",
            )
        )

        with patch("writer_agent.specialist_nodes.llm", fake_llm):
            result = review_agent(state)

        self.assertFalse(result["review_reports"][0]["passed"])
        self.assertEqual(result["review_reports"][0]["action"], "retry")
        fake_llm.with_structured_output.return_value.invoke.assert_called_once()

    def test_search_query_receives_latest_reviewer_feedback(self):
        captured = []
        subtask = research_subtask(retry_count=1)
        state = {
            "user_request": "Create a balanced report.",
            "current_subtask_id": "s1",
            "subtasks": [subtask],
            "review_reports": [
                {
                    "subtask_id": "s1",
                    "passed": False,
                    "score": 0.4,
                    "issues": ["Official government statements are missing."],
                    "action": "retry",
                }
            ],
        }

        with patch("writer_agent.specialist_nodes.llm", FakeLLM(captured)):
            build_search_query(state, subtask)

        prompt = next(
            message.content
            for message in captured
            if isinstance(message, HumanMessage)
        )
        self.assertIn("Official government statements are missing.", prompt)

    def test_exhausted_retry_replans_before_terminal_failure(self):
        subtask = research_subtask(retry_count=2)
        state = {
            "current_subtask_id": "s1",
            "subtasks": [subtask],
            "review_reports": [
                {
                    "subtask_id": "s1",
                    "passed": False,
                    "score": 0.4,
                    "issues": ["More credible sources are needed."],
                    "action": "retry",
                }
            ],
            "max_retries": 2,
            "replan_count": 0,
            "max_replans": 1,
        }

        self.assertEqual(route_after_subtask_review(state), "replan")

    def test_terminal_specialist_failure_bypasses_final_review(self):
        subtask = research_subtask(retry_count=2)
        state = {
            "current_subtask_id": "s1",
            "subtasks": [subtask],
            "review_reports": [
                {
                    "subtask_id": "s1",
                    "passed": False,
                    "score": 0.4,
                    "issues": ["More credible sources are needed."],
                    "action": "retry",
                }
            ],
            "max_retries": 2,
            "replan_count": 1,
            "max_replans": 1,
        }

        self.assertEqual(route_after_subtask_review(state), "fail")
        failed = {**state, **mark_subtask_failed(state)}
        self.assertEqual(failed["status"], "escalated")
        self.assertEqual(route_after_specialists(failed), "escalate")

    def test_search_requests_five_results(self):
        captured = {}

        class FakeClient:
            def __init__(self, *, api_key):
                captured["api_key"] = api_key

            def search(self, **kwargs):
                captured.update(kwargs)
                return {"results": []}

        with (
            patch.dict(os.environ, {"TAVILY_API_KEY": "test-key"}),
            patch("writer_agent.search.TavilyClient", FakeClient),
        ):
            search_web("credible protest sources")

        self.assertEqual(captured["max_results"], 5)


if __name__ == "__main__":
    unittest.main()
