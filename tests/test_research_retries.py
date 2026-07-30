"""Regression tests for research retries and exhausted specialist routing."""

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from writer_agent.parent_nodes import route_after_specialists
from writer_agent.schemas import SearchQuerySchema
from writer_agent.search import search_web
from writer_agent.specialist_nodes import (
    build_search_query,
    mark_subtask_failed,
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

    def test_search_requests_eight_results(self):
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

        self.assertEqual(captured["max_results"], 8)


if __name__ == "__main__":
    unittest.main()
