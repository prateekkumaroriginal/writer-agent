"""Live end-to-end test for a student-protest report request."""

import json
import os
import unicodedata
import unittest
from pathlib import Path

from writer_agent import build_supervisor_graph

USER_REQUEST = "Write a report on Student Protest on Jantar Mantar"
RUN_LIVE_TESTS = os.getenv("RUN_LIVE_TESTS") == "1"
RUN_OUTPUT = Path(__file__).parent / "runs" / "student-protest-jantar-mantar.json"


@unittest.skipUnless(
    RUN_LIVE_TESTS,
    "Set RUN_LIVE_TESTS=1 to run live Groq and Tavily integration tests.",
)
class StudentProtestFullGraphTest(unittest.TestCase):
    """Exercise the complete graph with live model and search providers."""

    def test_student_protest_report_completes(self):
        """Verify the full graph returns a reviewed and relevant report."""
        missing_keys = [
            key
            for key in ("GROQ_API_KEY", "TAVILY_API_KEY")
            if not os.getenv(key)
        ]
        self.assertFalse(
            missing_keys,
            f"Missing required live-test credentials: {', '.join(missing_keys)}",
        )

        result = build_supervisor_graph().invoke(
            {
                "user_id": "live_test_user",
                "user_request": USER_REQUEST,
            }
        )

        RUN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        RUN_OUTPUT.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

        self.assertEqual(
            result.get("status"),
            "completed",
            {
                "error": result.get("error"),
                "escalation_reason": result.get("escalation_reason"),
                "final_review": result.get("final_review"),
            },
        )
        self.assertIsNone(result.get("error"))
        self.assertIsNone(result.get("escalation_reason"))

        final_review = result.get("final_review")
        self.assertIsNotNone(final_review)
        self.assertTrue(final_review["passed"])
        self.assertEqual(final_review["action"], "return")

        final_answer = result.get("final_answer")
        self.assertIsInstance(final_answer, str)
        self.assertGreaterEqual(len(final_answer.split()), 150)

        normalized_answer = " ".join(
            unicodedata.normalize("NFKC", final_answer).casefold().split()
        )
        for required_term in ("student", "protest", "jantar mantar"):
            self.assertIn(required_term, normalized_answer)

        subtasks = result.get("subtasks", [])
        self.assertTrue(subtasks)
        self.assertTrue(
            all(subtask.get("status") == "passed" for subtask in subtasks)
        )

        results = result.get("subtask_results", {}).values()
        self.assertTrue(all(not item.get("errors") for item in results))


if __name__ == "__main__":
    unittest.main()
