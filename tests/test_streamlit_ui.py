"""Focused tests for Streamlit navigation behavior."""

import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from writer_agent.streamlit_ui import (
    APP_CSS,
    NEW_TASK_LABEL,
    _render_task_navigation,
    _render_supporting_details,
    render_task_body,
)
from writer_agent.ui_models import TaskSummary, TaskView, steps_for_stage


class SidebarNavigationTests(unittest.TestCase):
    def test_completed_active_task_is_clickable_and_has_no_tick(self):
        task = TaskSummary(
            id="task-1",
            title="Comparison of React and Angular",
            status="completed",
            stage="completed",
            updated_at=datetime.now(UTC),
        )

        with patch(
            "writer_agent.streamlit_ui.st.button",
            return_value=False,
        ) as render_button:
            _render_task_navigation(task, active_task_id=task.id)

        _, arguments = render_button.call_args
        self.assertEqual(
            render_button.call_args.args[0],
            "Comparison of React and Angular",
        )
        self.assertEqual(arguments["key"], "task-nav-active-task-1")
        self.assertNotIn("disabled", arguments)

    def test_sidebar_styles_define_selected_and_left_aligned_navigation(self):
        self.assertEqual(NEW_TASK_LABEL, "+ New")
        self.assertIn(
            '[class*="st-key-task-nav-active-"] button',
            APP_CSS,
        )
        self.assertIn("text-align: left;", APP_CSS)
        self.assertIn(
            '.stButton > [data-testid="stBaseButton-primary"]',
            APP_CSS,
        )

    def test_supporting_details_has_no_separate_review_tab(self):
        import inspect

        source = inspect.getsource(_render_supporting_details)

        self.assertIn('st.tabs(["Workflow", "Sources"])', source)
        self.assertNotIn("review_tab", source)
        self.assertNotIn("st.caption(label)", source)

    def test_running_task_renders_live_workflow_details(self):
        now = datetime.now(UTC)
        task = TaskView(
            id="task-1",
            thread_id="thread-1",
            user_id="local-user",
            title="Live task",
            request="Write a live workflow test.",
            status="running",
            stage="researching",
            status_message="Researching relevant sources…",
            progress_current=2,
            steps=steps_for_stage("researching"),
            workflow_events=[
                {
                    "id": "event-1",
                    "kind": "plan",
                    "title": "Initial plan",
                    "content": "Research and write.",
                }
            ],
            created_at=now,
            updated_at=now,
        )

        with (
            patch("writer_agent.streamlit_ui.render_progress"),
            patch("writer_agent.streamlit_ui.st.html"),
            patch("writer_agent.streamlit_ui.st.caption"),
            patch(
                "writer_agent.streamlit_ui._render_supporting_details"
            ) as render_details,
        ):
            render_task_body(task, service=object())

        render_details.assert_called_once_with(task)


if __name__ == "__main__":
    unittest.main()
