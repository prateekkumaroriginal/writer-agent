"""Focused tests for Streamlit navigation behavior."""

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, Mock, patch

from writer_agent.streamlit_ui import (
    APP_CSS,
    NEW_TASK_LABEL,
    _render_memory_list,
    _render_workflow_event,
    _render_task_navigation,
    _render_supporting_details,
    render_sidebar,
    render_task_body,
)
from writer_agent.ui_models import (
    TaskSummary,
    TaskView,
    WorkflowEventView,
    steps_for_stage,
)


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
        self.assertIn(
            '[class*="st-key-workflow-event-"] details summary',
            APP_CSS,
        )

    def test_sidebar_has_no_manual_memory_controls(self):
        import inspect

        source = inspect.getsource(render_sidebar)

        self.assertNotIn("add_memory", source)
        self.assertNotIn("update_memory", source)
        self.assertNotIn("delete_memory", source)
        self.assertNotIn("_render_memory_controls", source)

        memory_source = inspect.getsource(_render_memory_list)
        self.assertIn("list_memories", memory_source)
        self.assertNotIn("st.button", memory_source)
        self.assertNotIn("st.form", memory_source)
        self.assertNotIn("st.selectbox", memory_source)
        self.assertNotIn("st.text_area", memory_source)

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
                    "created_at": now,
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

    def test_workflow_event_renders_requested_details(self):
        event = WorkflowEventView(
            id="event-1",
            kind="research",
            title="Research response",
            created_at=datetime(2026, 7, 31, 8, 30, tzinfo=UTC),
            content="Research summary.",
            decision="retry",
            subtask_name="Research task",
            objective="Find supporting evidence.",
            agent="Research agent",
            review_criteria=["Uses authoritative sources"],
            attempt=2,
            retry_count=1,
            sources=[
                {
                    "title": "Evidence",
                    "url": "https://example.com/evidence",
                }
            ],
        )
        expander = MagicMock()
        metadata_container = MagicMock()

        with (
            patch(
                "writer_agent.streamlit_ui.st.expander",
                return_value=expander,
            ) as render_expander,
            patch(
                "writer_agent.streamlit_ui.st.caption"
            ) as render_caption,
            patch(
                "writer_agent.streamlit_ui.st.markdown"
            ) as render_markdown,
            patch("writer_agent.streamlit_ui.st.write"),
            patch(
                "writer_agent.streamlit_ui.st.container",
                return_value=metadata_container,
            ) as render_container,
            patch(
                "writer_agent.streamlit_ui.st.link_button"
            ) as render_link,
        ):
            _render_workflow_event(event)

        self.assertIn("31 Jul, 08:30 UTC", render_expander.call_args.args[0])
        self.assertTrue(
            render_expander.call_args.args[0].startswith(
                "`Retry` Research response"
            )
        )
        self.assertIn(
            "workflow-event-decision-orange-event-1",
            render_expander.call_args.kwargs["key"],
        )
        markdown_values = [
            call.args[0] for call in render_markdown.call_args_list
        ]
        self.assertIn(
            "**Research task · Research agent**",
            markdown_values,
        )
        caption_values = [
            call.args[0] for call in render_caption.call_args_list
        ]
        self.assertIn("Run details", caption_values)
        self.assertIn("Attempt 2 · 1 retries", caption_values)
        self.assertIn("Review criteria", caption_values)
        self.assertIn("Research output", caption_values)
        render_container.assert_called_once_with(
            key="workflow-meta-event-1"
        )
        render_link.assert_called_once()
        self.assertEqual(
            render_link.call_args.args[:2],
            ("Evidence", "https://example.com/evidence"),
        )

    def test_escalated_try_again_uses_conversation_retry(self):
        now = datetime.now(UTC)
        task = TaskView(
            id="task-2",
            thread_id="thread-2",
            user_id="local-user",
            conversation_id="conversation-1",
            parent_task_id="task-1",
            turn_number=2,
            title="Revision",
            request="Remove the word count.",
            status="escalated",
            stage="attention",
            status_message="Revision escalated.",
            progress_current=5,
            steps=steps_for_stage("attention"),
            created_at=now,
            updated_at=now,
        )
        service = Mock()

        with (
            patch("writer_agent.streamlit_ui.render_progress"),
            patch("writer_agent.streamlit_ui.st.warning"),
            patch("writer_agent.streamlit_ui.st.caption"),
            patch(
                "writer_agent.streamlit_ui.st.button",
                return_value=True,
            ),
            patch(
                "writer_agent.streamlit_ui._retry_task"
            ) as retry_task,
            patch("writer_agent.streamlit_ui._render_supporting_details"),
            patch("writer_agent.streamlit_ui._render_version_history"),
        ):
            render_task_body(task, service=service)

        retry_task.assert_called_once_with(service, task)

    def test_interrupted_resume_continues_same_task(self):
        now = datetime.now(UTC)
        task = TaskView(
            id="task-2",
            thread_id="thread-2",
            user_id="local-user",
            conversation_id="conversation-1",
            parent_task_id="task-1",
            turn_number=2,
            title="Revision",
            request="Remove the word count.",
            status="interrupted",
            stage="attention",
            status_message="Revision interrupted.",
            progress_current=4,
            steps=steps_for_stage("attention"),
            can_resume=True,
            created_at=now,
            updated_at=now,
        )
        service = Mock()

        with (
            patch("writer_agent.streamlit_ui.render_progress"),
            patch("writer_agent.streamlit_ui.st.warning"),
            patch(
                "writer_agent.streamlit_ui.st.button",
                return_value=True,
            ),
            patch("writer_agent.streamlit_ui.st.rerun"),
            patch("writer_agent.streamlit_ui._render_supporting_details"),
            patch("writer_agent.streamlit_ui._render_version_history"),
        ):
            render_task_body(task, service=service)

        service.resume_task.assert_called_once_with(task.id)


if __name__ == "__main__":
    unittest.main()
