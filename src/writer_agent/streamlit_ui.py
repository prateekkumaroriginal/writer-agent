"""Rendering helpers for the Streamlit product interface."""

from __future__ import annotations

import base64
import html
from datetime import UTC, datetime
from uuid import uuid4

import streamlit as st

from writer_agent.service import WriterAgentService
from writer_agent.ui_models import TaskSummary, TaskView, WorkflowEventView

STATUS_LABELS = {
    "queued": "Queued",
    "running": "In progress",
    "completed": "Reviewed",
    "escalated": "Needs attention",
    "failed": "Couldn’t complete",
    "interrupted": "Interrupted",
}
NEW_TASK_LABEL = "+ New"

APP_CSS = """
<style>
:root {
  color-scheme: dark;
  --wa-bg: #0b0b0f;
  --wa-sidebar: #111119;
  --wa-panel: #15151c;
  --wa-border: #2d2d3a;
  --wa-border-strong: #3b3b4c;
  --wa-text: #f0f0f5;
  --wa-muted: #a3a3b2;
  --wa-accent: #9697ff;
  --wa-accent-soft: #28284a;
  --wa-error: #f07979;
}

.stApp,
[data-testid="stAppViewContainer"] {
  background:
    linear-gradient(rgba(224, 224, 244, 0.026) 1px, transparent 1px),
    linear-gradient(90deg, rgba(224, 224, 244, 0.026) 1px, transparent 1px),
    var(--wa-bg);
  background-size: 28px 28px;
  color: var(--wa-text);
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: var(--wa-sidebar);
  border-right: 1px solid var(--wa-border);
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }
.block-container {
  max-width: 1080px;
  padding-top: 3.2rem;
  padding-bottom: 6rem;
}
h1, h2, h3, p, div, label { color: var(--wa-text); }
h1 { letter-spacing: -0.04em; }

.wa-brand {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  padding: 0.35rem 0 1.15rem;
  font-size: 1.08rem;
  font-weight: 760;
  letter-spacing: -0.025em;
}
.wa-mark {
  display: grid;
  place-items: center;
  width: 1.9rem;
  height: 1.9rem;
  border-radius: 0.58rem;
  background: var(--wa-accent);
  color: #101018;
  font-weight: 900;
}
.wa-section-label {
  margin: 1.15rem 0 0.35rem;
  color: var(--wa-muted);
  font-size: 0.69rem;
  font-weight: 750;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}
.wa-memory-item {
  padding: 0.68rem 0.75rem;
  margin: 0.42rem 0;
  border: 1px solid var(--wa-border);
  border-radius: 0.55rem;
  background: #171720;
}
.wa-memory-kind {
  margin-bottom: 0.2rem;
  color: #b7b7ff;
  font-size: 0.64rem;
  font-weight: 760;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.wa-memory-content {
  color: #dddde7;
  font-size: 0.82rem;
  line-height: 1.42;
}

.wa-hero {
  max-width: 820px;
  padding-top: min(10vh, 6rem);
}
.wa-kicker {
  display: inline-flex;
  padding: 0.3rem 0.62rem;
  margin-bottom: 1.1rem;
  border: 1px solid #434376;
  border-radius: 999px;
  background: var(--wa-accent-soft);
  color: #c2c2ff;
  font-size: 0.75rem;
  font-weight: 700;
}
.wa-hero h1 {
  max-width: 760px;
  margin: 0;
  font-size: clamp(2.7rem, 7vw, 5.2rem);
  line-height: 0.98;
}
.wa-hero p {
  max-width: 660px;
  margin: 1.25rem 0 2.1rem;
  color: var(--wa-muted);
  font-size: 1.08rem;
  line-height: 1.65;
}

.wa-task-header {
  padding-bottom: 1.2rem;
  border-bottom: 1px solid var(--wa-border);
}
.wa-task-title {
  max-width: 760px;
  margin: 0;
  font-size: clamp(1.7rem, 4vw, 2.55rem);
  line-height: 1.14;
  letter-spacing: -0.035em;
}
.wa-brief {
  margin: 1.3rem 0 1rem;
  padding: 1rem 1.15rem;
  border: 1px solid var(--wa-border);
  border-radius: 0.75rem;
  background: rgba(21, 21, 28, 0.92);
}
.wa-brief-label {
  margin-bottom: 0.35rem;
  color: var(--wa-muted);
  font-size: 0.68rem;
  font-weight: 760;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}
.wa-brief-text {
  color: #dedee8;
  font-size: 1rem;
  line-height: 1.58;
}

.wa-progress {
  padding: 1rem 1.1rem;
  margin: 1rem 0;
  border: 1px solid var(--wa-border);
  border-radius: 0.75rem;
  background: rgba(21, 21, 28, 0.92);
}
.wa-progress-head {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 0.85rem;
  font-size: 0.82rem;
  font-weight: 710;
}
.wa-progress-count {
  color: var(--wa-muted);
  font-weight: 600;
}
.wa-steps {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 0.5rem;
}
.wa-step {
  padding-top: 0.58rem;
  border-top: 3px solid #3d3d49;
  color: #7e7e8d;
  font-size: 0.71rem;
}
.wa-step.done {
  border-color: #7778dc;
  color: #aaaaff;
}
.wa-step.current {
  border-color: var(--wa-accent);
  color: var(--wa-text);
  font-weight: 760;
}
.wa-step.error {
  border-color: var(--wa-error);
  color: #ffaaaa;
  font-weight: 760;
}
.wa-status-line {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.9rem 1.05rem;
  margin-bottom: 1rem;
  border: 1px solid #3b3b5e;
  border-radius: 0.75rem;
  background: #1b1b2d;
  color: #ddddf2;
}

[class*="st-key-workflow-meta-"] {
  margin-bottom: 1rem;
  padding: 0.8rem 1rem 0.9rem;
  border: 1px solid var(--wa-border);
  border-radius: 0.65rem;
  background: rgba(150, 151, 255, 0.055);
}
[class*="st-key-workflow-meta-"] [data-testid="stCaptionContainer"] {
  color: #8f90a3;
  font-size: 0.68rem;
  font-weight: 760;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
[class*="st-key-workflow-event-decision-"] summary code {
  display: inline-flex;
  align-items: center;
  padding: 0.12rem 0.42rem;
  border: 1px solid currentColor;
  border-radius: 0.35rem;
  font-family: inherit;
  font-size: 0.74rem;
  font-weight: 700;
  line-height: 1.25;
  vertical-align: middle;
}
[class*="st-key-workflow-event-"] details summary {
  align-items: center;
  min-height: 3.35rem;
  padding-block: 0.7rem;
  overflow: visible;
}
[class*="st-key-workflow-event-"] details summary p {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin: 0;
  line-height: 1.35;
}
[class*="st-key-workflow-event-decision-green-"] summary code {
  background: rgba(46, 160, 86, 0.14);
  color: #52d681;
}
[class*="st-key-workflow-event-decision-orange-"] summary code {
  background: rgba(224, 137, 44, 0.14);
  color: #f1a654;
}
[class*="st-key-workflow-event-decision-yellow-"] summary code {
  background: rgba(203, 170, 48, 0.14);
  color: #e3c75d;
}
[class*="st-key-workflow-event-decision-red-"] summary code {
  background: rgba(217, 72, 72, 0.14);
  color: #ee7777;
}
[class*="st-key-workflow-event-decision-blue-"] summary code {
  background: rgba(75, 129, 221, 0.14);
  color: #78a7f5;
}
.wa-pulse {
  width: 0.62rem;
  height: 0.62rem;
  border-radius: 999px;
  background: var(--wa-accent);
  box-shadow: 0 0 0 0 rgba(150, 151, 255, 0.55);
  animation: wa-pulse 1.7s infinite;
}
@keyframes wa-pulse {
  0% { box-shadow: 0 0 0 0 rgba(150, 151, 255, 0.5); }
  70% { box-shadow: 0 0 0 9px rgba(150, 151, 255, 0); }
  100% { box-shadow: 0 0 0 0 rgba(150, 151, 255, 0); }
}
.wa-document-label {
  margin-top: 1.4rem;
  margin-bottom: 0.55rem;
  color: var(--wa-muted);
  font-size: 0.72rem;
  font-weight: 720;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

[data-testid="stChatInput"] {
  max-width: 880px;
  margin-inline: auto;
}
[data-testid="stChatInput"] > div {
  border: 1px solid var(--wa-border-strong);
  background: #171720;
  box-shadow: 0 12px 34px rgba(0, 0, 0, 0.32);
}
[data-testid="stChatInput"] textarea { color: var(--wa-text); }
.stButton > button,
.stDownloadButton > button,
.stLinkButton > a {
  border-color: var(--wa-border-strong);
  background: #1b1b25;
  color: var(--wa-text);
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stLinkButton > a:hover {
  border-color: var(--wa-accent);
  color: #c9c9ff;
}
[data-testid="stSidebar"] .stButton > button {
  justify-content: flex-start;
  width: 100%;
  border-color: transparent;
  background: transparent;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #20202e; }
[data-testid="stSidebar"] .stButton > [data-testid="stBaseButton-primary"] {
  justify-content: center;
  border-color: var(--wa-accent);
  background: var(--wa-accent);
  color: #101018;
  font-weight: 780;
}
[data-testid="stSidebar"] .stButton > [data-testid="stBaseButton-primary"]:hover {
  border-color: #b0b0ff;
  background: #aaaaff;
  color: #101018;
}
[data-testid="stSidebar"] [class*="st-key-task-nav-"] button {
  justify-content: flex-start;
  text-align: left;
}
[data-testid="stSidebar"] [class*="st-key-task-nav-"] button p {
  width: 100%;
  text-align: left;
}
[data-testid="stSidebar"] [class*="st-key-task-nav-active-"] button {
  border-color: #4b4b72;
  background: var(--wa-accent-soft);
  color: #d8d8ff;
}
[data-testid="stSidebar"] [class*="st-key-task-nav-active-"] button:hover {
  border-color: #5c5c8d;
  background: #303054;
  color: #eeeeff;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 0.35rem; }
[data-testid="stTabs"] [data-baseweb="tab"] { border-radius: 0.45rem; }

@media (max-width: 760px) {
  .block-container { padding-top: 1.8rem; }
  .wa-steps { grid-template-columns: 1fr; }
  .wa-step {
    padding: 0.35rem 0 0.35rem 0.65rem;
    border-top: 0;
    border-left: 3px solid #3d3d49;
  }
}
</style>
"""


def apply_product_styles() -> None:
    st.html(APP_CSS)


def render_sidebar(
    service: WriterAgentService,
    *,
    active_task_id: str | None,
) -> None:
    """Render navigation and mutate the active task selection."""
    with st.sidebar:
        st.html(
            """
            <div class="wa-brand">
              <span class="wa-mark">W</span>
              <span>Writer Agent</span>
            </div>
            """
        )
        if st.button(
            NEW_TASK_LABEL,
            type="primary",
            use_container_width=True,
        ):
            st.session_state.active_task_id = None
            st.session_state.submission_key = str(uuid4())
            st.query_params.clear()
            st.rerun()

        _render_memory_list(service)
        st.html('<div class="wa-section-label">Recent work</div>')
        try:
            recent_tasks = service.list_recent_tasks()
        except Exception:
            st.caption("Recent tasks are temporarily unavailable.")
            return

        if not recent_tasks:
            st.caption("Your writing tasks will appear here.")
            return

        for task in recent_tasks:
            _render_task_navigation(task, active_task_id=active_task_id)


def _render_memory_list(service: WriterAgentService) -> None:
    """Render the current user's memories without mutation controls."""
    with st.expander("Saved memory"):
        st.caption(
            "Writer Agent manages these memories from your messages. Changes "
            "appear in the relevant task’s Workflow tab."
        )
        try:
            memories = service.list_memories()
        except Exception:
            st.caption("Saved memory is temporarily unavailable.")
            return
        if not memories:
            st.caption("No long-term memories have been saved yet.")
            return
        for memory in memories:
            label = (
                "Core preference"
                if memory.kind == "core"
                else "Contextual memory"
            )
            st.html(
                f"""
                <div class="wa-memory-item">
                  <div class="wa-memory-kind">{html.escape(label)}</div>
                  <div class="wa-memory-content">
                    {html.escape(memory.content)}
                  </div>
                </div>
                """
            )


def _render_task_navigation(
    task: TaskSummary,
    *,
    active_task_id: str | None,
) -> None:
    prefix = {
        "completed": "",
        "escalated": "!",
        "failed": "×",
        "interrupted": "↻",
        "queued": "·",
        "running": "●",
    }[task.status]
    label = f"{prefix}  {task.title}" if prefix else task.title
    is_active = task.id == active_task_id
    if st.button(
        label,
        key=f"task-nav-{'active-' if is_active else ''}{task.id}",
        help=f"{STATUS_LABELS[task.status]} · {_relative_time(task.updated_at)}",
        type="secondary",
        use_container_width=True,
    ):
        st.session_state.active_task_id = task.id
        st.query_params["task"] = task.id
        st.rerun()


def render_empty_state(service: WriterAgentService) -> None:
    """Render the first-use screen and accept a new writing brief."""
    st.html(
        """
        <section class="wa-hero">
          <div class="wa-kicker">Research · Analyse · Write · Review</div>
          <h1>Turn a clear brief into a reviewed document.</h1>
          <p>
            Describe what you need. Writer Agent plans the work, researches
            sources, analyses the evidence, writes the document, and checks it
            before returning the result.
          </p>
        </section>
        """
    )

    st.caption("Try an example")
    examples = (
        "Write an overview of retrieval-augmented generation for product leaders.",
        "Create a balanced report on student protests at Jantar Mantar.",
        "Explain the benefits and risks of AI agents for a small business.",
    )
    columns = st.columns(3)
    for index, (column, example) in enumerate(zip(columns, examples, strict=True)):
        with column:
            if st.button(
                example,
                key=f"example-{index}",
                use_container_width=True,
            ):
                st.session_state.brief_input = example
                st.rerun()

    brief = st.chat_input(
        "What should I research and write?",
        key="brief_input",
        max_chars=8_000,
    )
    if brief:
        _submit_brief(service, brief)


def _submit_brief(service: WriterAgentService, brief: str) -> None:
    submission_key = st.session_state.get("submission_key") or str(uuid4())
    st.session_state.submission_key = submission_key
    try:
        task = service.create_task(brief, idempotency_key=submission_key)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error(
            "Writer Agent could not start this task. Check that PostgreSQL is "
            "running, then try again."
        )
        return

    st.session_state.submission_key = str(uuid4())
    st.session_state.active_task_id = task.id
    st.query_params["task"] = task.id
    st.rerun()


def render_task_header(task: TaskView) -> None:
    brief_label = (
        f"Follow-up · Version {task.turn_number}"
        if task.turn_number > 1
        else "Your brief"
    )
    st.html(
        f"""
        <header class="wa-task-header">
          <h1 class="wa-task-title">{html.escape(task.title)}</h1>
        </header>
        <section class="wa-brief">
          <div class="wa-brief-label">{html.escape(brief_label)}</div>
          <div class="wa-brief-text">{html.escape(task.request)}</div>
        </section>
        """
    )


def render_task_body(
    task: TaskView,
    *,
    service: WriterAgentService,
) -> None:
    render_progress(task)
    if task.status in {"queued", "running"}:
        st.html(
            f"""
            <div class="wa-status-line">
              <span class="wa-pulse"></span>
              <span>{html.escape(task.status_message)}</span>
            </div>
            """
        )
        st.caption(
            "This task runs independently. You can leave this page and return "
            "from Recent work."
        )
        _render_supporting_details(task)
        _render_version_history(task, service)
        return

    if task.status == "completed":
        _render_completed_task(task, service)
    elif task.status == "interrupted":
        st.warning(task.status_message)
        if task.can_resume and st.button(
            "Resume from checkpoint",
            type="primary",
            icon=":material/replay:",
        ):
            try:
                service.resume_task(task.id)
            except Exception:
                st.error("This task could not be resumed. Start a new task instead.")
            else:
                st.rerun()
        _render_supporting_details(task)
        _render_version_history(task, service)
    elif task.status == "escalated":
        st.warning(task.status_message)
        st.caption(
            "No final document was returned because the workflow could not "
            "complete with sufficient confidence."
        )
        if st.button(
            "Try again",
            type="primary",
            icon=":material/replay:",
            key=f"retry-task-{task.id}",
        ):
            _retry_task(service, task)
        _render_supporting_details(task)
        _render_version_history(task, service)
    else:
        st.error(task.status_message)
        if st.button(
            "Try again",
            type="primary",
            icon=":material/replay:",
            key=f"retry-task-{task.id}",
        ):
            _retry_task(service, task)
        _render_supporting_details(task)
        _render_version_history(task, service)


def _retry_task(service: WriterAgentService, task: TaskView) -> None:
    submission_state_key = f"retry-submission-{task.id}"
    submission_key = st.session_state.get(submission_state_key) or str(uuid4())
    st.session_state[submission_state_key] = submission_key
    try:
        retry = service.retry_task(
            task.id,
            idempotency_key=submission_key,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("Writer Agent could not retry this task. Try again.")
        return

    st.session_state[submission_state_key] = str(uuid4())
    st.session_state.active_task_id = retry.id
    st.query_params["task"] = retry.id
    st.rerun()


def render_progress(task: TaskView) -> None:
    step_markup = "".join(
        f'<div class="wa-step {html.escape(step.status)}">'
        f"{html.escape(step.label)}</div>"
        for step in task.steps
    )
    st.html(
        f"""
        <section class="wa-progress">
          <div class="wa-progress-head">
            <span>{html.escape(task.status_message)}</span>
            <span class="wa-progress-count">
              {task.progress_current} of {task.progress_total} stages
            </span>
          </div>
          <div class="wa-steps">{step_markup}</div>
        </section>
        """
    )


def _render_completed_task(
    task: TaskView,
    service: WriterAgentService,
) -> None:
    if not task.final_answer:
        st.error("The workflow completed without a document.")
        return

    st.html('<div class="wa-document-label">Reviewed document</div>')
    with st.container(border=True):
        st.markdown(task.final_answer)

    action_columns = st.columns([1, 1, 3])
    with action_columns[0]:
        render_copy_button(task.final_answer)
    with action_columns[1]:
        st.download_button(
            "Download Markdown",
            data=task.final_answer,
            file_name=f"{_safe_filename(task.title)}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    _render_supporting_details(task)
    _render_version_history(task, service)
    _render_follow_up(task, service)


def _render_version_history(
    task: TaskView,
    service: WriterAgentService,
) -> None:
    """Render selectable immutable runs from the current conversation."""
    try:
        versions = service.list_task_versions(task.id)
    except Exception:
        return
    if len(versions) <= 1:
        return

    with st.expander(f"Version history · {len(versions)} versions"):
        for version in reversed(versions):
            label = f"Version {version.turn_number}: {version.request}"
            if len(label) > 96:
                label = label[:95].rstrip() + "…"
            if st.button(
                label,
                key=f"version-{version.id}",
                disabled=version.id == task.id,
                use_container_width=True,
            ):
                st.session_state.active_task_id = version.id
                st.query_params["task"] = version.id
                st.rerun()


def _render_follow_up(
    task: TaskView,
    service: WriterAgentService,
) -> None:
    """Accept one revision instruction for the latest completed version."""
    try:
        versions = service.list_task_versions(task.id)
    except Exception:
        versions = []
    is_latest = not versions or versions[-1].id == task.id
    if not is_latest:
        st.caption(
            "Open the latest version to continue this conversation."
        )
        return

    follow_up = st.chat_input(
        "How should I revise or extend this?",
        key=f"follow-up-{task.conversation_id}",
        max_chars=8_000,
    )
    if not follow_up:
        return

    submission_state_key = f"follow-up-submission-{task.id}"
    submission_key = st.session_state.get(submission_state_key) or str(uuid4())
    st.session_state[submission_state_key] = submission_key
    try:
        next_task = service.create_follow_up(
            task.id,
            follow_up,
            idempotency_key=submission_key,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception:
        st.error("Writer Agent could not start this revision. Try again.")
        return

    st.session_state[submission_state_key] = str(uuid4())
    st.session_state.active_task_id = next_task.id
    st.query_params["task"] = next_task.id
    st.rerun()


def render_copy_button(content: str) -> None:
    """Render a self-contained clipboard action without another dependency."""
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    st.iframe(
        f"""
        <style>
          html, body {{ margin: 0; background: transparent; }}
          button {{
            width: 100%; height: 40px; padding: 0 14px;
            border: 1px solid #3b3b4c; border-radius: 8px;
            background: #1b1b25; color: #f0f0f5;
            font: 600 14px ui-sans-serif, system-ui, sans-serif;
            cursor: pointer;
          }}
          button:hover {{ border-color: #9697ff; color: #c9c9ff; }}
        </style>
        <button id="copy">Copy document</button>
        <script>
          const button = document.getElementById("copy");
          const bytes = Uint8Array.from(
            atob("{encoded}"),
            character => character.charCodeAt(0)
          );
          const content = new TextDecoder().decode(bytes);
          button.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(content);
            }} catch (_) {{
              const area = document.createElement("textarea");
              area.value = content;
              document.body.appendChild(area);
              area.select();
              document.execCommand("copy");
              area.remove();
            }}
            button.textContent = "Copied";
            setTimeout(() => button.textContent = "Copy document", 1600);
          }});
        </script>
        """,
        height=42,
    )


def _render_supporting_details(task: TaskView) -> None:
    if not task.sources and not task.workflow_events:
        return
    st.write("")
    workflow_tab, sources_tab = st.tabs(["Workflow", "Sources"])
    with workflow_tab:
        st.caption(
            "Meaningful workflow outputs in the order they were produced. "
            "Hidden reasoning, prompts, and operational errors are excluded."
        )
        for event in task.workflow_events:
            _render_workflow_event(event)
    with sources_tab:
        if not task.sources:
            st.caption("No research sources were recorded for this task.")
        else:
            st.caption(
                "These are research inputs gathered by the workflow. They are "
                "not guaranteed to be inline citations for every statement."
            )
            for source in task.sources:
                st.link_button(
                    source.title,
                    source.url,
                    icon=":material/open_in_new:",
                )
                if source.snippet:
                    st.caption(source.snippet)


def _render_workflow_event(event: WorkflowEventView) -> None:
    """Render one workflow artifact as a collapsible, scannable section."""
    timestamp = event.created_at.astimezone(UTC)
    label = f"{event.title} · {timestamp:%d %b, %H:%M UTC}"
    key = f"workflow-event-{event.id}"
    if event.decision:
        badge_label, badge_color = _decision_badge(event.decision)
        label = f"`{badge_label}` {label}"
        key = f"workflow-event-decision-{badge_color}-{event.id}"

    with st.expander(label, key=key):
        has_metadata = any(
            (
                event.subtask_name,
                event.agent,
                event.objective,
                event.attempt,
                event.review_criteria,
            )
        )
        if has_metadata:
            with st.container(key=f"workflow-meta-{event.id}"):
                st.caption("Run details")

                if event.subtask_name or event.agent:
                    assignment = " · ".join(
                        value
                        for value in (event.subtask_name, event.agent)
                        if value
                    )
                    st.markdown(f"**{assignment}**")
                if event.objective:
                    st.caption("Objective")
                    st.write(event.objective)
                if event.attempt is not None:
                    retries = event.retry_count or 0
                    st.caption(
                        f"Attempt {event.attempt} · {retries} retries"
                    )
                if event.review_criteria:
                    st.caption("Review criteria")
                    st.markdown(
                        "\n".join(
                            f"- {criterion}"
                            for criterion in event.review_criteria
                        )
                    )

        if event.content or event.details or event.sources:
            st.caption(_event_content_label(event.kind))
            if event.content:
                st.markdown(event.content)
            for detail in event.details:
                st.markdown(f"- {detail}")

            if event.sources:
                st.markdown("**Sources**")
                for source in event.sources:
                    st.link_button(
                        source.title,
                        source.url,
                        icon=":material/open_in_new:",
                        key=f"workflow-source-{event.id}-{source.url}",
                    )
                    if source.snippet:
                        st.caption(source.snippet)


def _event_content_label(kind: str) -> str:
    """Name the primary artifact separately from execution metadata."""
    return {
        "plan": "Plan",
        "search": "Search query",
        "research": "Research output",
        "analysis": "Analysis output",
        "draft": "Draft",
        "review": "Review result",
        "replan": "Revised plan",
        "memory": "Memory change",
    }[kind]


def _decision_badge(decision: str) -> tuple[str, str]:
    """Map stored workflow decisions to explicit, consistent badges."""
    badges = {
        "pass": ("Passed", "green"),
        "return": ("Passed", "green"),
        "retry": ("Retry", "orange"),
        "replan": ("Replan", "yellow"),
        "escalate": ("Escalated", "red"),
        "add": ("Added", "green"),
        "edit": ("Updated", "blue"),
        "delete": ("Deleted", "red"),
    }
    return badges.get(decision, (decision.replace("_", " ").title(), "gray"))


def _relative_time(value: datetime) -> str:
    now = datetime.now(UTC)
    normalized = value if value.tzinfo else value.replace(tzinfo=UTC)
    seconds = max(0, int((now - normalized).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3_600:
        return f"{seconds // 60}m ago"
    if seconds < 86_400:
        return f"{seconds // 3_600}h ago"
    return f"{seconds // 86_400}d ago"


def _safe_filename(title: str) -> str:
    filename = "".join(
        character.lower() if character.isalnum() else "-"
        for character in title
    )
    return "-".join(part for part in filename.split("-") if part)[:80] or "document"
