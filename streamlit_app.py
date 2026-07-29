"""Writer Agent end-user application."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv
from uuid import uuid4

from writer_agent.service import WriterAgentService
from writer_agent.streamlit_ui import (
    apply_product_styles,
    render_empty_state,
    render_sidebar,
    render_task_body,
    render_task_header,
)

load_dotenv()

st.set_page_config(
    page_title="Writer Agent",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_product_styles()


@st.cache_resource
def get_writer_service() -> WriterAgentService:
    """Create one process-level service and connection pool."""
    return WriterAgentService.from_environment()


if "active_task_id" not in st.session_state:
    st.session_state.active_task_id = None
if "submission_key" not in st.session_state:
    st.session_state.submission_key = str(uuid4())

linked_task_id = st.query_params.get("task")
if linked_task_id and linked_task_id != st.session_state.active_task_id:
    st.session_state.active_task_id = linked_task_id

try:
    service = get_writer_service()
except Exception:
    st.error("Writer Agent could not connect to its local database.")
    st.info(
        "Start PostgreSQL with `docker compose up -d --wait checkpoint-db`, "
        "then reload this page."
    )
    st.stop()

render_sidebar(
    service,
    active_task_id=st.session_state.active_task_id,
)

active_task_id = st.session_state.active_task_id
if active_task_id is None:
    render_empty_state(service)
    st.stop()

try:
    selected_task = service.get_task(active_task_id)
except KeyError:
    st.session_state.active_task_id = None
    st.query_params.clear()
    st.warning("That writing task no longer exists.")
    st.stop()
except Exception:
    st.error("Writer Agent could not load this task.")
    st.stop()

render_task_header(selected_task)


@st.fragment(run_every="2s")
def render_live_task(task_id: str) -> None:
    """Refresh only the running task body."""
    try:
        current_task = service.get_task(task_id)
    except Exception:
        st.error("Writer Agent could not refresh this task.")
        return
    render_task_body(current_task, service=service)
    if not current_task.is_active:
        st.rerun()


if selected_task.is_active:
    render_live_task(selected_task.id)
else:
    render_task_body(selected_task, service=service)
