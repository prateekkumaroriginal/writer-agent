"""Nodes and routers owned by the parent supervisor graph."""

import re
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from writer_agent.helpers import (
    all_subtasks_passed,
    build_reused_artifacts,
    build_runtime_subtasks,
    format_previous_artifacts_for_supervisor,
)
from writer_agent.model import llm
from writer_agent.prompts import (
    FINAL_REVIEW_SYSTEM_PROMPT,
    SUPERVISOR_REVISION_SYSTEM_PROMPT,
    SUPERVISOR_SYSTEM_PROMPT,
)
from writer_agent.provider_errors import raise_if_retryable_provider_error
from writer_agent.schemas import (
    FinalReviewSchema,
    PlannedSubtaskSchema,
    SupervisorPlanSchema,
    SupervisorRevisionPlanSchema,
)
from writer_agent.state import FinalReview, SupervisorState
from writer_agent.workflow_events import workflow_event

DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_FINAL_RETRIES = 2
DEFAULT_MAX_REPLANS = 1


def initialise_task(state: SupervisorState) -> SupervisorState:
    """Create a clean workflow state with bounded recovery counters."""
    return {
        "task_id": str(uuid4()),
        "thread_id": state.get("thread_id", ""),
        "conversation_id": state.get("conversation_id", ""),
        "turn_number": state.get("turn_number", 1),
        "user_id": state.get("user_id"),
        "user_request": state.get("user_request", ""),
        "effective_request": state.get("user_request", ""),
        "memory_context": state.get(
            "memory_context",
            "No relevant saved memories.",
        ),
        "run_metadata": dict(state.get("run_metadata", {})),
        "planning_mode": state.get("planning_mode", "initial"),
        "previous_effective_request": state.get(
            "previous_effective_request", ""
        ),
        "previous_final_answer": state.get("previous_final_answer", ""),
        "previous_subtasks": list(state.get("previous_subtasks", [])),
        "previous_subtask_results": dict(
            state.get("previous_subtask_results", {})
        ),
        "reuse_previous_answer": False,
        "reused_agent_types": [],
        "status": "initialised",
        "error": None,
        "plan": "",
        "plan_confidence": 0.0,
        "subtasks": [],
        "current_subtask_id": None,
        "subtask_results": {},
        "review_reports": [],
        "final_review": None,
        "workflow_events": list(state.get("workflow_events", [])),
        "max_retries": DEFAULT_MAX_RETRIES,
        "final_retry_count": 0,
        "max_final_retries": DEFAULT_MAX_FINAL_RETRIES,
        "revision_feedback": [],
        "replan_count": 0,
        "max_replans": DEFAULT_MAX_REPLANS,
        "replan_requested": False,
        "replan_feedback": [],
        "escalation_reason": None,
        "final_answer": None,
    }


def _initial_planning_request(state: SupervisorState, request: str) -> str:
    """Build the initial/replan request while preserving existing behavior."""
    replan_feedback = state.get("replan_feedback", [])
    memory_context = state.get(
        "memory_context",
        "No relevant saved memories.",
    )
    if not replan_feedback:
        return f"""
User request:
{request}

Relevant saved memories:
{memory_context}
""".strip()
    feedback_text = "\n".join(f"- {issue}" for issue in replan_feedback)
    return f"""
User request:
{request}

Relevant saved memories:
{memory_context}

Replanning feedback from the reviewer:
{feedback_text}

Create a corrected plan that addresses this feedback.
""".strip()


def _run_initial_plan(state: SupervisorState, request: str):
    supervisor_llm = llm.with_structured_output(SupervisorPlanSchema)
    return supervisor_llm.invoke(
        [
            SystemMessage(SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(_initial_planning_request(state, request)),
        ]
    )


def _forced_revision_subtasks(
    user_message: str,
    planned: list[PlannedSubtaskSchema],
) -> list[PlannedSubtaskSchema]:
    """Conservatively add specialist work required by explicit trigger words."""
    normalized = " ".join(user_message.casefold().split())
    agent_types = [subtask.agent_type for subtask in planned]
    research_required = bool(
        re.search(
            (
                r"\b(latest|today|up-to-date|verify|fact-check)\b"
                r"|\bcurrent(?:\s+\w+){0,3}\s+(information|data|version|"
                r"news|pricing|status|facts?)\b"
            ),
            normalized,
        )
        or any(
            phrase in normalized
            for phrase in ("add sources", "cite sources", "source this")
        )
    )
    data_required = bool(
        re.search(r"\b(recalculate|calculate|compute)\b", normalized)
        or "new dataset" in normalized
    )

    result = list(planned)
    writing_index = len(result) - 1
    if research_required and "research" not in agent_types:
        result.insert(
            0,
            PlannedSubtaskSchema(
                agent_type="research",
                objective=(
                    "Gather or verify the current source-grounded information "
                    "required by the follow-up."
                ),
                expected_output="Relevant research notes with sources.",
                review_criteria=[
                    "Uses current relevant sources",
                    "Addresses the follow-up directly",
                ],
            ),
        )
        writing_index += 1
    if data_required and "data" not in agent_types:
        result.insert(
            writing_index,
            PlannedSubtaskSchema(
                agent_type="data",
                objective=(
                    "Perform the calculation or updated analysis requested in "
                    "the follow-up."
                ),
                expected_output="A checked analysis suitable for the writer.",
                review_criteria=[
                    "Uses the stated inputs",
                    "Calculations are internally consistent",
                ],
            ),
        )
    return result


def _revision_planning_request(state: SupervisorState) -> str:
    feedback = state.get("replan_feedback", [])
    return f"""
Previous effective request:
{state.get("previous_effective_request") or "Not available"}

Previous reviewed answer:
{state.get("previous_final_answer") or "Not available"}

Available passed artifacts:
{format_previous_artifacts_for_supervisor(state)}

New user message:
{state.get("user_request")}

Relevant saved memories:
{state.get("memory_context") or "No relevant saved memories."}

Reviewer feedback for replanning:
{chr(10).join(f"- {item}" for item in feedback) if feedback else "None"}
""".strip()


def _run_revision_plan(state: SupervisorState):
    supervisor_llm = llm.with_structured_output(SupervisorRevisionPlanSchema)
    return supervisor_llm.invoke(
        [
            SystemMessage(SUPERVISOR_REVISION_SYSTEM_PROMPT),
            HumanMessage(_revision_planning_request(state)),
        ]
    )


def supervisor_plan(state: SupervisorState) -> SupervisorState:
    """Generate an initial or artifact-aware revision execution plan."""
    user_request = state.get("user_request")
    if not user_request:
        return {
            "status": "failed",
            "error": "Missing required field: user_request",
            "plan": "",
            "plan_confidence": 0.0,
            "subtasks": [],
            "final_answer": None,
        }

    replan_feedback = state.get("replan_feedback", [])
    is_revision = state.get("planning_mode") == "revision"
    try:
        if is_revision:
            revision_plan = _run_revision_plan(state)
            if revision_plan.plan_confidence <= 0.5:
                plan = _run_initial_plan(state, revision_plan.effective_request)
                effective_request = revision_plan.effective_request
                runtime_subtasks = build_runtime_subtasks(plan.subtasks)
                reused_results = {}
                reused_types = []
                reuse_previous_answer = False
                revision_intent = "replace"
            else:
                planned_subtasks = _forced_revision_subtasks(
                    user_request,
                    revision_plan.subtasks,
                )
                planned_types = {
                    subtask.agent_type for subtask in planned_subtasks
                }
                replacing = revision_plan.intent == "replace"
                reuse_research = (
                    revision_plan.reuse_research
                    and "research" not in planned_types
                    and not replacing
                )
                reuse_data = (
                    revision_plan.reuse_data
                    and "data" not in planned_types
                    and "research" not in planned_types
                    and not replacing
                )
                reused_subtasks, reused_results, reused_types = (
                    build_reused_artifacts(
                        state,
                        reuse_research=reuse_research,
                        reuse_data=reuse_data,
                    )
                )
                new_subtasks = build_runtime_subtasks(
                    planned_subtasks,
                    start_index=len(reused_subtasks) + 1,
                )
                runtime_subtasks = [*reused_subtasks, *new_subtasks]
                effective_request = revision_plan.effective_request
                plan = revision_plan
                reuse_previous_answer = (
                    revision_plan.reuse_previous_answer
                    and bool(state.get("previous_final_answer"))
                    and not replacing
                )
                revision_intent = revision_plan.intent
        else:
            plan = _run_initial_plan(state, user_request)
            effective_request = user_request
            runtime_subtasks = build_runtime_subtasks(plan.subtasks)
            reused_results = {}
            reused_types = []
            reuse_previous_answer = False
            revision_intent = None
    except Exception as exc:
        raise_if_retryable_provider_error(exc)
        return {
            "status": "failed",
            "error": f"Supervisor plan generation failed: {exc}",
            "plan": "",
            "plan_confidence": 0.0,
            "subtasks": [],
            "escalation_reason": (
                "Supervisor could not produce a valid structured plan."
            ),
            "final_answer": None,
        }

    return {
        "status": "planning",
        "error": None,
        "plan": plan.plan,
        "plan_confidence": plan.plan_confidence,
        "effective_request": effective_request,
        "subtasks": runtime_subtasks,
        "current_subtask_id": None,
        "subtask_results": reused_results,
        "reuse_previous_answer": reuse_previous_answer,
        "reused_agent_types": reused_types,
        "final_review": None,
        "final_retry_count": 0,
        "revision_feedback": [],
        "replan_requested": False,
        "replan_feedback": [],
        "escalation_reason": None,
        "final_answer": None,
        **(
            {"revision_intent": revision_intent}
            if revision_intent is not None
            else {}
        ),
        "workflow_events": [
            workflow_event(
                "replan" if state.get("replan_count", 0) else "plan",
                "Revised plan"
                if state.get("replan_count", 0)
                else ("Revision plan" if is_revision else "Initial plan"),
                content=plan.plan,
                details=[
                    *[
                        f"Reviewer feedback: {issue}"
                        for issue in replan_feedback
                    ],
                    *[
                        f"Reused {agent_type} artifact"
                        for agent_type in reused_types
                    ],
                    *[
                        (
                            f"{index}. {subtask['agent_type'].title()}: "
                            f"{subtask['objective']}\n"
                            f"Expected output: {subtask['expected_output']}\n"
                            "Review criteria: "
                            + "; ".join(subtask["review_criteria"])
                        )
                        for index, subtask in enumerate(
                            [
                                subtask
                                for subtask in runtime_subtasks
                                if not subtask["id"].startswith("reused-")
                            ],
                            start=1,
                        )
                    ],
                ],
            )
        ],
    }


def route_after_planning(state: SupervisorState) -> str:
    """Choose execution or escalation from plan confidence and completeness."""
    if state.get("plan_confidence", 0) <= 0.5:
        return "escalate"
    if not state.get("subtasks"):
        return "escalate"
    return "execute"


def get_final_writing_subtask_id(state: SupervisorState) -> str | None:
    """Find the last passed writing subtask in planned execution order."""
    for subtask in reversed(state.get("subtasks", [])):
        if (
            subtask.get("agent_type") == "writing"
            and subtask.get("status") == "passed"
        ):
            return subtask["id"]
    return None


def get_latest_writing_content(state: SupervisorState) -> str | None:
    """Return valid content from the final passed writing subtask."""
    subtask_id = get_final_writing_subtask_id(state)
    if subtask_id is None:
        return None

    result = state.get("subtask_results", {}).get(subtask_id)
    if result is None or result.get("errors"):
        return None
    return result.get("output", {}).get("content")


def _final_review_update(
    review: FinalReview,
    **updates: object,
) -> SupervisorState:
    """Return final-review state together with its safe audit artifact."""
    action = review["action"]
    title = {
        "return": "Final review passed",
        "retry": "Final review requested a rewrite",
        "replan": "Final review requested replanning",
        "escalate": "Final review could not approve the document",
    }[action]
    content = (
        "The document met the final review criteria."
        if review["passed"]
        else "The document did not yet meet the final review criteria."
    )
    issues = [
        issue
        for issue in review.get("issues", [])
        if not issue.startswith("Final review generation failed:")
    ]
    result: SupervisorState = {
        "status": "reviewing",
        "final_review": review,
        "workflow_events": [
            workflow_event(
                "review",
                title,
                content=content,
                details=[
                    f"Score: {review['score']:.0%}",
                    *issues,
                ],
                decision=action,
            )
        ],
    }
    result.update(updates)
    return result


def final_review(state: SupervisorState) -> SupervisorState:
    """Review final writing and enforce retry and replan limits."""
    final_review_llm = llm.with_structured_output(FinalReviewSchema)

    if not all_subtasks_passed(state):
        incomplete = [
            subtask["id"]
            for subtask in state.get("subtasks", [])
            if subtask.get("status") != "passed"
        ]
        review: FinalReview = {
            "passed": False,
            "score": 0.0,
            "issues": [f"These subtasks did not pass: {', '.join(incomplete)}"],
            "action": "escalate",
        }
        return _final_review_update(
            review,
            escalation_reason="Final review found incomplete subtasks.",
            final_answer=None,
        )

    final_content = get_latest_writing_content(state)
    if not final_content:
        review = {
            "passed": False,
            "score": 0.0,
            "issues": ["No writing content was available for final review."],
            "action": "escalate",
        }
        return _final_review_update(
            review,
            escalation_reason="Final review could not find writing content.",
            final_answer=None,
        )

    try:
        llm_review = final_review_llm.invoke(
            [
                SystemMessage(content=FINAL_REVIEW_SYSTEM_PROMPT),
                HumanMessage(content=final_content),
            ]
        )
        review = {
            "passed": llm_review.passed,
            "score": llm_review.score,
            "issues": llm_review.issues,
            "action": llm_review.action,
        }
    except Exception as exc:
        raise_if_retryable_provider_error(exc)
        review = {
            "passed": False,
            "score": 0.0,
            "issues": [f"Final review generation failed: {exc}"],
            "action": "escalate",
        }
        return _final_review_update(
            review,
            escalation_reason=(
                "Final reviewer could not produce a valid decision."
            ),
            final_answer=None,
        )

    if (
        review["action"] == "retry"
        and state.get("final_retry_count", 0)
        >= state.get("max_final_retries", 0)
    ):
        review = {
            **review,
            "issues": [
                *review["issues"],
                "Final writing retry limit reached.",
            ],
            "action": "escalate",
        }
        return _final_review_update(
            review,
            escalation_reason="Final writing failed after retries.",
            final_answer=None,
        )

    if (
        review["action"] == "replan"
        and state.get("replan_count", 0) >= state.get("max_replans", 0)
    ):
        review = {
            **review,
            "issues": [
                *review["issues"],
                "Workflow replan limit reached.",
            ],
            "action": "escalate",
        }
        return _final_review_update(
            review,
            escalation_reason="Workflow failed after replanning.",
            final_answer=None,
        )

    return _final_review_update(review)


def route_after_final_review(state: SupervisorState) -> str:
    """Route the workflow according to the final review action."""
    review = state.get("final_review")
    if review is not None:
        return review.get("action", "escalate")
    return "escalate"


def prepare_final_retry(state: SupervisorState) -> SupervisorState:
    """Reopen final writing with reviewer feedback and no stale result."""
    subtask_id = get_final_writing_subtask_id(state)
    if subtask_id is None:
        return {
            "status": "escalated",
            "escalation_reason": (
                "Final retry was requested, but no passed writing subtask was found."
            ),
            "final_answer": None,
        }

    review = state.get("final_review")
    feedback = review.get("issues", []) if review else []
    results = dict(state.get("subtask_results", {}))
    results.pop(subtask_id, None)

    return {
        "status": "executing",
        "subtasks": [
            {
                **subtask,
                "status": "pending",
            }
            if subtask["id"] == subtask_id
            else subtask
            for subtask in state.get("subtasks", [])
        ],
        "current_subtask_id": None,
        "subtask_results": results,
        "final_review": None,
        "final_retry_count": state.get("final_retry_count", 0) + 1,
        "revision_feedback": feedback,
        "escalation_reason": None,
        "final_answer": None,
    }


def route_after_final_retry_preparation(state: SupervisorState) -> str:
    """Continue a prepared final retry unless preparation escalated."""
    return "escalate" if state.get("status") == "escalated" else "execute"


def route_after_specialists(state: SupervisorState) -> str:
    """Choose final review, supervisor replanning, or escalation."""
    if state.get("status") == "escalated":
        return "escalate"
    if not state.get("replan_requested"):
        return "review"
    if state.get("replan_count", 0) >= state.get("max_replans", 0):
        return "escalate"
    return "replan"


def prepare_replan(state: SupervisorState) -> SupervisorState:
    """Prepare bounded reviewer feedback for a new supervisor plan."""
    feedback = state.get("replan_feedback", [])
    if not feedback:
        review = state.get("final_review")
        feedback = review.get("issues", []) if review else []

    return {
        "status": "planning",
        "replan_count": state.get("replan_count", 0) + 1,
        "replan_requested": True,
        "replan_feedback": feedback,
        "current_subtask_id": None,
        "final_answer": None,
    }


def return_response(state: SupervisorState) -> SupervisorState:
    """Complete the workflow with content from the final writing subtask."""
    final_answer = get_latest_writing_content(state)
    if final_answer is None:
        final_answer = "The task completed, but no writing output was produced."
    return {"status": "completed", "final_answer": final_answer}


def escalate(state: SupervisorState) -> SupervisorState:
    """End the workflow without exposing a final answer."""
    return {
        "status": "escalated",
        "escalation_reason": (
            state.get("escalation_reason") or "Workflow escalated for review."
        ),
        "final_answer": None,
    }
