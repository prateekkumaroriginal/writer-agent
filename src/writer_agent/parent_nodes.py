"""Nodes and routers owned by the parent supervisor graph."""

from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from writer_agent.helpers import all_subtasks_passed, build_runtime_subtasks
from writer_agent.model import llm
from writer_agent.prompts import FINAL_REVIEW_SYSTEM_PROMPT, SUPERVISOR_SYSTEM_PROMPT
from writer_agent.provider_errors import raise_if_retryable_provider_error
from writer_agent.schemas import FinalReviewSchema, SupervisorPlanSchema
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
        "user_id": state.get("user_id"),
        "user_request": state.get("user_request", ""),
        "run_metadata": dict(state.get("run_metadata", {})),
        "status": "initialised",
        "error": None,
        "plan": "",
        "plan_confidence": 0.0,
        "subtasks": [],
        "current_subtask_id": None,
        "subtask_results": {},
        "review_reports": [],
        "final_review": None,
        "workflow_events": [],
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


def supervisor_plan(state: SupervisorState) -> SupervisorState:
    """Generate an executable plan, incorporating replan feedback when present."""
    supervisor_llm = llm.with_structured_output(SupervisorPlanSchema)
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
    planning_request = user_request
    if replan_feedback:
        feedback_text = "\n".join(f"- {issue}" for issue in replan_feedback)
        planning_request = f"""
User request:
{user_request}

Replanning feedback from the reviewer:
{feedback_text}

Create a corrected plan that addresses this feedback.
""".strip()

    try:
        plan = supervisor_llm.invoke(
            [
                SystemMessage(SUPERVISOR_SYSTEM_PROMPT),
                HumanMessage(planning_request),
            ]
        )
        runtime_subtasks = build_runtime_subtasks(plan.subtasks)
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
        "subtasks": runtime_subtasks,
        "current_subtask_id": None,
        "subtask_results": {},
        "final_review": None,
        "final_retry_count": 0,
        "revision_feedback": [],
        "replan_requested": False,
        "replan_feedback": [],
        "escalation_reason": None,
        "final_answer": None,
        "workflow_events": [
            workflow_event(
                "replan" if state.get("replan_count", 0) else "plan",
                "Revised plan"
                if state.get("replan_count", 0)
                else "Initial plan",
                content=plan.plan,
                details=[
                    *[
                        f"Reviewer feedback: {issue}"
                        for issue in replan_feedback
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
                            runtime_subtasks,
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
