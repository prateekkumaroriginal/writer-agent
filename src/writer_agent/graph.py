"""LangGraph assembly for the writer workflow."""

from langgraph.graph import END, START, StateGraph

from writer_agent.parent_nodes import (
    escalate,
    final_review,
    initialise_task,
    prepare_final_retry,
    prepare_replan,
    return_response,
    route_after_final_retry_preparation,
    route_after_final_review,
    route_after_planning,
    route_after_specialists,
    supervisor_plan,
)
from writer_agent.specialist_nodes import (
    data_agent,
    escalate_current_subtask,
    mark_subtask_complete,
    mark_subtask_failed,
    pick_next_subtask,
    research_agent,
    request_replan,
    retry_subtask,
    review_agent,
    route_after_pick_next_subtask,
    route_after_subtask_review,
    writing_agent,
)
from writer_agent.state import SupervisorState


def build_specialist_graph():
    """Build the graph that executes and reviews specialist subtasks."""
    builder = StateGraph(SupervisorState)
    builder.add_node("pick_next_subtask", pick_next_subtask)
    builder.add_node("research_agent", research_agent)
    builder.add_node("data_agent", data_agent)
    builder.add_node("writing_agent", writing_agent)
    builder.add_node("review_agent", review_agent)
    builder.add_node("mark_subtask_complete", mark_subtask_complete)
    builder.add_node("retry_subtask", retry_subtask)
    builder.add_node("mark_subtask_failed", mark_subtask_failed)
    builder.add_node("escalate_current_subtask", escalate_current_subtask)
    builder.add_node("request_replan", request_replan)

    builder.add_edge(START, "pick_next_subtask")
    builder.add_conditional_edges(
        "pick_next_subtask",
        route_after_pick_next_subtask,
        {
            "research": "research_agent",
            "data": "data_agent",
            "writing": "writing_agent",
            "done": END,
        },
    )
    builder.add_edge("research_agent", "review_agent")
    builder.add_edge("data_agent", "review_agent")
    builder.add_edge("writing_agent", "review_agent")
    builder.add_conditional_edges(
        "review_agent",
        route_after_subtask_review,
        {
            "pass": "mark_subtask_complete",
            "retry": "retry_subtask",
            "fail": "mark_subtask_failed",
            "replan": "request_replan",
            "escalate": "escalate_current_subtask",
        },
    )
    builder.add_edge("mark_subtask_complete", "pick_next_subtask")
    builder.add_edge("retry_subtask", "pick_next_subtask")
    builder.add_edge("mark_subtask_failed", END)
    builder.add_edge("escalate_current_subtask", END)
    builder.add_edge("request_replan", END)
    return builder.compile()


def build_supervisor_graph():
    """Build the parent graph that plans, executes, reviews, and returns work."""
    builder = StateGraph(SupervisorState)
    builder.add_node("initialise_task", initialise_task)
    builder.add_node("supervisor_plan", supervisor_plan)
    builder.add_node("execute_specialists", build_specialist_graph())
    builder.add_node("final_review", final_review)
    builder.add_node("prepare_final_retry", prepare_final_retry)
    builder.add_node("prepare_replan", prepare_replan)
    builder.add_node("return_response", return_response)
    builder.add_node("escalate", escalate)

    builder.add_edge(START, "initialise_task")
    builder.add_edge("initialise_task", "supervisor_plan")
    builder.add_conditional_edges(
        "supervisor_plan",
        route_after_planning,
        {
            "execute": "execute_specialists",
            "escalate": "escalate",
        },
    )
    builder.add_conditional_edges(
        "execute_specialists",
        route_after_specialists,
        {
            "review": "final_review",
            "replan": "prepare_replan",
            "escalate": "escalate",
        },
    )
    builder.add_conditional_edges(
        "final_review",
        route_after_final_review,
        {
            "return": "return_response",
            "retry": "prepare_final_retry",
            "replan": "prepare_replan",
            "escalate": "escalate",
        },
    )
    builder.add_conditional_edges(
        "prepare_final_retry",
        route_after_final_retry_preparation,
        {
            "execute": "execute_specialists",
            "escalate": "escalate",
        },
    )
    builder.add_edge("prepare_replan", "supervisor_plan")
    builder.add_edge("return_response", END)
    builder.add_edge("escalate", END)
    return builder.compile()
