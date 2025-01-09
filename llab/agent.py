from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from llab.utils.nodes import call_model, should_continue, tool_node # import nodes
from llab.utils.state import AgentState # import state

# Define the config
class GraphConfig(TypedDict):
    model_name: Literal["anthropic", "openai"]

workflow = StateGraph(AgentState, config_schema=GraphConfig)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)
workflow.add_edge("action", "agent")

graph = workflow.compile()