# this is the state schema used by the prebuilt create_react_agent we'll be using below
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from typing import List
from typing_extensions import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

class State(AgentState):
    docs: List[str]


@tool
def get_context(question: str, state: Annotated[dict, InjectedState]):
    """Get relevant context for answering the question."""
    print("THIS IS STATE!!!")
    pprint(state["docs"])
    return "\n\n".join(doc for doc in state["docs"])


print(get_context.get_input_schema().model_json_schema())
print(get_context.tool_call_schema.model_json_schema())


model = ChatOpenAI(model="gpt-4", temperature=0)
tools = [get_context]

# ToolNode will automatically take care of injecting state into tools
tool_node = ToolNode(tools)

checkpointer = MemorySaver()
graph = create_react_agent(model, tools, state_schema=State, checkpointer=checkpointer)

# Create initial state with some test documents
initial_state = {
    "messages": [{"type": "user", "content": "what's the latest news about FooBar"}],
    "docs": [
        "FooBar announced new product launch in 2024",
        "FooBar stock prices increased by 10%",
    ]
}

# Invoke with initial state
answer = graph.invoke(
    initial_state,
    config={"configurable": {"thread_id": "1", "user_id": "1"}},
)

print("Final Answer:", answer)