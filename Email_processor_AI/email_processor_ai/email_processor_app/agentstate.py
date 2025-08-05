from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

# Define the AgentState with the messages annotation
class AgentState(TypedDict):
    messages: str