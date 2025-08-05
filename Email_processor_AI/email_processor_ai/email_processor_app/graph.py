from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from .agents import Ai_bot  # Make sure to import Ai_bot correctly
from .agentstate import AgentState

# Function to initialize the state graph and define the entry point
def start_graph():
    # Initialize the state graph with AgentState
    graph = StateGraph(AgentState)

    # Add the Ai_bot node to the graph
    graph.add_node("agent", Ai_bot)

    # Set the entry point of the graph to the "agent" node
    graph.set_entry_point("agent")

    # Define the end state for the agent node
    graph.add_edge("agent", END)

    # Compile and return the graph
    return graph.compile()