import sys
import asyncio
from langgraph.graph import StateGraph, END
from .agents import (
    analyzer_agent, 
    intent_classifier_agent, 
    field_extractor_agent, 
    response_generator_agent,
    general_response_agent,
    spam_response_agent,
    router
)
from .agentstate import AgentState
from .memory import *

# # ✅ Fix for Windows event loop issues with psycopg async
# if sys.platform.startswith("win"):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def create_email_processor_graph():
    """Define the full multi-agent graph logic without compiling."""
    
    # Initialize the state graph
    graph = StateGraph(AgentState)
    
    # Add agent nodes
    graph.add_node("analyzer", analyzer_agent)
    graph.add_node("intent_classifier", intent_classifier_agent)
    graph.add_node("field_extractor", field_extractor_agent)
    graph.add_node("response_generator", response_generator_agent)
    graph.add_node("general_response", general_response_agent)
    graph.add_node("spam_response", spam_response_agent)
    
    # Entry point
    graph.set_entry_point("analyzer")
    
    # Conditional routing
    graph.add_conditional_edges(
        "analyzer",
        lambda state: router(state),
        {
            "intent_classifier": "intent_classifier",
            "general_response": "general_response",
            "spam_response": "spam_response"
        }
    )
    
    # Sequential logic
    graph.add_edge("intent_classifier", "field_extractor")
    graph.add_edge("field_extractor", "response_generator")
    
    # Final endpoints
    graph.add_edge("response_generator", END)
    graph.add_edge("general_response", END)
    graph.add_edge("spam_response", END)
    
    return graph.compile(checkpointer=memory)


