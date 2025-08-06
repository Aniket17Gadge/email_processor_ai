from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated


from typing import Optional, Dict, Any
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """Enhanced AgentState to handle multi-agent workflow with session management"""
    session_id: str
    original_message: str
    current_message: str
    classification: Optional[str]  # email, general_query, spam_or_empty
    intent: Optional[str]  # complaint, query, request, update
    extracted_fields: Optional[Dict[str, Any]]  # name, email, date, issue
    response_draft: Optional[str]
    error: Optional[str]
    current_agent: Optional[str] 

# # Define the AgentState with the messages annotation
# class AgentState(TypedDict):
#     messages: str