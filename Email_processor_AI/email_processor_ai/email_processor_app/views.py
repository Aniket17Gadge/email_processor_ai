from rest_framework.response import Response
from rest_framework.decorators import api_view
from .graph import create_email_processor_graph
from .agentstate import AgentState
import asyncio
import uuid
from datetime import datetime

@api_view(['POST'])
def ai_response(request):
    """
    Main API endpoint for processing emails with multi-agent system
    
    Expected input:
    {
        "session_id": "optional_session_id",
        "message": "email content or query"
    }
    
    Returns:
    {
        "session_id": "session_identifier",
        "classification": "email/general_query/spam_or_empty",
        "intent": "complaint/query/request/update",
        "name": "extracted_name",
        "email": "extracted_email",
        "date": "extracted_date",
        "issue": "extracted_issue_description",
        "response_draft": "generated_response",
        "error": "error_message_if_any"
    }
    """
    
    try:
        # Extract data from request
        user_message = request.data.get("message", "").strip()
        session_id = request.data.get("session_id") or str(uuid.uuid4())
        
        if not user_message:
            return Response({
                "error": "No message provided",
                "session_id": session_id
            }, status=400)
        
        # Create initial state
        initial_state = AgentState(
            session_id=session_id,
            original_message=user_message,
            current_message=user_message,
            classification=None,
            intent=None,
            extracted_fields=None,
            response_draft=None,
            error=None,
            current_agent=None
        )
        
        # Get compiled graph
        compiled_graph = create_email_processor_graph()
        
        # Configuration for graph execution with memory
        config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "email_processor"
            }
        }
        
        print(f"Processing message for session {session_id}: {user_message[:100]}...")
        
        # Execute the graph
        try:
            final_state = compiled_graph.invoke(initial_state, config=config)
        except Exception as graph_error:
            print(f"Graph execution error: {graph_error}")
            # Fallback to execution without memory
            final_state = compiled_graph.invoke(initial_state)
        
        print(f"Final state: {final_state}")
        
        # Prepare response based on classification
        response_data = {
            "session_id": session_id,
            "classification": final_state.get("classification"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Handle different classifications
        classification = final_state.get("classification")
        
        if classification == "email":
            # Full email processing response
            extracted_fields = final_state.get("extracted_fields", {})
            response_data.update({
                "intent": final_state.get("intent"),
                "name": extracted_fields.get("name"),
                "email": extracted_fields.get("email"),
                "date": extracted_fields.get("date"),
                "issue": extracted_fields.get("issue"),
                "response_draft": final_state.get("response_draft")
            })
        else:
            # General query or spam response
            response_data["error"] = final_state.get("error")
            
        return Response(response_data, status=200)
        
    except Exception as e:
        print(f"Unexpected error in ai_response: {e}")
        return Response({
            "error": f"An unexpected error occurred: {str(e)}",
            "session_id": session_id if 'session_id' in locals() else str(uuid.uuid4())
        }, status=500)


@api_view(['GET'])
def get_session_history(request, session_id):
    """
    Get conversation history for a specific session
    """
    try:
        compiled_graph = get_compiled_graph_sync()
        
        config = {
            "configurable": {
                "thread_id": session_id,
                "checkpoint_ns": "email_processor"
            }
        }
        
        # Get conversation history
        try:
            history = []
            for state in compiled_graph.get_state_history(config):
                history.append({
                    "timestamp": state.created_at.isoformat() if state.created_at else None,
                    "state": state.values,
                    "next_step": state.next
                })
            
            return Response({
                "session_id": session_id,
                "history": history[:10]  # Limit to last 10 states
            }, status=200)
            
        except Exception as history_error:
            print(f"Error getting session history: {history_error}")
            return Response({
                "session_id": session_id,
                "error": "Could not retrieve session history",
                "message": "Memory feature may not be available"
            }, status=200)
            
    except Exception as e:
        print(f"Error in get_session_history: {e}")
        return Response({
            "error": f"An error occurred: {str(e)}"
        }, status=500)


@api_view(['GET'])
def health_check(request):
    """Health check endpoint"""
    return Response({
        "status": "healthy",
        "service": "email_processor_ai",
        "timestamp": datetime.now().isoformat()
    }, status=200)