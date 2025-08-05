from rest_framework.response import Response
from rest_framework.decorators import api_view
from .graph import start_graph  # Import the graph function

@api_view(['POST'])
def ai_response(request):
    # Extract user message from JSON request
    user_message = request.data.get("messages", "")

    if not user_message:
        return Response({"error": "No message provided"}, status=400)

    # Create a LangGraph state with the message
    # Ensure that state["messages"] is a list of strings
    state = {"messages": [user_message]}

    # Get compiled graph and invoke it
    compiled_graph = start_graph()
    response = compiled_graph.invoke(state)

    # Extract the AI response from the returned state
    ai_message = response.get("messages", "")

    return Response({"response": ai_message})