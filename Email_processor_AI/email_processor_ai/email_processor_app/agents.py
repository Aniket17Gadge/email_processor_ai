from langchain_ollama import ChatOllama
from .graph import *  # Ensure that this import is correct and does not cause circular import
from .agentstate import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
groq_key = os.getenv("CHATGROQ_API")

llm = ChatGroq(model="llama-3.1-8b-instant",api_key=groq_key)
# Initialize the model
# llm = ChatOllama(model="llama3.2:1b")

# Define the Ai_bot function
def Ai_bot(state: AgentState):
    # Ensure state["messages"] is properly formatted
    message = state["messages"]  # Retrieve the messages list from AgentState
    messages = [
    SystemMessage(
        content=f"""You are an email analyzer agent.
        Classify the following text as one of:
        - email
        - general_query
        - spam_or_empty
        Text:
        \"\"\"{message}\"\"\"
        Respond with only one word: email, general_query, or spam_or_empty.
        """
    ),
    HumanMessage(
        content=f"""{message}"""
    )
]

    # Get the model response
    response = llm.invoke((messages)).content
    print("this is ai generate :",response)

    # Return the response in the expected format
    return {"messages": response}