from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from .agentstate import AgentState
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("CHATGROQ_API")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_key)

def analyzer_agent(state: AgentState) -> AgentState:
    """Agent 1: Analyzes if input is email or general query"""
    message = state["original_message"]
    
    messages = [
        SystemMessage(
            content=f"""You are an email analyzer agent.
            Classify the following text as one of:
            - email (if it looks like an email with sender info, subject-like content, or typical email format)
            - general_query (if it's a general question not in email format)
            - spam_or_empty (if it's spam, empty, or meaningless content)
            
            Text: "{message}"
            
            Respond with ONLY one word: email, general_query, or spam_or_empty.
            """
        ),
        HumanMessage(content=message)
    ]
    
    response = llm.invoke(messages).content.strip().lower()
    print(f"Analyzer Agent Response: {response}")
    
    return {
        **state,
        "classification": response,
        "current_agent": "analyzer",
        "current_message": message
    }

def intent_classifier_agent(state: AgentState) -> AgentState:
    """Agent 2: Classifies email intent"""
    message = state["original_message"]
    
    messages = [
        SystemMessage(
            content=f"""You are an email intent classifier.
            Classify the email intent as one of:
            - complaint (user reporting problems, issues, dissatisfaction)
            - query (asking questions, seeking information)
            - request (asking for something to be done, services, help)
            - update (providing information, status updates, notifications)
            
            Email content: "{message}"
            
            Respond with ONLY one word: complaint, query, request, or update.
            """
        ),
        HumanMessage(content=message)
    ]
    
    response = llm.invoke(messages).content.strip().lower()
    print(f"Intent Classifier Response: {response}")
    
    return {
        **state,
        "intent": response,
        "current_agent": "intent_classifier"
    }

def field_extractor_agent(state: AgentState) -> AgentState:
    """Agent 3: Extracts key fields from email"""
    message = state["original_message"]
    
    messages = [
        SystemMessage(
            content=f"""You are an information extraction agent.Your task is to extract and return the following fields from the given email in strict **JSON format**:
            - `name`: The full name of the sender. Check the sign-off (e.g., "Thanks, John Doe") or beginning ("Hi, I’m Jane")
            - `email`: The sender’s email address. Look for patterns like "name@example.com".
            - `date`: The date mentioned in the email body or header. Accept formats like:
            - "August 3, 2025"
            - "03/08/2025"
            - "3rd Aug 2025"
            - "2025-08-03"
            Standardize all extracted dates to **"YYYY-MM-DD"** format.
            - `issue`: Identify the **main issue, concern, or request** described in the email. Use a short phrase (5–12 words). Read the body and summarize the main problem or intent in the user's words. Ignore greetings or pleasantries.
            Additional Rules:
            - If any field is not found, set it as `null`.
            - Return output in proper JSON: keys and string values in **double quotes**, no trailing commas.
            this is the email:\"\"\"{message}\"\"\"
            
            Example output:
            {{
                  "name": "Ravi Kumar",
                    "email": "ravi.kumar@gmail.com",
                    "date": "2025-08-01",
                    "issue": "Unable to reset password for my account"
                    }}
                    Only return the JSON object. Do not include explanations, markdown, or formatting."""),
    HumanMessage(content=message)
    ]
    
    response = llm.invoke(messages).content.strip()
    print(f"Field Extractor Raw Response: {response}")
    
    try:
        # Try to parse JSON from response
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '').strip()
        elif response.startswith('```'):
            response = response.replace('```', '').strip()
            
        extracted_fields = json.loads(response)
    except json.JSONDecodeError:
        # Fallback extraction using regex if JSON parsing fails
        extracted_fields = {
            "name": None,
            "email": None,
            "date": None,
            "issue": "Could not extract specific issue details"
        }
        
        # Try to extract email with regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, message)
        if email_match:
            extracted_fields["email"] = email_match.group()
    
    print(f"Extracted Fields: {extracted_fields}")
    
    return {
        **state,
        "extracted_fields": extracted_fields,
        "current_agent": "field_extractor"
    }

def response_generator_agent(state: AgentState) -> AgentState:
    """Agent 4: Generates draft response based on intent and extracted fields"""
    intent = state.get("intent", "")
    extracted_fields = state.get("extracted_fields", {})
    
    name = extracted_fields.get("name", "Customer")
    issue = extracted_fields.get("issue", "your inquiry")
    
    messages = [
        SystemMessage(
            content=f"""You are a customer service response generator.
            Generate a professional, empathetic email response based on:
            - Intent: {intent}
            - Customer name: {name}
            - Issue: {issue}
            
            Guidelines:
            - Be professional and empathetic
            - Address the customer by name if available
            - Acknowledge their concern/request
            - Provide helpful next steps or solutions
            - Keep it concise but comprehensive
            - For complaints: apologize and offer solutions
            - For queries: provide helpful information
            - For requests: acknowledge and explain next steps
            - For updates: thank them and confirm receipt
            
            Generate a draft email response (just the body, no subject line needed).
            """
        ),
        HumanMessage(content=f"Generate response for {intent} from {name} about: {issue}")
    ]
    
    response = llm.invoke(messages).content.strip()
    print(f"Response Generator: {response}")
    
    return {
        **state,
        "response_draft": response,
        "current_agent": "response_generator"
    }

def router(state: AgentState) -> str:
    """Router function to determine next agent"""
    current_agent = state.get("current_agent")
    classification = state.get("classification")
    
    if not current_agent:
        return "analyzer"
    
    if current_agent == "analyzer":
        if classification == "email":
            return "intent_classifier"
        elif classification == "general_query":
            return "general_response"
        else:  # spam_or_empty
            return "spam_response"
    
    elif current_agent == "intent_classifier":
        return "field_extractor"
    
    elif current_agent == "field_extractor":
        return "response_generator"
    
    return "end"

def general_response_agent(state: AgentState) -> AgentState:
    """Handles general queries that are not emails"""
    return {
        **state,
        "error": "I am an email processor. I can only help with email-related tasks. Please provide an email for processing.",
        "current_agent": "general_response"
    }

def spam_response_agent(state: AgentState) -> AgentState:
    """Handles spam or empty content"""
    return {
        **state,
        "error": "The provided content appears to be spam or empty. Please provide a valid email for processing.",
        "current_agent": "spam_response"
    }