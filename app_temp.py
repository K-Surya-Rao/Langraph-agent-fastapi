"""
FastAPI application for LangGraph Agent with BODMA and CODMA Mathematical Tools
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any
import os
from typing_extensions import TypedDict
from typing import Annotated, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="LangGraph BODMA/CODMA Agent API",
    description="API for mathematical calculations using BODMA and CODMA tools",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment
google_key = os.getenv("GOOGLE_API_KEY")

if not google_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Define the tools
@tool
def BODMA(a: float, b: float) -> float:
    """
    BODMA: Calculates (a^b) / (a * b)
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Result of (a to the power b) divided by (a multiplied by b)
    """
    if a == 0 or b == 0:
        return "Error: Neither a nor b can be zero for BODMA calculation"
    
    result = (a ** b) / (a * b)
    return result

@tool
def CODMA(a: float, b: float) -> float:
    """
    CODMA: Calculates (a * b) / (a^b)
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Result of (a multiplied by b) divided by (a to the power b)
    """
    if a ** b == 0:
        return "Error: a^b cannot be zero for CODMA calculation"
    
    result = (a * b) / (a ** b)
    return result

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the tools list
tools = [BODMA, CODMA]

# Initialize the LLM with tools
# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    google_api_key=google_key
)
llm_with_tools = llm.bind_tools(tools)

# Define the agent node
def agent(state: State):
    """The agent node that calls the LLM"""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# Define the tool node
def tool_node(state: State):
    """Execute the tools based on the agent's tool calls"""
    messages = state["messages"]
    last_message = messages[-1]
    
    outputs = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Find and execute the appropriate tool
        if tool_name == "BODMA":
            result = BODMA.invoke(tool_args)
        elif tool_name == "CODMA":
            result = CODMA.invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"
        
        outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": outputs}

# Define routing logic
def should_continue(state: State) -> Literal["tools", "end"]:
    """Determine whether to continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, end
    return "end"

# Build the graph
def create_agent_graph():
    """Create and compile the LangGraph agent"""
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app_graph = workflow.compile()
    return app_graph

# Initialize the agent graph once at startup
agent_graph = create_agent_graph()

# Request and Response models
class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Calculate BODMA for a=2 and b=3"
            }
        }
    )
    
    message: str

class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "The BODMA calculation for a=2 and b=3 is 1.333",
                "success": True
            }
        }
    )
    
    response: str
    success: bool

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "LangGraph BODMA/CODMA Agent API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Send a message to the agent",
            "/health": "GET - Check API health status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LangGraph Agent API"
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint to interact with the LangGraph agent
    
    Send a message and get a response from the agent that can use BODMA and CODMA tools.
    
    Example requests:
    - "Calculate BODMA for a=2 and b=3"
    - "Calculate CODMA for a=5 and b=2"
    - "Compare BODMA and CODMA for a=3 and b=2"
    """
    try:
        # Invoke the agent with the user's message
        result = agent_graph.invoke({
            "messages": [HumanMessage(content=request.message)]
        })
        
        # Extract the final AI response
        response_text = ""
        for message in result["messages"]:
            if isinstance(message, AIMessage) and message.content:
                response_text = message.content
        
        if not response_text:
            response_text = "I processed your request but didn't generate a text response."
        
        return ChatResponse(
            response=response_text,
            success=True
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)