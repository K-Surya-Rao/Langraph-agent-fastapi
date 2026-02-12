"""
LangGraph Agent with BODMA and CODMA Mathematical Tools
"""

from typing import Annotated, Literal
import os
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from dotenv import load_dotenv

load_dotenv()

google_key = os.getenv("GOOGLE_API_KEY")

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
    app = workflow.compile()
    return app


# Main execution
def main():
    """Run the agent with example queries"""
    app = create_agent_graph()
    
    print("=" * 60)
    print("LangGraph Agent with BODMA and CODMA Tools")
    print("=" * 60)
    
    # Example 1: Using BODMA
    print("\n📊 Example 1: Calculate BODMA for a=2, b=3")
    print("-" * 60)
    
    result = app.invoke({
        "messages": [HumanMessage(content="Calculate BODMA for a=2 and b=3")]
    })
    
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"🤖 Agent: {message.content}")
    
    # Example 2: Using CODMA
    print("\n📊 Example 2: Calculate CODMA for a=2, b=3")
    print("-" * 60)
    
    result = app.invoke({
        "messages": [HumanMessage(content="Calculate CODMA for a=2 and b=3")]
    })
    
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"🤖 Agent: {message.content}")
    
    # Example 3: Using both tools
    print("\n📊 Example 3: Compare BODMA and CODMA for a=3, b=2")
    print("-" * 60)
    
    result = app.invoke({
        "messages": [HumanMessage(content="Calculate both BODMA and CODMA for a=3 and b=2, then compare the results")]
    })
    
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print(f"🤖 Agent: {message.content}")
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        user_input = input("\n💬 You: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })
        
        for message in result["messages"]:
            if isinstance(message, AIMessage):
                print(f"🤖 Agent: {message.content}")


if __name__ == "__main__":
    main()