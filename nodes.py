from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are helpful assistant that can use tools to answer questions.
"""

#runnable nodes for the react agent
def run_agent_reasoning(state : MessagesState)-> MessagesState:
    """
        Run the agent reasoning node.
    """
    response = llm.invoke([{"role" : "system", "content":SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages" : [response]}

#tool node to execute the tolls which are decided by llm 
tool_node = ToolNode(tools)