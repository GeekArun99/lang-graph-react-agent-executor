"""
Main entry point for the React LangGraph agent with function calling.
"""
from langgraph.graph.message import MessagesState


import os
import logging
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END

from nodes import run_agent_reasoning, tool_node

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1


def should_continue(state: MessagesState) -> Literal["act", "end"]:
    """
    Determine whether to continue to tool execution or end the graph.

    Args:
        state: Current message state

    Returns:
        "act" if last message has tool calls, "end" otherwise
    """
    messages = state.get("messages", [])
    if not messages:
        logger.warning("No messages in state, ending graph")
        return END

    last_message = messages[LAST]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return END

    return ACT


def build_graph() -> StateGraph:
    """Build and compile the LangGraph state graph."""
    flow = StateGraph(MessagesState)

#reasoning nodes
    #reasoning node is the node that will reason about the input and output the next node to execute
    flow.add_node(AGENT_REASON, run_agent_reasoning)
    #set the entry point to the agent_reason node
    flow.set_entry_point(AGENT_REASON)
    #add the tool node
    flow.add_node(ACT, tool_node)

    #the conditional edges are the edges that will be taken based on the condition as per the agent reason node output
    flow.add_conditional_edges( 
        AGENT_REASON,
        should_continue,
        {END: END, ACT: ACT},
    )
    #add the edge from act to agent_reason (back from ACT to AGENT_REASON)
    flow.add_edge(ACT, AGENT_REASON) 

    return flow.compile()


def draw_graph(app, output_path: str = "flow.png") -> None:
    """Draw the graph as Mermaid PNG if graphviz is available."""
    try:
        app.get_graph().draw_mermaid_png(output_file_path=output_path)
        logger.info(f"Graph diagram saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not generate graph diagram: {e}")


def main() -> None:
    """Run the agent with a sample input."""
    if not os.getenv("GROQ_API_KEY"):
        logger.error("GROQ_API_KEY not set in environment")
        return

    app = build_graph()
    draw_graph(app)

    initial_input = {"messages": [HumanMessage(content="What is the current temperature raibag right now..? List it and triple it.")]}

    try:
        result = app.invoke(initial_input)
        logger.info(f"Final state: {result["messages"][LAST].content}")
    except Exception as e:
        logger.exception(f"Error during graph execution: {e}")


if __name__ == "__main__":
    main()