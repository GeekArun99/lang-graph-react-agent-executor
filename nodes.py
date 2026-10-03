from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
import logging

from react import llm, tools, agent_prompt

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_agent_reasoning(state: MessagesState) -> dict:
    """Run the agent reasoning node with error handling."""
    if not state.get("messages"):
        logger.warning("Empty messages in state")
        return {"messages": []}

    try:
        prompt_value = agent_prompt.invoke({
            "messages": state["messages"],
            "tool_names": ", ".join(t.name for t in tools)
        })
        response = llm.invoke(prompt_value)
        logger.debug(f"Agent response: {response}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise


tool_node = ToolNode(tools)
"""Tool execution node - runs tools selected by the LLM."""