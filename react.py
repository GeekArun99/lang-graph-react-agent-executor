import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentConfig:
    MODEL_NAME = "llama-3.1-8b-instant"
    TEMPERATURE = 0
    MAX_SEARCH_RESULTS = 1
    MAX_RETRIES = 3
    REQUEST_TIMEOUT = 30


def validate_env_vars() -> None:
    required_vars = ["GROQ_API_KEY", "TAVILY_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")


@tool
def triple(num: float) -> float:
    """Triple a number.
    
    Args:
        num: The number to triple
        
    Returns:
        The triple of the input number
    """
    try:
        return float(num) * 3
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid input to triple: {num}, error: {e}")
        raise ValueError(f"Expected a number, got: {num}") from e


def create_tavily_tool() -> TavilySearch:
    try:
        return TavilySearch(max_results=AgentConfig.MAX_SEARCH_RESULTS)
    except Exception as e:
        logger.error(f"Failed to initialize TavilySearch: {e}")
        raise


def create_llm() -> ChatGroq:
    validate_env_vars()
    try:
        return ChatGroq(
            model=AgentConfig.MODEL_NAME,
            temperature=AgentConfig.TEMPERATURE,
            max_retries=AgentConfig.MAX_RETRIES,
            request_timeout=AgentConfig.REQUEST_TIMEOUT,
        )
    except Exception as e:
        logger.error(f"Failed to initialize ChatGroq: {e}")
        raise


tools = [create_tavily_tool(), triple]

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", 
    "You are a helpful assistant that can use tools to answer questions.\n"
    "Available tools: {tool_names}\n"
    "Always think step by step before calling tools.\n"
    "If a tool fails, try an alternative approach or ask for clarification."
    ),
    MessagesPlaceholder(variable_name="messages"),
])

llm = create_llm().bind_tools(tools)