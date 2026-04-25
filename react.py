from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch


load_dotenv()


@tool
def triple(num: float) -> float:
    """Returns the triple of a number."""
    return num * 3

tools = [TavilySearch(max_results=1), triple]

llm = init_chat_model(f"ollama:qwen2.5-coder:0.5b", temperature=0.9).bind_tools(tools)