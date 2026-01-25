import langchain
print(f"LangChain Version: {langchain.__version__}")
import langchain.agents
print(f"AgentExecutor in langchain.agents: {'AgentExecutor' in dir(langchain.agents)}")
try:
    from langchain.agents import AgentExecutor
    print("Successfully imported AgentExecutor")
except ImportError as e:
    print(f"ImportError: {e}")

try:
    from langchain.agents import create_tool_calling_agent
    print("Successfully imported create_tool_calling_agent")
except ImportError as e:
    print(f"ImportError for create_tool_calling_agent: {e}")
