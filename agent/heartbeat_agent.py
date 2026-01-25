import os
import json
from typing import Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool, tool

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# --- Retriever setup ---
def build_retriever(json_path: str):
    """Load JSON, split into chunks, and return a retriever."""
    with open(json_path, "r") as f:
        report_data = json.load(f)

    # Convert JSON to pretty string
    report_text = json.dumps(report_data, indent=2)

    # Wrap into LangChain Document
    docs = [Document(page_content=report_text, metadata={"source": "heartbeat_report"})]

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    doc_chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    # Vectorstore + retriever
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore.as_retriever()


# --- Tool wrapper ---
def make_retriever_tool(retriever):
    
    @tool
    def heart_retriever_tool(query: str) -> str:
        """
        Retrieve information from heartbeat_report.json.
        Call this tool whenever a user asks about heartbeat analysis results.
        Returns the content of relevant documents.
        """
        docs = retriever.get_relevant_documents(query)
        # Combine docs into a single string for valid tool output
        return "\n\n".join([d.page_content for d in docs])

    return heart_retriever_tool


# --- Agent factory ---
def build_heartbeat_agent(json_path: str):
    retriever = build_retriever(json_path)
    # The tool must be a list for the agent
    retriever_tool = make_retriever_tool(retriever)
    tools = [retriever_tool]

    # Initialize Groq Chat Model
    # Ensuring GROQ_API_KEY is in env is the user's responsibility or handled by load_dotenv if present
    llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        temperature=0
    )

    # System Message
    system_message = """
You are HeartbeatAnalysisAgent, an expert assistant for analyzing heartbeat recordings.
You have access to pre-computed JSON reports via your tools.

Your role:
- Answer user questions using ONLY the data provided in the JSON report or general definitions of the metrics.
- If asked about medical interpretation, provide general information only (e.g., what a high HRV usually means), not medical advice.
- If a query is unrelated to heartbeat analysis, classification, or signal processing, reply with: "I don't know."
- Do not make up or hallucinate answers.
- If user greets you, greet back.
- Keep explanations clear and concise.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Construct the agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor
