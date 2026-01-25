import os
import json
from typing import Optional
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.tools import Tool, tool
from langgraph.prebuilt import create_react_agent


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
        docs = retriever.invoke(query)
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
        model_name="llama-3.3-70b-versatile",
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

    # Create the agent using LangGraph
    # In this version, system_message is passed via 'prompt'
    agent = create_react_agent(llm, tools, prompt=system_message)


    return agent

