import os
import json
from typing import Optional
from dotenv import load_dotenv

from typing import Optional
from dotenv import load_dotenv

# Heavy imports moved inside functions to reduce startup latency
from langchain_core.messages import SystemMessage


# Load environment variables
load_dotenv()
if hf_token := os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token

# --- Retriever setup ---
import streamlit as st

@st.cache_resource
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_data
def build_retriever(json_path: str):
    """Load JSON, split into chunks, and return a retriever."""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
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
    embeddings = get_embeddings()

    # Vectorstore + retriever
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore.as_retriever()


# --- Tool wrapper ---
def make_retriever_tool(retriever):
    from langchain_core.tools import tool
    
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
def build_heartbeat_agent(json_path: str, groq_api_key: Optional[str] = None, hf_token: Optional[str] = None):
    from langchain_groq import ChatGroq
    from langgraph.prebuilt import create_react_agent
    
    # Load report data directly for faster context injection
    try:
        with open(json_path, "r") as f:
            report_data = json.load(f)
        # Create a more compact and readable summary for the prompt
        report_summary = json.dumps(report_data, indent=2)
    except Exception as e:
        report_summary = f"Error loading report: {str(e)}"
    
    # Set HF token if provided
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Optional: Keep the tool for backward compatibility or complex queries
    # but the primary info is now in the system prompt for speed
    retriever = build_retriever(json_path)
    retriever_tool = make_retriever_tool(retriever)
    tools = [retriever_tool]

    # Initialize Groq Chat Model
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=groq_api_key or os.getenv("GROQ_API_KEY")
    )

    # System Message with DIRECT DATA INJECTION
    system_message = f"""
You are HeartbeatAnalysisAgent, an expert assistant for analyzing heartbeat recordings.
The following is the analysis result for the current recording:

--- REPORT START ---
{report_summary}
--- REPORT END ---

Your role:
- Answer user questions using the data provided above.
- If the data above doesn't have the answer, you can use your 'heart_retriever_tool' to double-check.
- If asked about medical interpretation, provide general information only, not medical advice.
- Do not make up or hallucinate answers.
- Keep explanations clear and concise.
"""

    # Create the agent using LangGraph
    # By putting data in the system message, the agent can often answer 
    # in 1 turn instead of 2-3 (think -> use tool -> final response).
    agent = create_react_agent(llm, tools, prompt=system_message)

    return agent

