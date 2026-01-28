import os
import hashlib
import json
import streamlit as st
import tempfile
import time
from heart_murmur_analysis import (
    HeartbeatAnalyzer, 
    HeartSoundClassifier, 
    load_model,
    build_heartbeat_agent,
    export_json,
    generate_hospital_report
)

CLASS_MAP = {0: "Artifact", 1: "Murmur", 2: "Normal"}

@st.cache_data
def get_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

@st.cache_data
def run_heavy_analysis(file_hash, file_bytes):
    """Cached analysis and classification results."""
    # Create a temporary file inside the cached function
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    
    try:
        start_time = time.time()
        # 1. Run Analysis (Returns fs, raw, and results)
        analyzer = HeartbeatAnalyzer(tmp_path)
        results = analyzer.analyze()
        print(f"DEBUG: Signal analysis took {time.time() - start_time:.2f}s")
        
        # 2. Classification (Avoid reloading file with librosa)
        start_time = time.time()
        raw_data = results["_data"]["raw"]
        fs = results["fs"]
        
        model = cached_load_model()
        classifier = HeartSoundClassifier(model)
        pred_class, _ = classifier.predict(raw_data, fs)
        results["classification"] = CLASS_MAP.get(pred_class, "Unknown")
        print(f"DEBUG: Classification took {time.time() - start_time:.2f}s")
        
        return results
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@st.cache_resource
def cached_load_model():
    return load_model()

@st.cache_resource
def cached_build_agent(report_content_hash, results_dict, groq_key, hf_token):
    """
    Build agent based on content hash to avoid cache miss from temp paths.
    """
    # Create a stable temp file for this content hash
    report_path = os.path.join(tempfile.gettempdir(), f"report_{report_content_hash}.json")
    export_json(results_dict, report_path)
    
    return build_heartbeat_agent(
        report_path,
        groq_api_key=groq_key,
        hf_token=hf_token
    )
def main():
    st.set_page_config(page_title="AI Cardiologist", layout="wide")
    st.title("🩺 AI Cardiologist Assistant")
    # --- Sidebar for Patient Info ---
    st.sidebar.header("👤 Patient Details")
    p_name = st.sidebar.text_input("Patient Name", "Anonymous")
    p_age = st.sidebar.number_input("Age", 0, 120, 30)
    p_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
    patient_info = {"name": p_name, "age": p_age, "gender": p_gender}
    uploaded_file = st.file_uploader("Upload Heartbeat (.wav)", type=["wav"])
    if uploaded_file:
        start_total = time.time()
        file_bytes = uploaded_file.getvalue()
        file_hash = get_file_hash(file_bytes)
        print(f"DEBUG: File hashing took {time.time() - start_total:.2f}s")
        
        try:
            # 1. Run Analysis & Classification (Cached by file content)
            with st.spinner("Analyzing Heartbeat..."):
                results = run_heavy_analysis(file_hash, file_bytes)
            
            serializable_results = {k: v for k, v in results.items() if k != "_data"}
            
            # 2. Dashboard Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "💬 Chat with AI", "📄 Full Report"])
            with tab1:
                st.success(f"Diagnosis: {results['classification']}")
                
                # Re-run ONLY the plotting part, using cached results data
                # No need for the wav file on disk here because we inject results
                start_plot = time.time()
                analyzer = HeartbeatAnalyzer(None)
                analyzer._last_results = results # Inject results to avoid re-analysis
                analyzer.plot_all()
                print(f"DEBUG: Plotting took {time.time() - start_plot:.2f}s")
                print(f"DEBUG: TOTAL UI update took {time.time() - start_total:.2f}s")
            
            with tab2:
                st.subheader("Interactive Analysis Chat")
                
                # Initialize Agent (Cached Resource by content hash)
                report_content_hash = hashlib.md5(json.dumps(serializable_results).encode()).hexdigest()
                
                if "agent" not in st.session_state or st.session_state.get("last_report_hash") != report_content_hash:
                    with st.spinner("Preparing AI Agent..."):
                        start_agent = time.time()
                        st.session_state.agent = cached_build_agent(
                            report_content_hash,
                            serializable_results,
                            os.getenv("GROQ_API_KEY"),
                            os.getenv("HF_TOKEN")
                        )
                        print(f"DEBUG: Agent preparation took {time.time() - start_agent:.2f}s")
                        st.session_state.last_report_hash = report_content_hash
                        st.session_state.messages = [] # Clear history on new data
                
                # Simple Chat Interface
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])
                if prompt := st.chat_input("Ask about the analysis..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.write(prompt)
                    
                    with st.chat_message("assistant"):
                        # Use container to display streamed content
                        response_container = st.empty()
                        full_reply = ""
                        
                        # Use stream method for better UI feel
                        with st.spinner("Agent is thinking..."):
                            # Filter out system messages and format for LangGraph if needed
                            # but direct invoke is also fine. Let's use stream for token-by-token.
                            for chunk in st.session_state.agent.stream(
                                {"messages": [("user", prompt)]}, 
                                stream_mode="messages"
                            ):
                                # Check if it's an AIMessage chunk
                                msg, metadata = chunk
                                if msg.content and metadata.get("langgraph_node") == "agent":
                                    full_reply += msg.content
                                    response_container.markdown(full_reply + "▌")
                        
                        response_container.markdown(full_reply)
                        st.session_state.messages.append({"role": "assistant", "content": full_reply})
            with tab3:
                st.subheader("Final Medical Report")
                # Create a persistent report file for the generator
                report_path = os.path.join(tempfile.gettempdir(), f"report_{report_content_hash}.json")
                if not os.path.exists(report_path):
                    export_json(serializable_results, report_path)
                
                generate_hospital_report(report_path, patient_info)
        except Exception as e:
            st.error(f"Error during analysis: {e}")
if __name__ == "__main__":
    main()