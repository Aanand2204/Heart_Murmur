import streamlit as st
import librosa
import librosa.display
import os
import tempfile
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import asyncio
import asyncio
from heart_murmur_analysis.report_generator import generate_hospital_report

from heart_murmur_analysis.classification import load_model, HeartSoundClassifier
from heart_murmur_analysis.signal_processing import HeartbeatAnalyzer
from heart_murmur_analysis.utils import pretty_print_analysis, export_json

# Agent imports
from heart_murmur_analysis.agent import build_heartbeat_agent

# --- Suppress TF / deprecation warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# --- CLASS LABELS ---
CLASS_MAP = {
    0: "Artifact",
    1: "Murmur",
    2: "Normal"
}

REPORT_PATH = "reports/heartbeat_report.json"


@st.cache_data
def load_audio(uploaded_file):
    """Cached audio loading."""
    y, sr = librosa.load(uploaded_file, sr=22050)
    return y, sr


@st.cache_data
def get_classification_results(y, sr):
    """Cached classification logic (logic only, no UI)."""
    model = load_model()
    classifier = HeartSoundClassifier(model)
    pred_class, scores = classifier.predict(y, sr)
    class_name = CLASS_MAP.get(pred_class, "Unknown")
    return class_name, pred_class, scores


def run_classification(y, sr, results_dict):
    st.subheader("🔎 Classification (Deep Learning)")
    
    # Classification is now mostly handled in run_analysis or cached
    class_name, pred_class, scores = get_classification_results(y, sr)

    st.write(f"**Predicted Class:** {class_name} ({pred_class})")
    st.write("**Raw Scores:**", scores.tolist())

    # Add classification result into results dict
    results_dict["classification"] = class_name

    st.subheader("Waveform of Input Sound")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    return results_dict





def run_signal_processing(uploaded_file, results_dict):
    st.subheader("📊 Signal Processing Analysis")

    # Use file bytes for caching key
    file_bytes = uploaded_file.getvalue()
    sp_results_data = get_signal_processing_results_with_data(file_bytes)
    
    # Merge classification + signal processing results (minus the raw data for storage)
    serializable_results = {k: v for k, v in sp_results_data.items() if k != "_data"}
    results_dict.update(serializable_results)

    # Pretty print
    pretty_print_analysis(results_dict)

    # Show results in Streamlit
    st.json(serializable_results)

    # Export results
    os.makedirs("reports", exist_ok=True)
    export_json(results_dict, REPORT_PATH)

    # Use INJECTED results for plotting - MUCH FASTER
    # No need to re-run analysis
    from heart_murmur_analysis.signal_processing import HeartbeatAnalyzer
    analyzer = HeartbeatAnalyzer(None) # Path not needed if we inject results
    analyzer._last_results = sp_results_data
    analyzer.plot_all()

@st.cache_data
def get_signal_processing_results_with_data(file_bytes):
    """Cached signal processing logic INCLUDING raw data for plotting."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        temp_wav_path = tmp.name

    try:
        analyzer = HeartbeatAnalyzer(temp_wav_path)
        sp_results = analyzer.analyze()
        return sp_results
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)


# ----------------------------
# Agent Chat UI
# ----------------------------
def run_agent_chat():
    if not os.path.exists(REPORT_PATH):
        st.warning("⚠️ Please run Signal Processing first to generate a report.")
        return

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Build agent lazily
    if "agent" not in st.session_state:
        st.session_state.agent = build_heartbeat_agent(REPORT_PATH)

    # Chat input
    user_input = st.chat_input("Ask me about your heartbeat analysis...")
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        
        # Display existing history
        for role, msg in st.session_state.chat_history[:-1]:
            with st.chat_message(role):
                st.write(msg)
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("bot"):
            response_container = st.empty()
            full_reply = ""
            
            with st.spinner("Agent is thinking..."):
                try:
                    for chunk in st.session_state.agent.stream(
                        {"messages": [("user", user_input)]}, 
                        stream_mode="messages"
                    ):
                        msg, metadata = chunk
                        if msg.content and metadata.get("langgraph_node") == "agent":
                            full_reply += msg.content
                            response_container.markdown(full_reply + "▌")
                    
                    response_container.markdown(full_reply)
                    st.session_state.chat_history.append(("bot", full_reply))

                except Exception as e:
                    response_container.write(f"Error processing request: {str(e)}")
                    st.session_state.chat_history.append(("bot", f"Error: {str(e)}"))
    else:
        # Display chat history if no new input
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)


# ----------------------------
# Streamlit Main
# ----------------------------
def main():
    st.title("❤️ Heartbeat Analysis App")
    
    # --- Sidebar for Patient Details ---
    st.sidebar.header("👤 Patient Information")
    p_name = st.sidebar.text_input("Full Name", "John Doe")
    p_age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=52)
    p_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"], index=0)
    
    patient_info = {
        "name": p_name,
        "age": p_age,
        "gender": p_gender
    }

    st.write("Upload a heartbeat `.wav` file to analyze using Deep Learning, Signal Processing, and chat with the AI agent.")

    uploaded_file = st.file_uploader("Upload heartbeat audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        y, sr = load_audio(uploaded_file)

        results_dict = {}

        tab1, tab2, tab3 , tab4 = st.tabs(["Classification", "Signal Processing", "💬 Chat with Agent" , "Final Report"])

        with tab1:
            results_dict = run_classification(y, sr, results_dict)

        with tab2:
            run_signal_processing(uploaded_file, results_dict)

        with tab3:
            run_agent_chat()
        
        with tab4:
            st.header("📄 Heart Sound Report")
            
            # Call the report generator function
            generate_hospital_report("reports/heartbeat_report.json", patient_info)


if __name__ == "__main__":
    main()
