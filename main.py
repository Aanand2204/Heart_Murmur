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
from report_generator.report_generator import generate_hospital_report

from classification import load_model, HeartSoundClassifier
from signal_processing import HeartbeatAnalyzer
from utils import pretty_print_analysis, export_json

# Agent imports
from agent.heartbeat_agent import build_heartbeat_agent

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


# ----------------------------
# Deep Learning Classification
# ----------------------------
def run_classification(y, sr, results_dict):
    st.subheader("🔎 Classification (Deep Learning)")
    model = load_model()
    classifier = HeartSoundClassifier(model)

    pred_class, scores = classifier.predict(y, sr)
    class_name = CLASS_MAP.get(pred_class, "Unknown")

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


# ----------------------------
# Signal Processing Analysis
# ----------------------------
def run_signal_processing(uploaded_file, results_dict):
    st.subheader("📊 Signal Processing Analysis")

    # Save uploaded file to temp path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_wav_path = tmp.name

    analyzer = HeartbeatAnalyzer(temp_wav_path)
    sp_results = analyzer.analyze()

    # Merge classification + signal processing results
    results_dict.update(sp_results)

    # Pretty print
    pretty_print_analysis(results_dict)

    # Show results in Streamlit
    st.json({k: v for k, v in results_dict.items() if k != "_data"})

    # Export results (classification + signal processing)
    os.makedirs("reports", exist_ok=True)
    export_json(results_dict, REPORT_PATH)

    # Show plots
    analyzer.plot_all()

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

        with st.spinner("Agent is thinking..."):
            try:
                response = st.session_state.agent.invoke({"messages": [("user", user_input)]})
                # LangGraph returns 'messages' list, last one is the AI response
                reply = response["messages"][-1].content

            except Exception as e:
                reply = f"Error processing request: {str(e)}"

        st.session_state.chat_history.append(("bot", reply))

    # Display chat
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)


# ----------------------------
# Streamlit Main
# ----------------------------
def main():
    st.title("❤️ Heartbeat Analysis App")
    st.write("Upload a heartbeat `.wav` file to analyze using Deep Learning, Signal Processing, and chat with the AI agent.")

    uploaded_file = st.file_uploader("Upload heartbeat audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        y, sr = librosa.load(uploaded_file, sr=22050)

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
    patient_info = {
        "name": "John Doe",
        "age": 52,
        "gender": "Male"
    }
    main()
