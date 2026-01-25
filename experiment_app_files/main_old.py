import streamlit as st
import librosa
import os
import tempfile
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt

from classification import load_model, HeartSoundClassifier
from signal_processing import HeartbeatAnalyzer
from utils import pretty_print_analysis, export_json



# --- Suppress TF / deprecation warnings for cleaner logs ---
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF info/warnings
tf.get_logger().setLevel("ERROR")


# --- CLASS LABELS ---
CLASS_MAP = {
    0: "Artifact",
    1: "Murmur",
    2: "Normal"
}


def run_classification(y, sr):
    """
    Run LSTM classification on the audio.
    """
    st.subheader("🔎 Classification (Deep Learning)")
    model = load_model()
    classifier = HeartSoundClassifier(model)

    pred_class, scores = classifier.predict(y, sr)
    class_name = CLASS_MAP.get(pred_class, "Unknown")

    st.write(f"**Predicted Class:** {class_name} ({pred_class})")
    st.write("**Raw Scores:**", scores.tolist())
    
    st.subheader("Waveform of Input Sound")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)



def run_signal_processing(uploaded_file):
    """
    Run full signal processing analysis.
    """
    st.subheader("📊 Signal Processing Analysis")

    # Save uploaded file to a temporary .wav path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_wav_path = tmp.name

    analyzer = HeartbeatAnalyzer(temp_wav_path)
    results = analyzer.analyze()

    # Pretty print in console
    pretty_print_analysis(results)

    # Show results in Streamlit
    st.json({k: v for k, v in results.items() if k != "_data"})

    # Export results
    os.makedirs("reports", exist_ok=True)
    export_json(results, "reports/heartbeat_report.json")

    # Show plots
    analyzer.plot_all()

    # Cleanup (optional: remove temp file)
    os.remove(temp_wav_path)


def main():
    st.title("❤️ Heartbeat Analysis App")
    st.write("Upload a heartbeat `.wav` file to analyze using Deep Learning and Signal Processing.")

    uploaded_file = st.file_uploader("Upload heartbeat audio (.wav)", type=["wav"])

    if uploaded_file is not None:
        # Load audio with librosa for classification
        y, sr = librosa.load(uploaded_file, sr=22050)

        # Tabs: Classification vs Signal Processing
        tab1, tab2 = st.tabs(["Classification", "Signal Processing"])

        with tab1:
            run_classification(y, sr)

        with tab2:
            run_signal_processing(uploaded_file)

        

import asyncio
from autogen_agentchat.ui import Console
from agents.heartbeat_agent import build_heartbeat_agent

async def main():
    # Build agent with retriever
    agent = build_heartbeat_agent("reports/heartbeat_report.json")

    # Run interactive console
    print("🔊 Heartbeat Analysis Agent is ready. Ask your questions!\n")
    await Console(agent.run_stream(task="Hello!"))


if __name__ == "__main__":
    asyncio.run(main())



if __name__ == "__main__":
    main()
