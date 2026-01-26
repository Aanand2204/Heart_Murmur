import streamlit as st
import tempfile
import os
import librosa
from heart_murmur_analysis import (
    HeartbeatAnalyzer, 
    HeartSoundClassifier, 
    load_model,
    build_heartbeat_agent,
    export_json
)
CLASS_MAP = {0: "Artifact", 1: "Murmur", 2: "Normal"}
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            # 1. Run Analysis & Classification
            analyzer = HeartbeatAnalyzer(tmp_path)
            results = analyzer.analyze()
            
            y, sr = librosa.load(tmp_path)
            model = load_model()
            classifier = HeartSoundClassifier(model)
            pred_class, _ = classifier.predict(y, sr)
            results["classification"] = CLASS_MAP.get(pred_class, "Unknown")
            # 2. Save Results to Temp JSON (required by the Agent)
            report_fd, report_path = tempfile.mkstemp(suffix=".json")
            os.close(report_fd)
            export_json(results, report_path)
            # 3. Create Dashboard Tabs
            tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "💬 Chat with AI", "📄 Full Report"])
            with tab1:
                st.success(f"Diagnosis: {results['classification']}")
                analyzer.plot_all()
            with tab2:
                st.subheader("Interactive Analysis Chat")
                
                # Initialize Agent
                if "agent" not in st.session_state:
                    st.session_state.agent = build_heartbeat_agent(
                        report_path,
                        groq_api_key=os.getenv("GROQ_API_KEY"),
                        hf_token=os.getenv("HF_TOKEN")
                    )
                
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
                    response = st.session_state.agent.invoke({"messages": [("user", prompt)]})
                    reply = response["messages"][-1].content
                    
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with st.chat_message("assistant"):
                        st.write(reply)
            with tab3:
                st.subheader("Final Medical Report")
                # Use the dynamic patient_info from the sidebar
                from heart_murmur_analysis import generate_hospital_report
                generate_hospital_report(report_path, patient_info)
        finally:
            os.remove(tmp_path)
            if 'report_path' in locals():
                os.remove(report_path)
if __name__ == "__main__":
    main()