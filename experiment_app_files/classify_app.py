import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.layers import LSTM as OriginalLSTM

# Custom LSTM to handle 'time_major' argument
class CustomLSTM(OriginalLSTM):
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs:
            del kwargs['time_major']
        super().__init__(*args, **kwargs)

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lstm_model.h5", custom_objects={'LSTM': CustomLSTM})  # your saved LSTM model
    return model

model = load_model()

st.title("🎵 Heart Murmur Detection with LSTM")

# File uploader
uploaded_file = st.file_uploader("Upload a heart sound (WAV/MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=22050)

    # Show waveform
    st.subheader("Waveform of Input Sound")
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # Feature extraction (example: MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # Reshape for LSTM (samples, timesteps, features)
    X_input = np.expand_dims(mfcc_scaled, axis=0)
    X_input = np.expand_dims(X_input, axis=2)

    # Prediction
    prediction = model.predict(X_input)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.subheader("🔮 Prediction Result")
    st.write(f"Predicted Class: **{predicted_class}**")
    st.write("Raw Prediction Scores:", prediction)
