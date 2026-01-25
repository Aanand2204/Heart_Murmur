# classification/model_loader.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM as OriginalLSTM
import streamlit as st


# Custom LSTM to handle 'time_major' argument in saved model
class CustomLSTM(OriginalLSTM):
    def __init__(self, *args, **kwargs):
        if 'time_major' in kwargs:
            del kwargs['time_major']
        super().__init__(*args, **kwargs)


import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "lstm_model.h5")

@st.cache_resource
def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads the trained LSTM model with custom LSTM layer.
    """
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'LSTM': CustomLSTM}
    )
    return model
