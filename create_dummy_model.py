import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

def create_dummy_model():
    # Input shape based on classifier.py: (40, 1)
    # 40 MFCC features, 1 channel (or 40 timesteps, 1 feature)
    input_shape = (40, 1)
    
    model = Sequential([
        Input(shape=input_shape),
        # Use standard LSTM. model_loader.py's CustomLSTM just handles loading compatibility
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax') # 3 classes: Artifact, Murmur, Normal
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Model summary:")
    model.summary()
    
    save_path = "models/lstm_model.h5"
    model.save(save_path)
    print(f"Dummy model saved to {save_path}")

if __name__ == "__main__":
    create_dummy_model()
