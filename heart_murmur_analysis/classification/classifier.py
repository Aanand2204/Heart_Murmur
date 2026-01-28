# classification/classifier.py
import numpy as np



class HeartSoundClassifier:
    def __init__(self, model):
        """
        Initialize classifier with a pre-trained model.
        """
        self.model = model

    def extract_features(self, y, sr, n_mfcc=40):
        """
        Extract MFCC features from the audio.
        """
        import librosa
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled

    def prepare_input(self, mfcc_scaled):
        """
        Reshape features for LSTM input.
        Expected shape: (samples, timesteps, features)
        """
        X_input = np.expand_dims(mfcc_scaled, axis=0)  # Add batch dimension
        X_input = np.expand_dims(X_input, axis=2)      # Add feature dimension
        return X_input

    def predict(self, y, sr):
        """
        Full pipeline: extract features, prepare input, and predict class.
        Returns predicted_class, raw_scores
        """
        features = self.extract_features(y, sr)
        X_input = self.prepare_input(features)
        # Use direct call for faster inference than model.predict
        prediction = self.model(X_input, training=False).numpy()
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        return predicted_class, prediction
