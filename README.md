

# 🎵 Heart Murmur Detection with LSTM

A deep learning application that uses LSTM neural networks to detect heart murmurs from audio recordings. The application provides a user-friendly Streamlit interface for uploading audio files and getting real-time predictions with signal processing.

---
<img width="1001" height="488" alt="Heart_Murmur_Pipelinee drawio" src="https://github.com/user-attachments/assets/317367a4-20ad-410e-b3f3-4e53b8f816d4" />
---


##  Quick Start

### Prerequisites

- **Python 3.9 or higher**
- **Windows 10/11** (PowerShell)
- **Git** (optional, for cloning)

### Installation Steps
1. **Create Virtual Environment**
   ```powershell
   python -m venv hvenv
   ```

2. **Activate Virtual Environment**
   ```powershell
   hvenv\Scripts\activate
   ```
   You should see `(hvenv)` at the beginning of your command prompt.

3. **Install the package**
   ```bash
   pip install heart-murmur-analysis
   ```

4. **Run the Application**
   ```powershell
   streamlit run test_file.py
   ```
   Download the test_file.py 

5. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - Upload a WAV or MP3 audio file
   - Get instant heart murmur predictions with signal processing !


## 🧠 Concepts & Terminologies

### Heart Sounds
- **S1 ("Lub")**: Closing of the mitral and tricuspid valves.
- **S2 ("Dub")**: Closing of the aortic and pulmonary valves.
- **Murmur**: Abnormal sound caused by turbulent blood flow.
- **Artifact**: Noise unrelated to actual heart sounds (e.g., movement, microphone noise).

---

## 🛠️ Signal Processing Pipeline

### 1. Preprocessing
- **Bandpass Filter (20–500 Hz)**: Retains important heart sound frequencies while removing noise.
- **Envelope Detection**: Smooths the waveform to highlight heartbeats for peak detection.

### 2. Extracted Metrics
The application extracts several clinical metrics to assist in diagnosis:

| Metric | Description |
|--------|-------------|
| **BPM** | Average heart rate in beats per minute. |
| **IBV/IBI** | Inter-Beat Interval – time between successive heartbeats. |
| **HRV** | Heart Rate Variability (SDNN, RMSSD, pNN50). |
| **SNR** | Signal-to-Noise Ratio (recording quality). |
| **S1/S2 Ratio** | Helps distinguish murmurs from normal beats. |
| **Spectral Energy** | Energy distribution below 200 Hz. |

---

## 📊 Visualizations
- **Waveform with Peaks**: Visualizes detected beats.
- **Spectrogram**: Frequency vs. time representation.
- **Poincaré Plot**: Nonlinear HRV visualization.

---

## 🤖 Model Architecture
The LSTM model uses a hybrid CNN-LSTM architecture:
- **Input**: Raw audio data (52 timesteps).
- **CNN Layers**: 3 Conv1D layers for feature extraction.
- **LSTM Layers**: 2 LSTM layers for temporal modeling.
- **Output**: 3 classes (**Normal**, **Murmur**, **Artifact**).

---



##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


**Happy Heart Murmur Detection!**





