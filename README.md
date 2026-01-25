

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
   hvenv\Scripts\Activate.ps1
   ```
   You should see `(hvenv)` at the beginning of your command prompt.

3. **Install the package**
```bash
pip install heart-murmur-analysis
```

4. **Run the Application**
   Create a file with the code
   ```powershell
   streamlit run file_name.py
   ```

5. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - Upload a WAV or MP3 audio file
   - Get instant heart murmur predictions with signal processing !


##  About Model Architecture

The LSTM model uses a hybrid CNN-LSTM architecture:

- **Input**: Raw audio data (52 timesteps, 1 feature)
- **CNN Layers**: 3 Conv1D layers with MaxPooling and BatchNormalization
- **LSTM Layers**: 2 LSTM layers for sequence modeling
- **Dense Layers**: 3 fully connected layers with dropout
- **Output**: 3 classes (Normal, Abnormal, Murmur)
- **Total Parameters**: 14,130,371 (53.90 MB)


### Performance Tips

- **Sample Rate**: The model expects 22050 Hz (automatically handled)


## Technical Details

### Input Preprocessing
- Audio is loaded at 22050 Hz sample rate
- Truncated or padded to exactly 52 samples
- Reshaped to (1, 52, 1) for model input



##  Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request


**Happy Heart Murmur Detection!**





