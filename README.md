# 🎵 Heart Murmur Detection with LSTM

A deep learning application that uses LSTM neural networks to detect heart murmurs from audio recordings. The application provides a user-friendly Streamlit interface for uploading audio files and getting real-time predictions with signal processing.



## 📁 Project Structure

```
heart_app/
│
├── main.py                         # Streamlit entry point
│
├── classification/
│   ├── __init__.py
│   ├── model_loader.py             # Loads the trained LSTM model
│   ├── classifier.py               # Runs prediction + preprocessing
│
├── signal_processing/
│   ├── __init__.py
│   ├── loader.py                   # Load & normalize wav
│   ├── preprocessing.py            # Bandpass filter, envelope extraction
│   ├── features.py                 # Peak detection, HRV, SNR, Energy, etc.
│   ├── visualizer.py               # Plot waveform, spectrogram, histograms
│   ├── analyzer.py                 # Combines all steps, generates report
│
├── utils/
│   ├── __init__.py
│   ├── printer.py                  # Pretty print reports
│
├── models/
│   └── lstm_model.h5               # Pre-trained LSTM model
│
├── reports/
│   ├── heartbeat_report.json       # Generated reports (saved outputs)
│
├── requirements.txt
└── README.md

```

##  Quick Start

### Prerequisites

- **Python 3.8 or higher**
- **Windows 10/11** (PowerShell)
- **Git** (optional, for cloning)

### Installation Steps

1. **Clone or Download the Project**
   ```bash
   git clone <repository-url>
   cd Heart-Murmur-Disease
   ```
   Or download and extract the project folder.

2. **Create Virtual Environment**
   ```powershell
   python -m venv hvenv
   ```

3. **Activate Virtual Environment**
   ```powershell
   hvenv\Scripts\Activate.ps1
   ```
   You should see `(hvenv)` at the beginning of your command prompt.

4. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Run the Application**
   ```powershell
   streamlit run app.py
   ```

6. **Access the Application**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - Upload a WAV or MP3 audio file
   - Get instant heart murmur predictions with signal processing !


###  Install Dependencies

The `requirements.txt` file contains all necessary packages:



Install them with:
```cmd
pip install -r requirements.txt
```



### Step 4: Run the Application

```cmd
streamlit run app.py
```


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





