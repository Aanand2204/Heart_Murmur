# Heart Sound Analysis App

## Concepts & Terminologies

---

### 1. Heart Sounds

- **S1 ("Lub")**: Closing of the mitral and tricuspid valves.  
- **S2 ("Dub")**: Closing of the aortic and pulmonary valves.  
- **Murmur**: Abnormal sound caused by turbulent blood flow.  
- **Artifact**: Noise unrelated to actual heart sounds (e.g., movement, microphone noise).

---

### 2. Deep Learning Classification

- **Feature Extraction**: Uses **MFCCs (Mel-Frequency Cepstral Coefficients)** to represent heart sound audio.  
- **Model**: LSTM (**Long Short-Term Memory**) network.  
- **Output Classes**:
  - `0` = Artifact  
  - `1` = Murmur  
  - `2` = Normal  

**Why LSTM?**  
Heart sounds are **time-series signals**. LSTMs capture temporal dependencies, such as repeated "lub-dub" patterns, which are critical for accurate classification.

---

### 3. Signal Processing Pipeline

#### 3.1 Preprocessing
- **Bandpass Filter (20–500 Hz)**: Retains important heart sound frequencies while removing low-frequency drift and high-frequency noise.  
- **Envelope Detection**: Smooths the waveform to highlight heartbeats.

#### 3.2 Peak Detection
- Detects **S1/S2 peaks** from the envelope.  
- Used to compute:
  - Heart rate (BPM)  
  - Inter-beat intervals (IBIs)

#### 3.3 Metrics Extracted

| Metric | Description |
|--------|-------------|
| **BPM** | Average heart rate in beats per minute. |
| **IBI** | Inter-Beat Interval – time between successive heartbeats. |
| **HRV (Heart Rate Variability)** | Measures beat-to-beat variation. |
| SDNN | Standard deviation of IBIs → overall variability. |
| RMSSD | Short-term variability of IBIs. |
| pNN50 | Percentage of successive IBIs differing by >50ms → reflects autonomic balance. |
| Poincaré Plot (SD1/SD2) | Nonlinear HRV visualization. |
| **SNR (Signal-to-Noise Ratio)** | Quality of recording. |
| Energy Distribution | % energy below 200 Hz (dominant for heart sounds). |
| Spectral Centroid | Average "center" of signal frequency energy. |
| S1/S2 Amplitude Ratio | Helps distinguish murmurs from normal beats. |
| Extra Peaks | May indicate arrhythmias or noise. |
| Irregular Spacing | Detects abnormal rhythm patterns. |
| 150–500 Hz Band Energy | Captures murmurs that occur in higher frequency bands. |

---

### 4. Visualizations

- **Waveform with Peaks**: Shows detected beats in the time domain.  
- **Spectrogram**: Frequency vs. time representation (in dB).  
- **IBI Histogram**: Distribution of inter-beat intervals.  
- **Poincaré Plot**: Nonlinear HRV analysis for visualizing variability.

---
