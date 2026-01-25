import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import streamlit as st  

def plot_results(time, raw, filtered, env, fs, peaks, intervals_s):
    """
    Generate standard plots for heartbeat analysis (Streamlit-friendly).
    """
    # waveform + peaks
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time, raw, label="raw", alpha=0.4)
    ax.plot(time, filtered, label="filtered", linewidth=0.8)
    ax.plot(time, env / np.max(env) * 0.8 * np.max(filtered),
            label="envelope (scaled)", linewidth=1)
    ax.scatter(peaks / fs, env[peaks], c="red", marker="x", label="detected peaks")
    ax.set_xlabel("Time (s)")
    ax.set_title("Waveform with detected peaks")
    ax.legend()
    st.pyplot(fig)   

    # spectrogram
    fig, ax = plt.subplots(figsize=(14, 4))
    f, t, Sxx = signal.spectrogram(filtered, fs=fs, nperseg=1024, noverlap=512)
    pcm = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading='gouraud')
    ax.set_ylim(0, 800)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Spectrogram (dB)")
    fig.colorbar(pcm, ax=ax, label="dB")
    st.pyplot(fig)

    # IBI histogram
    if intervals_s.size > 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(intervals_s * 1000.0, bins=20)
        ax.set_xlabel("IBI (ms)")
        ax.set_title("Inter-beat interval histogram")
        st.pyplot(fig)

    # Poincaré plot
    if intervals_s.size > 1:
        fig, ax = plt.subplots(figsize=(5, 5))
        x, y = intervals_s[:-1] * 1000.0, intervals_s[1:] * 1000.0
        ax.scatter(x, y, s=10)
        ax.set_xlabel("IBI_n (ms)")
        ax.set_ylabel("IBI_{n+1} (ms)")
        ax.set_title("Poincaré Plot")
        st.pyplot(fig)
