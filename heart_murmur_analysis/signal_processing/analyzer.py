# signal_processing/analyzer.py
import numpy as np
from typing import Dict, Any
from .loader import load_wav
from .preprocessing import bandpass_filter, envelope
from .features import (
    detect_peaks_from_envelope,
    compute_intervals, compute_hrv_metrics,
    estimate_snr, energy_distribution, s1_s2_amplitude_ratio,
    detect_extra_peaks_per_cycle, irregular_spacing_stats, frequency_band_energy
)
from .visualizer import plot_results


class HeartbeatAnalyzer:
    def __init__(self, wav_path: str, resample_fs: int = 2000):
        self.wav_path = wav_path
        self.resample_fs = resample_fs
        self._last_results = None  # cache last analysis

    def analyze(self) -> Dict[str, Any]:
        fs, raw = load_wav(self.wav_path, target_fs=self.resample_fs)
        raw = raw - np.mean(raw)  # remove DC
        filtered = bandpass_filter(raw, fs, lowcut=20.0, highcut=500.0, order=4)
        env = envelope(filtered, fs, win_ms=40.0)
        peaks, _ = detect_peaks_from_envelope(env, fs)
        intervals_s = compute_intervals(peaks, fs)

        hrv = compute_hrv_metrics(intervals_s)
        snr_db = estimate_snr(filtered, peaks, fs)
        energy = energy_distribution(filtered, fs)
        s1s2 = s1_s2_amplitude_ratio(env, peaks)
        extra_peaks = detect_extra_peaks_per_cycle(peaks, fs, intervals_s)
        irregular = irregular_spacing_stats(intervals_s)
        band = frequency_band_energy(filtered, fs)

        results = {
            "file": self.wav_path,
            "fs": fs,
            "duration_s": len(raw) / fs,
            "beats_detected": len(peaks),
            "bpm": hrv.get("bpm_mean"),
            "hrv": hrv,
            "SNR_dB": snr_db,
            "energy": energy,
            "S1S2": s1s2,
            "extra_peaks": extra_peaks,
            "irregular_spacing": irregular,
            "150_500Hz_band": band,
            "_data": {
                "time": np.arange(len(raw)) / fs,
                "raw": raw,
                "filtered": filtered,
                "env": env,
                "peaks": peaks,
                "intervals_s": intervals_s
            }
        }

        self._last_results = results
        return results

    def plot_all(self):
        """Wrapper: forward cached results to visualizer.plot_results()."""
        if self._last_results is None:
            raise RuntimeError("Run analyze() before calling plot_all().")

        data = self._last_results["_data"]
        plot_results(
            time=data["time"],
            raw=data["raw"],
            filtered=data["filtered"],
            env=data["env"],
            fs=self._last_results["fs"],
            peaks=data["peaks"],
            intervals_s=data["intervals_s"]
        )
