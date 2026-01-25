# signal_processing/loader.py
import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as signal
from typing import Tuple


def load_wav(path: str, target_fs: int = None) -> Tuple[int, np.ndarray]:
    """
    Load a WAV file, convert to float32 [-1,1], mono, and resample if needed.
    """
    rate, data = wav.read(path)

    # Convert to float32 in [-1,1]
    if data.dtype.kind == "i":
        maxv = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / maxv
    else:
        data = data.astype(np.float32)

    # Stereo -> mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if target_fs is not None and target_fs != rate:
        n_samples = round(len(data) * float(target_fs) / rate)
        data = signal.resample(data, n_samples)
        rate = target_fs

    return rate, data
