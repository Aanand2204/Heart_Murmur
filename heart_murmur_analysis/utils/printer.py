# utils/printer.py
import json
from typing import Dict, Any


def pretty_print_analysis(result: Dict[str, Any]):
    """
    Nicely formatted console output of the analysis.
    """
    print("=" * 60)
    print(f" File: {result['file']}")
    print(f" Duration: {result['duration_s']:.2f} s")
    print(f" Beats Detected: {result['beats_detected']}")
    print(f" Estimated BPM: {result.get('bpm', 'N/A'):.2f}" if result.get("bpm") else " Estimated BPM: N/A")
    print("=" * 60)

    # --- HRV ---
    print("\n[Heart Rate Variability (HRV)]")
    for k, v in result["hrv"].items():
        if v is not None:
            print(f"  {k:20s} : {v:.2f}")

    # --- Signal Quality ---
    print("\n[Signal Quality]")
    print(f"  SNR (dB)            : {result.get('SNR_dB', 'N/A'):.2f}" if result.get("SNR_dB") else "  SNR (dB)            : N/A")
    for k, v in result["energy"].items():
        print(f"  {k:20s} : {v:.2f}")
    if result["S1S2"]:
        for k, v in result["S1S2"].items():
            print(f"  {k:20s} : {v:.2f}")

    # --- Abnormality Detection ---
    print("\n[Abnormality Checks]")
    print(f"  Extra Peaks per Cycle: {result['extra_peaks']}")
    print(f"  Irregular Spacing     : {result['irregular_spacing']}")
    print(f"  150-500Hz Band Energy : {result['150_500Hz_band']}")

    print("=" * 60)


def export_json(result: Dict[str, Any], path: str):
    """
    Save the analysis result to JSON file.
    """
    safe_result = {k: v for k, v in result.items() if k != "_data"}  # exclude raw arrays
    with open(path, "w") as f:
        json.dump(safe_result, f, indent=4)
    print(f"[INFO] Exported results to {path}")
