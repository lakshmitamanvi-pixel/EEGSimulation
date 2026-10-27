"""
config.py - Frozen constants and configuration for eeg-event-tx.
FROZEN CONSTANTS (never change, tune, or "fix").
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"

# PhysioNet Source
BASE_URL = "https://physionet.org/files/chbmit/1.0.0/chb01/"
EDF_FILE = "chb01_03.edf"
SUMMARY_FILE = "chb01-summary.txt"
PATIENT = "chb01"
CHANNEL_INDEX = 0

# Signal Window (file time in seconds)
WINDOW_START_S = 2700
WINDOW_END_S = 3300
WINDOW_DURATION_S = 600

# Slices relative to window (in seconds)
# Seizure slice: window seconds 296-336 (= file time 2996-3036 s)
SEIZURE_WINDOW_START_S = 296
SEIZURE_WINDOW_END_S = 336
SEIZURE_FILE_START_S = 2996
SEIZURE_FILE_END_S = 3036

# Background slice: window seconds 0-250
BACKGROUND_WINDOW_START_S = 0
BACKGROUND_WINDOW_END_S = 250

# Sweep values for c (fixed) and k (adaptive)
SWEEP_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]

# Frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Spectral Analysis
WELCH_NPERSEG = 512

# Adaptive Threshold Floor Factor
ADAPTIVE_MIN_FACTOR = 0.02
