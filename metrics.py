"""
metrics.py - Evaluation metrics for eeg-event-tx.
Computes events/s, MSE, relative band-power error per band = |P(xh) - P(x)| / P(x) from Welch, and their mean.
Computes separately on seizure and background slices.
"""
from typing import Dict, Any, Tuple
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid

from config import (
    BANDS,
    WELCH_NPERSEG,
    BACKGROUND_WINDOW_START_S,
    BACKGROUND_WINDOW_END_S,
    SEIZURE_WINDOW_START_S,
    SEIZURE_WINDOW_END_S,
)


def compute_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    f_low: float,
    f_high: float,
) -> float:
    """Compute band power by integrating PSD between f_low and f_high."""
    idx = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(idx):
        return 0.0
    return float(trapezoid(psd[idx], freqs[idx]))


def compute_slice_metrics(
    x_slice: np.ndarray,
    xh_slice: np.ndarray,
    events_slice: np.ndarray,
    fs: float,
    nperseg: int = WELCH_NPERSEG,
) -> Dict[str, Any]:
    """
    Compute metrics for a specific slice:
    - events/s
    - MSE
    - relative band-power error per band = |P(xh) - P(x)| / P(x)
    - mean band-power error
    """
    duration_s = len(x_slice) / fs
    num_events = int(np.sum(events_slice))
    events_per_sec = num_events / duration_s if duration_s > 0 else 0.0

    mse = float(np.mean((x_slice - xh_slice) ** 2))

    # Welch PSD computation
    # If the slice is shorter than nperseg, adjust nperseg for that slice to len(x_slice)
    actual_nperseg = min(nperseg, len(x_slice))
    freqs_x, psd_x = welch(x_slice, fs=fs, nperseg=actual_nperseg)
    _, psd_xh = welch(xh_slice, fs=fs, nperseg=actual_nperseg)

    band_powers_x = {}
    band_powers_xh = {}
    rel_band_errors = {}

    for band_name, (f_low, f_high) in BANDS.items():
        px = compute_band_power(freqs_x, psd_x, f_low, f_high)
        pxh = compute_band_power(freqs_x, psd_xh, f_low, f_high)
        band_powers_x[band_name] = px
        band_powers_xh[band_name] = pxh
        rel_err = abs(pxh - px) / px if px > 0 else 0.0
        rel_band_errors[band_name] = rel_err

    mean_band_error = float(np.mean(list(rel_band_errors.values())))

    return {
        "duration_s": duration_s,
        "num_events": num_events,
        "events_per_sec": events_per_sec,
        "mse": mse,
        "band_powers_x": band_powers_x,
        "band_powers_xh": band_powers_xh,
        "rel_band_errors": rel_band_errors,
        "mean_band_error": mean_band_error,
        "mean_band_error_pct": mean_band_error * 100.0,
    }


def evaluate_reconstruction(
    x: np.ndarray,
    xh: np.ndarray,
    events: np.ndarray,
    fs: float,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate reconstruction metrics separately on:
    - background slice (window seconds 0-250)
    - seizure slice (window seconds 296-336)
    - overall window (0-600 s)
    """
    bg_start = int(round(BACKGROUND_WINDOW_START_S * fs))
    bg_end = int(round(BACKGROUND_WINDOW_END_S * fs))

    sz_start = int(round(SEIZURE_WINDOW_START_S * fs))
    sz_end = int(round(SEIZURE_WINDOW_END_S * fs))

    bg_metrics = compute_slice_metrics(
        x[bg_start:bg_end],
        xh[bg_start:bg_end],
        events[bg_start:bg_end],
        fs=fs,
    )

    sz_metrics = compute_slice_metrics(
        x[sz_start:sz_end],
        xh[sz_start:sz_end],
        events[sz_start:sz_end],
        fs=fs,
    )

    overall_metrics = compute_slice_metrics(
        x,
        xh,
        events,
        fs=fs,
    )

    return {
        "background": bg_metrics,
        "seizure": sz_metrics,
        "overall": overall_metrics,
    }
