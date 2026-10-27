"""
encoders.py - Event-triggered EEG transmission encoders (Fixed vs Adaptive threshold).
"""
import math
from typing import Dict, Any, Optional
import numpy as np
from config import ADAPTIVE_MIN_FACTOR


def encode_fixed(
    x: np.ndarray,
    fs: float,
    c: Optional[float] = None,
    delta: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fixed threshold send-on-delta encoder.
    delta = c * std(x) (or specified directly).
    Loop over samples; if |x[n] - last| >= delta: last = x[n], mark event.
    xh[n] = last (zero-order hold).

    Parameters:
        x: Input signal (1D numpy array)
        fs: Sampling frequency in Hz
        c: Threshold scaling factor (frozen sweep values)
        delta: Explicit threshold value (optional)

    Returns:
        dict containing:
            'xh': reconstructed signal (same shape as x)
            'events': boolean array of event locations
            'num_events': total number of events
            'event_rate': events per second
            'reduction_pct': % reduction vs fixed-rate streaming (fs samples/s)
            'delta': scalar threshold value
    """
    n_samples = len(x)
    if n_samples == 0:
        raise ValueError("Input signal is empty.")

    std_x = float(np.std(x))
    if delta is None:
        if c is None:
            raise ValueError("Either c or delta must be provided.")
        delta = float(c * std_x)
    else:
        delta = float(delta)

    xh = np.empty(n_samples, dtype=float)
    events = np.zeros(n_samples, dtype=bool)

    # First sample is transmitted as the initial reference event
    last = float(x[0])
    xh[0] = last
    events[0] = True
    num_events = 1

    for n in range(1, n_samples):
        val = float(x[n])
        diff = abs(val - last)
        # Trigger when change meets or exceeds delta (and change is non-zero)
        if diff >= delta and diff > 0.0:
            last = val
            events[n] = True
            num_events += 1
        xh[n] = last

    duration_s = n_samples / fs
    event_rate = num_events / duration_s
    reduction_pct = (1.0 - (event_rate / fs)) * 100.0

    return {
        "xh": xh,
        "events": events,
        "num_events": num_events,
        "event_rate": event_rate,
        "reduction_pct": reduction_pct,
        "delta": delta,
    }


def encode_adaptive(
    x: np.ndarray,
    fs: float,
    k: float,
) -> Dict[str, Any]:
    """
    Adaptive threshold send-on-delta encoder.
    delta[n] = max(k * std(xh[n-fs:n]), 0.02 * std(x)),
    using ONLY past reconstructed samples xh (receiver can compute it; no side info; no look-ahead).
    Same trigger and hold as fixed.

    Parameters:
        x: Input signal (1D numpy array)
        fs: Sampling frequency in Hz
        k: Threshold scaling factor (frozen sweep values)

    Returns:
        dict containing:
            'xh': reconstructed signal (same shape as x)
            'events': boolean array of event locations
            'num_events': total number of events
            'event_rate': events per second
            'reduction_pct': % reduction vs fixed-rate streaming (fs samples/s)
            'deltas': array of delta values at each sample
    """
    n_samples = len(x)
    if n_samples == 0:
        raise ValueError("Input signal is empty.")

    std_x = float(np.std(x))
    min_delta = float(ADAPTIVE_MIN_FACTOR * std_x)
    w_size = int(round(fs))

    xh = np.empty(n_samples, dtype=float)
    events = np.zeros(n_samples, dtype=bool)
    deltas = np.empty(n_samples, dtype=float)

    # At n=0: no past samples (past window is empty). delta[0] = min_delta
    deltas[0] = min_delta
    last = float(x[0])
    xh[0] = last
    events[0] = True
    num_events = 1

    sum_v = float(xh[0])
    sum_sq = float(xh[0] * xh[0])

    for n in range(1, n_samples):
        # The past reconstructed window is xh[max(0, n - w_size) : n]
        if n > 1:
            in_val = float(xh[n - 1])
            sum_v += in_val
            sum_sq += in_val * in_val
            if n > w_size:
                out_val = float(xh[n - 1 - w_size])
                sum_v -= out_val
                sum_sq -= out_val * out_val

        # Periodic resynchronization to prevent floating-point drift
        if n % 256 == 0:
            start_idx = max(0, n - w_size)
            past_slice = xh[start_idx:n]
            sum_v = float(np.sum(past_slice))
            sum_sq = float(np.sum(past_slice * past_slice))

        m = min(n, w_size)
        if m < 2:
            past_std = 0.0
        else:
            var = max(0.0, (sum_sq / m) - (sum_v / m) ** 2)
            past_std = math.sqrt(var)

        delta_n = max(k * past_std, min_delta)
        deltas[n] = delta_n

        val = float(x[n])
        diff = abs(val - last)
        if diff >= delta_n and diff > 0.0:
            last = val
            events[n] = True
            num_events += 1
        xh[n] = last

    duration_s = n_samples / fs
    event_rate = num_events / duration_s
    reduction_pct = (1.0 - (event_rate / fs)) * 100.0

    return {
        "xh": xh,
        "events": events,
        "num_events": num_events,
        "event_rate": event_rate,
        "reduction_pct": reduction_pct,
        "deltas": deltas,
    }
