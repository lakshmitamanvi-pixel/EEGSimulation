"""
data.py - Data ingestion, verification, and preprocessing for eeg-event-tx.
"""
import sys
import time
import urllib.request
import urllib.error
import re
from pathlib import Path
import numpy as np
import mne

from config import (
    DATA_DIR,
    ASSETS_DIR,
    BASE_URL,
    EDF_FILE,
    SUMMARY_FILE,
    CHANNEL_INDEX,
    WINDOW_START_S,
    WINDOW_END_S,
    SEIZURE_FILE_START_S,
    SEIZURE_FILE_END_S,
)


def download_file(filename: str, max_retries: int = 3, timeout: int = 60) -> Path:
    """Download a file from PhysioNet with retries, skipping if already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = DATA_DIR / filename
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[data.py] {filename} already exists at {dest_path}, skipping download.", flush=True)
        return dest_path

    url = BASE_URL + filename
    print(f"[data.py] Downloading {url} -> {dest_path}...", flush=True)
    temp_path = DATA_DIR / f"{filename}.part"

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(temp_path, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)  # 1 MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"[data.py] {downloaded}/{total_size} bytes ({pct:.1f}%)", flush=True)

            if temp_path.exists():
                temp_path.replace(dest_path)
            print(f"[data.py] Successfully downloaded {filename} ({dest_path.stat().st_size} bytes).", flush=True)
            return dest_path
        except Exception as err:
            print(f"[data.py] Attempt {attempt}/{max_retries} failed for {filename}: {err}", flush=True)
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {filename} after {max_retries} attempts.") from err
            time.sleep(2 * attempt)
    return dest_path


def verify_seizure_annotations(summary_path: Path) -> tuple[int, int]:
    """
    Parse chb01-summary.txt for chb01_03.edf and assert seizure start == 2996 and end == 3036.
    If not, STOP and print what the file says. Do NOT edit the seizure times.
    """
    with open(summary_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    # Search for section corresponding to chb01_03.edf
    pattern = rf"File Name:\s*{re.escape(EDF_FILE)}.*?(?=File Name:|\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        print(f"[ERROR] Could not find section for {EDF_FILE} in {summary_path}!\nFull text content:\n{text}", flush=True)
        sys.exit(1)

    section = match.group(0)
    start_match = re.search(r"Seizure\s+(?:1\s+)?Start\s+Time:\s*(\d+)\s*seconds", section, re.IGNORECASE)
    end_match = re.search(r"Seizure\s+(?:1\s+)?End\s+Time:\s*(\d+)\s*seconds", section, re.IGNORECASE)

    if not start_match or not end_match:
        print(f"[ERROR] Failed to parse seizure start/end times in section for {EDF_FILE}:\n{section}", flush=True)
        sys.exit(1)

    start_s = int(start_match.group(1))
    end_s = int(end_match.group(1))

    print(f"[data.py] Parsed seizure annotation for {EDF_FILE}: start={start_s}s, end={end_s}s", flush=True)

    if start_s != SEIZURE_FILE_START_S or end_s != SEIZURE_FILE_END_S:
        print(
            f"[ERROR] Seizure time mismatch! Expected start={SEIZURE_FILE_START_S}, end={SEIZURE_FILE_END_S}.\n"
            f"File reports: start={start_s}, end={end_s}.\nSection content:\n{section}",
            flush=True,
        )
        sys.exit(1)

    return start_s, end_s


def load_and_preprocess(edf_path: Path) -> tuple[np.ndarray, float]:
    """
    Read with mne (preload=True, verbose=False).
    Take channel 0, slice window (2700-3300s), convert V to µV, subtract mean.
    """
    print(f"[data.py] Reading {edf_path} with MNE...", flush=True)
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    fs = float(raw.info["sfreq"])
    channel_name = raw.ch_names[CHANNEL_INDEX]
    print(f"[data.py] Sampling rate: {fs} Hz, Channel {CHANNEL_INDEX}: {channel_name}", flush=True)

    # Extract channel 0 data (MNE raw.get_data gives shape (n_channels, n_times))
    raw_channel = raw.get_data(picks=[CHANNEL_INDEX])[0]

    # Slice window: 2700 - 3300 s (600 s)
    start_idx = int(round(WINDOW_START_S * fs))
    end_idx = int(round(WINDOW_END_S * fs))
    windowed = raw_channel[start_idx:end_idx]

    # Convert V to µV (MNE stores EDF in Volts)
    signal_uv = windowed * 1e6

    # Subtract mean
    signal = signal_uv - np.mean(signal_uv)

    print(f"[data.py] Preprocessed signal length: {len(signal)} samples ({len(signal)/fs:.1f} s), mean={np.mean(signal):.4e} µV, std={np.std(signal):.4f} µV", flush=True)
    return signal, fs


def prepare_data() -> Path:
    """Download, verify, process, and save segment to assets/segment.npz."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = ASSETS_DIR / "segment.npz"

    # Step 1: Download summary and EDF
    summary_path = download_file(SUMMARY_FILE)
    edf_path = download_file(EDF_FILE)

    # Step 2: Verify seizure annotation
    verify_seizure_annotations(summary_path)

    # Step 3: Load and preprocess signal
    signal, fs = load_and_preprocess(edf_path)

    # Step 4: Save to assets/segment.npz
    np.savez_compressed(out_npz, signal=signal, fs=fs)
    print(f"[data.py] Saved preprocessed segment to {out_npz}", flush=True)
    return out_npz


if __name__ == "__main__":
    prepare_data()
