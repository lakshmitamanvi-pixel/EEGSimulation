"""
app.py - Streamlit application for interactive exploration of event-triggered EEG transmission.
Loads assets/segment.npz only (no network).
"""
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from config import (
    ASSETS_DIR,
    SEIZURE_WINDOW_START_S,
    BANDS,
)
from encoders import encode_adaptive, encode_fixed
from metrics import evaluate_reconstruction

# Set page config
st.set_page_config(
    page_title="EEG Event-Triggered Transmission",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS for polished aesthetic
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_cached_segment() -> tuple[np.ndarray, float]:
    """Load preprocessed EEG segment from local assets/segment.npz only (no network)."""
    npz_path = ASSETS_DIR / "segment.npz"
    if not npz_path.exists():
        st.error(f"Asset file not found at {npz_path}. Please run data.py first.")
        st.stop()
    data = np.load(npz_path)
    signal = data["signal"]
    fs = float(data["fs"])
    return signal, fs


def main():
    st.title("🧠 EEG Event-Triggered (Send-on-Delta) Transmission")
    st.caption("CHB-MIT Scalp EEG (chb01_03, Ch 0 FP1-F7) | Zero-Order Hold (ZOH) Reconstruction")

    # Load preprocessed segment locally
    signal, fs = load_cached_segment()

    # Sidebar controls
    st.sidebar.header("Encoder Parameters")
    st.sidebar.markdown("Configure the **Adaptive Threshold** multiplier $k$:")
    k_val = st.sidebar.slider(
        "Adaptive Threshold Factor (k)",
        min_value=0.05,
        max_value=0.80,
        value=0.20,
        step=0.01,
        help="Scaling factor for rolling standard deviation: delta[n] = max(k * std(xh[n-fs:n]), 0.02 * std(x))",
    )

    compare_fixed = st.sidebar.checkbox(
        "Compare with Fixed Threshold (c = k)",
        value=True,
        help="Overlay fixed threshold send-on-delta with c equal to the selected k.",
    )

    # Encode with adaptive threshold
    enc = encode_adaptive(signal, fs, k=k_val)
    evals = evaluate_reconstruction(signal, enc["xh"], enc["events"], fs)

    sz_eval = evals["seizure"]
    bg_eval = evals["background"]
    ov_eval = evals["overall"]

    # Fixed encoder for comparison if enabled
    if compare_fixed:
        enc_fix = encode_fixed(signal, fs, c=k_val)
        evals_fix = evaluate_reconstruction(signal, enc_fix["xh"], enc_fix["events"], fs)
        sz_eval_fix = evals_fix["seizure"]

    # Display KPI metrics row for the Seizure Slice
    st.subheader(f"⚡ Performance Metrics on Seizure Slice (296–336 s) at k = {k_val:.2f}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Seizure Event Rate</div>
                <div class="metric-value">{sz_eval['events_per_sec']:.1f} <span style="font-size:1rem;color:#94a3b8;">ev/s</span></div>
                <div class="metric-sub">Baseline: {fs:.0f} samples/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Bandwidth Reduction</div>
                <div class="metric-value">{sz_eval['duration_s'] and (1.0 - sz_eval['events_per_sec']/fs)*100.0:.1f}%</div>
                <div class="metric-sub">Savings vs Continuous Streaming</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Mean Band-Power Error</div>
                <div class="metric-value">{sz_eval['mean_band_error_pct']:.2f}%</div>
                <div class="metric-sub">Welch PSD across 4 Bands</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Seizure Slice MSE</div>
                <div class="metric-value">{sz_eval['mse']:.2f} <span style="font-size:1rem;color:#94a3b8;">µV²</span></div>
                <div class="metric-sub">Mean Squared Error vs Raw</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Raw vs Reconstructed Seizure Segment (10 seconds)
    st.subheader("📈 Raw vs Reconstructed Seizure Segment (10 s)")
    st.write(
        "Displaying 10 seconds of the seizure slice (window seconds 296–306 s = file seconds 2996–3006 s). "
        "Each dot marks an event transmitted across the link."
    )

    seg_start_s = SEIZURE_WINDOW_START_S
    seg_end_s = seg_start_s + 10.0
    idx_start = int(round(seg_start_s * fs))
    idx_end = int(round(seg_end_s * fs))

    t_axis = np.linspace(0, 10.0, idx_end - idx_start)
    raw_seg = signal[idx_start:idx_end]
    recon_seg = enc["xh"][idx_start:idx_end]
    events_seg = enc["events"][idx_start:idx_end]
    event_times = t_axis[events_seg]
    event_vals = recon_seg[events_seg]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(t_axis, raw_seg, color="#64748b", linewidth=1.2, alpha=0.8, label="Raw Signal x(t)")
    ax.step(t_axis, recon_seg, where="post", color="#ef4444", linewidth=1.8, label=f"Adaptive Reconstruction x̂(t) (k={k_val:.2f})")
    ax.scatter(event_times, event_vals, color="#0284c7", s=22, zorder=4, label=f"Transmitted Events ({len(event_times)} events)")

    if compare_fixed:
        recon_fix_seg = enc_fix["xh"][idx_start:idx_end]
        ax.step(t_axis, recon_fix_seg, where="post", color="#10b981", linewidth=1.3, linestyle="--", alpha=0.8, label=f"Fixed Reconstruction (c={k_val:.2f})")

    ax.set_xlabel("Time within Seizure Segment (s)")
    ax.set_ylabel("EEG Amplitude (µV)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    st.pyplot(fig)
    plt.close(fig)

    # Per-band error comparison
    st.subheader("📊 Spectral Fidelity: Relative Band-Power Errors")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**Seizure Slice vs Background Slice Band Errors (%)**")
        band_names = list(BANDS.keys())
        fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
        x_pos = np.arange(len(band_names))
        w = 0.35
        sz_vals = [sz_eval["rel_band_errors"][b] * 100.0 for b in band_names]
        bg_vals = [bg_eval["rel_band_errors"][b] * 100.0 for b in band_names]

        b1 = ax_bar.bar(x_pos - w/2, bg_vals, w, label="Background (0–250 s)", color="#3b82f6")
        b2 = ax_bar.bar(x_pos + w/2, sz_vals, w, label="Seizure (296–336 s)", color="#f97316")

        for b in b1:
            ax_bar.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
        for b in b2:
            ax_bar.annotate(f"{b.get_height():.1f}%", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                            xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)

        ax_bar.set_xticks(x_pos)
        ax_bar.set_xticklabels([f"{b.capitalize()}" for b in band_names])
        ax_bar.set_ylabel("Error (%)")
        ax_bar.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax_bar.legend()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with col_b:
        st.markdown("**Numerical Breakdown**")
        table_data = []
        for b in band_names:
            low, high = BANDS[b]
            table_data.append({
                "Band": f"{b.capitalize()} ({low}–{high} Hz)",
                "Background Error (%)": f"{bg_eval['rel_band_errors'][b] * 100.0:.2f}%",
                "Seizure Error (%)": f"{sz_eval['rel_band_errors'][b] * 100.0:.2f}%",
            })
        st.table(table_data)

        if compare_fixed:
            st.markdown(
                f"""
                **Comparative Summary at multiplier = {k_val:.2f}:**
                - **Adaptive Seizure Rate**: {sz_eval['events_per_sec']:.1f} ev/s vs **Fixed**: {sz_eval_fix['events_per_sec']:.1f} ev/s
                - **Adaptive Seizure Mean Band Err**: {sz_eval['mean_band_error_pct']:.2f}% vs **Fixed**: {sz_eval_fix['mean_band_error_pct']:.2f}%
                """
            )


if __name__ == "__main__":
    main()
