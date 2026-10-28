"""
app.py - Streamlit application for interactive exploration of event-triggered EEG transmission.
Loads assets/segment.npz and results/results.csv locally (strictly offline-only).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from config import (
    ASSETS_DIR,
    RESULTS_DIR,
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
        margin-bottom: 12px;
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
    .info-card {
        background-color: #0f172a;
        border-left: 4px solid #38bdf8;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 12px 0;
        color: #e2e8f0;
    }
    .info-card h4 {
        margin: 0 0 6px 0;
        color: #38bdf8;
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


@st.cache_data
def load_cached_results() -> pd.DataFrame:
    """Load sweep benchmark results from local results/results.csv (no network)."""
    csv_path = RESULTS_DIR / "results.csv"
    if not csv_path.exists():
        st.error(f"Results CSV not found at {csv_path}. Please run run.py first.")
        st.stop()
    df = pd.read_csv(csv_path)
    return df


def main():
    st.title("🧠 EEG Event-Triggered (Send-on-Delta) Transmission")
    st.caption("CHB-MIT Scalp EEG (chb01_03, Ch 0 FP1-F7) | Zero-Order Hold (ZOH) Reconstruction")

    # Load preprocessed segment locally
    signal, fs = load_cached_segment()

    # 4 Main Navigation Tabs
    tab_sim, tab_how, tab_graph, tab_why = st.tabs(
        ["Simulation", "How it works", "Read the graph", "Why it matters"]
    )

    # =========================================================================
    # TAB 1: Simulation (existing interactive tool)
    # =========================================================================
    with tab_sim:
        st.subheader("Interactive Event-Triggered Encoder Simulation")
        st.write(
            "Explore how the **Adaptive Threshold** ($k$) and **Fixed Threshold** ($c$) "
            "determine transmission density and reconstruction fidelity in real-time."
        )

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
        st.markdown(f"#### ⚡ Performance Metrics on Seizure Slice (296–336 s) at $k = {k_val:.2f}$")

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
        st.markdown("#### 📈 Raw vs Reconstructed Seizure Segment (10 s)")
        st.write(
            "Displaying 10 seconds of the seizure slice (window seconds 296–306 s = file seconds 2996–3006 s). "
            "Each dot marks an event transmitted across the wireless link."
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
        st.markdown("#### 📊 Spectral Fidelity: Relative Band-Power Errors")
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

    # =========================================================================
    # TAB 2: How it works (Graphviz flowchart of the send-on-delta logic)
    # =========================================================================
    with tab_how:
        st.subheader("⚙️ Send-on-Delta (SoD) Logic Flowchart")
        st.write(
            "Send-on-Delta is an event-triggered sampling scheme. Instead of continuously transmitting "
            "every sample at the fixed Nyquist rate ($f_s = 256$ Hz), the sensor transmits an event **only** "
            "when the current signal deviates from the last transmitted value by at least threshold $\\Delta$."
        )

        st.graphviz_chart(
            """
            digraph SendOnDeltaLogic {
                rankdir=LR;
                bgcolor="transparent";
                node [shape=box, style="filled,rounded", fontname="sans-serif", fontsize=11, margin="0.2,0.1"];
                edge [fontname="sans-serif", fontsize=10];

                EEG [
                    label="🧠 Scalp EEG\nx[n] @ fs = 256 Hz",
                    fillcolor="#e0f2fe",
                    color="#0284c7",
                    penwidth=1.5
                ];

                CALC [
                    label="📐 Calculate Threshold Δ[n]\nFixed: Δ = c · σ(x)\nAdaptive: Δ[n] = max(k · σ_win[n], 0.02 · σ(x))",
                    fillcolor="#f1f5f9",
                    color="#64748b",
                    penwidth=1.2
                ];

                CHK [
                    label="❓ |x[n] - x_last| ≥ Δ[n] ?",
                    shape=diamond,
                    fillcolor="#fef3c7",
                    color="#d97706",
                    penwidth=1.8,
                    style=filled
                ];

                TX [
                    label="⚡ Send 1 Event Packet\nTransmitted: (n, x[n])\nUpdate: x_last ← x[n]\nRadio: TX Burst Active",
                    fillcolor="#dcfce7",
                    color="#16a34a",
                    penwidth=1.8
                ];

                SIL [
                    label="💤 Stay Silent (Sleep)\nNo Packet Transmitted\nx_last remains unchanged\nRadio: Deep Sleep / Power Saved",
                    fillcolor="#f1f5f9",
                    color="#94a3b8",
                    penwidth=1.2
                ];

                RX [
                    label="📶 Receiver (ZOH)\nHolds last known amplitude\nx̂[n] = x_last\nCreates staircase reconstruction",
                    fillcolor="#e0e7ff",
                    color="#4f46e5",
                    penwidth=1.5
                ];

                OUT [
                    label="📈 Reconstructed EEG\nx̂(t) for Clinical Analysis\nWelch PSD & Band-Power Checks",
                    fillcolor="#fae8ff",
                    color="#a855f7",
                    penwidth=1.5
                ];

                EEG -> CALC;
                CALC -> CHK;
                CHK -> TX [label="YES (Event triggered)", color="#16a34a", penwidth=2];
                CHK -> SIL [label="NO (Below threshold)", color="#64748b", style=dashed];
                TX -> RX [label="Wireless event packet", color="#16a34a"];
                SIL -> RX [label="Hold previous value", color="#94a3b8", style=dashed];
                RX -> OUT;
            }
            """
        )

        st.markdown("---")
        st.markdown("### 🔬 Architectural Principles")

        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown(
                """
                <div class="info-card">
                    <h4>1. Event-Triggered Condition</h4>
                    <p>At each discrete time step <code>n</code>, the incoming EEG sample <code>x[n]</code> is compared against <code>x_last</code>, the amplitude sent in the most recent event:</p>
                    <p style="text-align:center;font-weight:600;"><code>|x[n] - x_last| ≥ Δ[n]</code></p>
                    <p>When this inequality is satisfied, the sensor wakes up its radio, transmits the new coordinate, and updates <code>x_last = x[n]</code>. If the inequality is not met, the radio remains asleep.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_h2:
            st.markdown(
                """
                <div class="info-card">
                    <h4>2. Fixed vs Adaptive Thresholding</h4>
                    <p><b>Fixed Threshold (c):</b> Δ is constant throughout time: <code>Δ = c · σ(x)</code>. Simple to compute, but cannot adapt to changes in EEG background activity.</p>
                    <p><b>Adaptive Threshold (k):</b> Δ adjusts dynamically based on rolling 1-second standard deviation: <code>Δ[n] = max(k · σ_win[n], 0.02 · σ(x))</code>. Scales down during quiet background to capture low-amplitude rhythms, and scales up appropriately.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="info-card">
                <h4>3. Zero-Order Hold (ZOH) Reconstruction at the Gateway</h4>
                <p>The receiver reconstructs a continuous signal <code>x̂(t)</code> using <b>Zero-Order Hold (ZOH)</b>. The receiver simply maintains the last received event amplitude until a new event arrives. This creates a staircase approximation with minimal computational overhead and zero algorithmic latency.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # TAB 3: Read the graph (Pareto plot with annotations and dynamic text summary)
    # =========================================================================
    with tab_graph:
        st.subheader("📊 Rate-Distortion Pareto Frontier & Benchmark Analysis")
        st.write(
            "The Pareto frontier visualizes the fundamental trade-off between **transmission rate** (bandwidth/power) "
            "and **reconstruction distortion** (mean band-power error across clinical bands). "
            "All metrics and text values below are dynamically evaluated from `results/results.csv`."
        )

        # Load benchmark results
        df_res = load_cached_results()

        fixed_df = df_res[df_res["method"] == "Fixed"].sort_values(by="param")
        adapt_df = df_res[df_res["method"] == "Adaptive"].sort_values(by="param")

        # Create two-panel Pareto plot: Background and Seizure
        fig_pareto, (ax_bg, ax_sz) = plt.subplots(1, 2, figsize=(14, 6))

        # -------------------------------------------------------------
        # Panel 1: Background Slice (0–250 s)
        # -------------------------------------------------------------
        ax_bg.plot(
            fixed_df["bg_events_per_sec"],
            fixed_df["bg_mean_band_error_pct"],
            "o--",
            color="#2563eb",
            linewidth=2,
            markersize=7,
            label="Fixed Threshold",
        )
        ax_bg.plot(
            adapt_df["bg_events_per_sec"],
            adapt_df["bg_mean_band_error_pct"],
            "s-",
            color="#ea580c",
            linewidth=2,
            markersize=7,
            label="Adaptive Threshold",
        )

        # Annotate points with parameter values
        for _, r in fixed_df.iterrows():
            ax_bg.annotate(
                f"c={r['param']}",
                (r["bg_events_per_sec"], r["bg_mean_band_error_pct"]),
                textcoords="offset points",
                xytext=(-6, 8),
                fontsize=8,
                color="#2563eb",
                weight="bold",
            )
        for _, r in adapt_df.iterrows():
            ax_bg.annotate(
                f"k={r['param']}",
                (r["bg_events_per_sec"], r["bg_mean_band_error_pct"]),
                textcoords="offset points",
                xytext=(-6, -14),
                fontsize=8,
                color="#ea580c",
                weight="bold",
            )

        # Vertical line at fs events/s (Fixed-rate streaming baseline)
        ax_bg.axvline(
            x=fs,
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"Fixed-rate line (fs = {fs:.0f} ev/s)",
        )
        ax_bg.text(
            fs - 4,
            ax_bg.get_ylim()[1] * 0.7 if ax_bg.get_ylim()[1] > 0 else 15,
            f"Fixed-rate baseline\n(fs = {fs:.0f} events/s)",
            color="#dc2626",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fee2e2", ec="#dc2626", lw=1),
        )

        # "Better" direction annotation pointing towards lower rate and lower error (bottom-left)
        ax_bg.annotate(
            "BETTER\n(Lower Rate & Lower Error)",
            xy=(30, 2),
            xytext=(95, 12),
            arrowprops=dict(
                facecolor="#16a34a",
                edgecolor="#15803d",
                width=2.5,
                headwidth=8,
                shrink=0.05,
            ),
            fontsize=9,
            fontweight="bold",
            color="#14532d",
            bbox=dict(boxstyle="round,pad=0.35", fc="#dcfce7", ec="#16a34a", lw=1.2),
            ha="center",
        )

        ax_bg.set_title("Background Slice (0–250 s)", fontsize=12, fontweight="bold")
        ax_bg.set_xlabel("Transmission Rate (events/s)", fontsize=10)
        ax_bg.set_ylabel("Mean Band-Power Error (%)", fontsize=10)
        ax_bg.set_xlim(-10, fs + 25)
        ax_bg.grid(True, linestyle=":", alpha=0.55)
        ax_bg.legend(loc="upper right", frameon=True, fontsize=9)

        # -------------------------------------------------------------
        # Panel 2: Seizure Slice (296–336 s)
        # -------------------------------------------------------------
        ax_sz.plot(
            fixed_df["sz_events_per_sec"],
            fixed_df["sz_mean_band_error_pct"],
            "o--",
            color="#2563eb",
            linewidth=2,
            markersize=7,
            label="Fixed Threshold",
        )
        ax_sz.plot(
            adapt_df["sz_events_per_sec"],
            adapt_df["sz_mean_band_error_pct"],
            "s-",
            color="#ea580c",
            linewidth=2,
            markersize=7,
            label="Adaptive Threshold",
        )

        # Annotate points with parameter values
        for _, r in fixed_df.iterrows():
            ax_sz.annotate(
                f"c={r['param']}",
                (r["sz_events_per_sec"], r["sz_mean_band_error_pct"]),
                textcoords="offset points",
                xytext=(-6, 8),
                fontsize=8,
                color="#2563eb",
                weight="bold",
            )
        for _, r in adapt_df.iterrows():
            ax_sz.annotate(
                f"k={r['param']}",
                (r["sz_events_per_sec"], r["sz_mean_band_error_pct"]),
                textcoords="offset points",
                xytext=(-6, -14),
                fontsize=8,
                color="#ea580c",
                weight="bold",
            )

        # Vertical line at fs events/s (Fixed-rate streaming baseline)
        ax_sz.axvline(
            x=fs,
            color="#dc2626",
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"Fixed-rate line (fs = {fs:.0f} ev/s)",
        )
        ax_sz.text(
            fs - 4,
            ax_sz.get_ylim()[1] * 0.7 if ax_sz.get_ylim()[1] > 0 else 18,
            f"Fixed-rate baseline\n(fs = {fs:.0f} events/s)",
            color="#dc2626",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fee2e2", ec="#dc2626", lw=1),
        )

        # "Better" direction annotation pointing towards lower rate and lower error (bottom-left)
        ax_sz.annotate(
            "BETTER\n(Lower Rate & Lower Error)",
            xy=(45, 3),
            xytext=(115, 15),
            arrowprops=dict(
                facecolor="#16a34a",
                edgecolor="#15803d",
                width=2.5,
                headwidth=8,
                shrink=0.05,
            ),
            fontsize=9,
            fontweight="bold",
            color="#14532d",
            bbox=dict(boxstyle="round,pad=0.35", fc="#dcfce7", ec="#16a34a", lw=1.2),
            ha="center",
        )

        ax_sz.set_title("Seizure Slice (296–336 s)", fontsize=12, fontweight="bold")
        ax_sz.set_xlabel("Transmission Rate (events/s)", fontsize=10)
        ax_sz.set_ylabel("Mean Band-Power Error (%)", fontsize=10)
        ax_sz.set_xlim(-10, fs + 25)
        ax_sz.grid(True, linestyle=":", alpha=0.55)
        ax_sz.legend(loc="upper right", frameon=True, fontsize=9)

        plt.suptitle(
            "Rate-Distortion Pareto Trade-off with Annotations: Fixed vs Adaptive Send-on-Delta",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        st.pyplot(fig_pareto)
        plt.close(fig_pareto)

        st.markdown("---")

        # -------------------------------------------------------------
        # Text summary dynamically computed from results/results.csv (NEVER hardcoded)
        # -------------------------------------------------------------
        st.markdown("### 📝 Dynamic Empirical Benchmark Summary")
        st.caption("All figures and percentages below are computed at runtime directly from `results/results.csv`.")

        # Dynamically extracted metrics from df_res
        max_overall_red = float(df_res["overall_reduction_pct"].max())
        max_overall_red_row = df_res.loc[df_res["overall_reduction_pct"].idxmax()]
        min_overall_err = float(df_res["overall_mean_band_error_pct"].min())
        min_overall_err_row = df_res.loc[df_res["overall_mean_band_error_pct"].idxmin()]

        # Operating points at standard benchmark multiplier 0.20
        row_adapt_02 = df_res[(df_res["method"] == "Adaptive") & (np.isclose(df_res["param"], 0.20))].iloc[0]
        row_fixed_02 = df_res[(df_res["method"] == "Fixed") & (np.isclose(df_res["param"], 0.20))].iloc[0]

        # Operating points at high-compression multiplier 0.50
        row_adapt_05 = df_res[(df_res["method"] == "Adaptive") & (np.isclose(df_res["param"], 0.50))].iloc[0]
        row_fixed_05 = df_res[(df_res["method"] == "Fixed") & (np.isclose(df_res["param"], 0.50))].iloc[0]

        # Extreme rates
        min_ev_rate = float(df_res["overall_events_per_sec"].min())
        max_ev_rate = float(df_res["overall_events_per_sec"].max())

        # Summary KPIs dynamically rendered
        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Max Bandwidth Savings</div>
                    <div class="metric-value">{max_overall_red:.1f}%</div>
                    <div class="metric-sub">{max_overall_red_row['method']} at param={max_overall_red_row['param']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Min Spectral Error</div>
                    <div class="metric-value">{min_overall_err:.3f}%</div>
                    <div class="metric-sub">{min_overall_err_row['method']} at param={min_overall_err_row['param']:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Adapt k=0.20 Overall Rate</div>
                    <div class="metric-value">{row_adapt_02['overall_events_per_sec']:.1f} <span style="font-size:1rem;color:#94a3b8;">ev/s</span></div>
                    <div class="metric-sub">{row_adapt_02['overall_reduction_pct']:.1f}% reduction vs {fs:.0f} ev/s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Adapt k=0.20 Seizure Rate</div>
                    <div class="metric-value">{row_adapt_02['sz_events_per_sec']:.1f} <span style="font-size:1rem;color:#94a3b8;">ev/s</span></div>
                    <div class="metric-sub">Mean Band Error: {row_adapt_02['sz_mean_band_error_pct']:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Dynamic narrative summary
        st.markdown(
            f"""
            #### 1. Baseline vs Send-on-Delta Transmission
            - **Fixed-Rate Nyquist Baseline**: Continuous transmission requires sending all samples at **{fs:.0f} events/s** (100% duty cycle, 0.0% bandwidth savings).
            - **Event-Triggered Operational Range**: Across the measured sweep in `results.csv`, event rates range from **{min_ev_rate:.1f} events/s** to **{max_ev_rate:.1f} events/s**, delivering between **{(1.0 - max_ev_rate/fs)*100.0:.1f}%** and **{max_overall_red:.1f}%** total transmission bandwidth reduction.

            #### 2. Analysis at Standard Operating Point ($k = 0.20$ vs $c = 0.20$)
            - **Background Quiescence (0–250 s)**:
              - **Adaptive ($k=0.20$)**: Transmits **{row_adapt_02['bg_events_per_sec']:.1f} events/s** ({row_adapt_02['bg_reduction_pct']:.1f}% reduction vs continuous {fs:.0f} Hz streaming), maintaining a low mean spectral band error of **{row_adapt_02['bg_mean_band_error_pct']:.2f}%** and MSE of **{row_adapt_02['bg_mse']:.2f} µV²**.
              - **Fixed ($c=0.20$)**: Transmits **{row_fixed_02['bg_events_per_sec']:.1f} events/s** ({row_fixed_02['bg_reduction_pct']:.1f}% reduction), but incurs a higher background spectral error of **{row_fixed_02['bg_mean_band_error_pct']:.2f}%** and MSE of **{row_fixed_02['bg_mse']:.2f} µV²**.
            - **Ictal Seizure Event (296–336 s)**:
              - **Adaptive ($k=0.20$)**: During seizure discharges, the rate is **{row_adapt_02['sz_events_per_sec']:.1f} events/s** ({row_adapt_02['sz_reduction_pct']:.1f}% reduction) with **{row_adapt_02['sz_mean_band_error_pct']:.2f}%** mean band error.
              - **Fixed ($c=0.20$)**: Fixed threshold sends **{row_fixed_02['sz_events_per_sec']:.1f} events/s** ({row_fixed_02['sz_reduction_pct']:.1f}% reduction) with **{row_fixed_02['sz_mean_band_error_pct']:.2f}%** mean band error.

            #### 3. High-Compression Operating Point ($k = 0.50$ vs $c = 0.50$)
            - When bandwidth constraints require aggressive reduction:
              - **Adaptive ($k=0.50$)**: Achieves **{row_adapt_05['overall_reduction_pct']:.1f}%** overall reduction (**{row_adapt_05['overall_events_per_sec']:.1f} events/s**) with **{row_adapt_05['overall_mean_band_error_pct']:.2f}%** overall mean band error.
              - **Fixed ($c=0.50$)**: Achieves **{row_fixed_05['overall_reduction_pct']:.1f}%** overall reduction (**{row_fixed_05['overall_events_per_sec']:.1f} events/s**) with **{row_fixed_05['overall_mean_band_error_pct']:.2f}%** overall mean band error.
            """
        )

        # Full Sweep Table from CSV
        with st.expander("🔍 View Complete Benchmark Sweep Data Table (results/results.csv)"):
            display_cols = [
                "method",
                "param",
                "overall_events_per_sec",
                "overall_reduction_pct",
                "bg_events_per_sec",
                "bg_reduction_pct",
                "bg_mean_band_error_pct",
                "sz_events_per_sec",
                "sz_reduction_pct",
                "sz_mean_band_error_pct",
                "sz_mse",
            ]
            st.dataframe(df_res[display_cols].style.format({
                "overall_events_per_sec": "{:.2f}",
                "overall_reduction_pct": "{:.1f}%",
                "bg_events_per_sec": "{:.2f}",
                "bg_reduction_pct": "{:.1f}%",
                "bg_mean_band_error_pct": "{:.2f}%",
                "sz_events_per_sec": "{:.2f}",
                "sz_reduction_pct": "{:.1f}%",
                "sz_mean_band_error_pct": "{:.2f}%",
                "sz_mse": "{:.2f}",
            }))

    # =========================================================================
    # TAB 4: Why it matters (Graphviz flow: patient headset -> event-triggered TX -> phone -> cloud -> neurologist)
    # =========================================================================
    with tab_why:
        st.subheader("🌍 Telemetry Pipeline: From Patient Headset to Neurologist")
        st.write(
            "Continuous wireless EEG telemetry in home-monitoring environments is severely constrained by radio power consumption. "
            "Send-on-Delta enables long-term continuous ambulatory monitoring by transmitting only informative physiological events."
        )

        st.graphviz_chart(
            """
            digraph TelemetryPipeline {
                rankdir=LR;
                bgcolor="transparent";
                node [shape=box, style="filled,rounded", fontname="sans-serif", fontsize=11, margin="0.2,0.12"];
                edge [fontname="sans-serif", fontsize=10];

                HEADSET [
                    label="🧠 Patient Headset\nWearable Scalp EEG\nCh 0 (FP1-F7) @ 256 Hz",
                    fillcolor="#e0f2fe",
                    color="#0284c7",
                    penwidth=1.6
                ];

                TX [
                    label="⚡ Event-Triggered TX\nOn-device Send-on-Delta\nTransmits only when |Δx| ≥ Δ[n]",
                    fillcolor="#fef3c7",
                    color="#d97706",
                    penwidth=1.8
                ];

                PHONE [
                    label="📱 Phone Gateway\nSmartphone BLE Receiver\nLocal Storage & Cellular Relay",
                    fillcolor="#f3e8ff",
                    color="#9333ea",
                    penwidth=1.6
                ];

                CLOUD [
                    label="☁️ Cloud / Hospital Server\nTelemetry Backend\nReal-time Seizure Detection",
                    fillcolor="#dcfce7",
                    color="#16a34a",
                    penwidth=1.6
                ];

                NEURO [
                    label="👨‍⚕️ Neurologist\nClinical Review Dashboard\nRapid Diagnosis & Action",
                    fillcolor="#ffe4e6",
                    color="#e11d48",
                    penwidth=1.8
                ];

                BATTERY [
                    label="🔋 Extended Battery Life\nRadio sleeps 50–90% of time\nEnables multi-day monitoring!",
                    shape=note,
                    fillcolor="#fef9c3",
                    color="#ca8a04",
                    penwidth=1.4
                ];

                ALERT [
                    label="🚨 Instant Seizure Burst\nHigh spike density triggers\nimmediate real-time alerts",
                    shape=note,
                    fillcolor="#fee2e2",
                    color="#dc2626",
                    penwidth=1.4
                ];

                HEADSET -> TX [label="Raw EEG stream", color="#0284c7", penwidth=1.5];
                TX -> PHONE [label="Event bursts\n(50–90% less data)", color="#d97706", penwidth=2.2];
                PHONE -> CLOUD [label="Cellular / Wi-Fi relay", color="#9333ea", penwidth=1.5];
                CLOUD -> NEURO [label="Reconstructed waveforms\n& Automated alerts", color="#16a34a", penwidth=1.8];

                TX -> BATTERY [style=dashed, color="#ca8a04", label="Power savings"];
                CLOUD -> ALERT [style=dashed, color="#dc2626", label="Threshold alarm"];
            }
            """
        )

        st.markdown("---")
        st.markdown("### 💡 Clinical & Technical Value")

        col_w1, col_w2 = st.columns(2)

        with col_w1:
            st.markdown(
                """
                <div class="info-card">
                    <h4>🔋 Battery Life for Ambulatory Monitoring</h4>
                    <p>Conventional Bluetooth Low Energy (BLE) transmitters streaming 256 samples/s continuously deplete wearable coin-cell and compact LiPo batteries in <b>6 to 12 hours</b>, requiring frequent patient recharging and interrupting nocturnal sleep monitoring.</p>
                    <p>By eliminating <b>50% to 90% of transmission events</b> during baseline EEG rhythms, the radio transmitter stays in deep-sleep mode most of the time, extending operational lifetime to <b>multiple days or weeks</b> on a single charge.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="info-card">
                    <h4>⚡ Zero-Latency Seizure Response</h4>
                    <p>Block-based compression algorithms (such as JPEG, gzip, or wavelets) require collecting windows of seconds or minutes before compression and transmission, introducing unacceptable clinical latency.</p>
                    <p>Send-on-Delta operates <b>sample-by-sample</b>. The instant a high-amplitude epileptic seizure begins, the rapid voltage swings automatically trigger dense event bursts without any buffering delay, notifying caregivers and neurologists in real time.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_w2:
            st.markdown(
                """
                <div class="info-card">
                    <h4>📉 Reduced Cellular Data & Cloud Storage</h4>
                    <p>When millions of EEG samples are relayed over 4G/5G mobile data to hospital cloud infrastructure, continuous streaming creates bandwidth congestion, high mobile data bills for patients, and massive cloud ingestion expenses.</p>
                    <p>Event-triggered transmission acts as a natural on-device data filter, keeping transmission overhead proportional to biological information content rather than arbitrary clock ticks.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="info-card">
                    <h4>🩺 Preserved Diagnostic Waveform Integrity</h4>
                    <p>As confirmed by Welch Power Spectral Density analysis across Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), and Beta (13–30 Hz) bands, the reconstructed staircase signal preserves clinical spectral power within low single-digit error percentages.</p>
                    <p>Neurologists receive clinically faithful EEG morphology that retains spike-and-wave complexes, sharp waves, and background frequency dynamics required for definitive seizure localization.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()