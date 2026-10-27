"""
run.py - Experiment sweep, metric collection, visualization generation, and reporting.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    ASSETS_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    SWEEP_VALUES,
    SEIZURE_WINDOW_START_S,
    SEIZURE_WINDOW_END_S,
    BANDS,
)
from encoders import encode_fixed, encode_adaptive
from metrics import evaluate_reconstruction
from data import prepare_data


def load_dataset() -> tuple[np.ndarray, float]:
    """Load preprocessed segment from assets/segment.npz, generating it if needed."""
    npz_path = ASSETS_DIR / "segment.npz"
    if not npz_path.exists():
        print("[run.py] assets/segment.npz not found, running prepare_data()...")
        prepare_data()

    data = np.load(npz_path)
    signal = data["signal"]
    fs = float(data["fs"])
    print(f"[run.py] Loaded segment.npz: {len(signal)} samples, fs={fs} Hz")
    return signal, fs


def run_sweep(signal: np.ndarray, fs: float) -> pd.DataFrame:
    """Run sweep over fixed c and adaptive k values, recording detailed metrics."""
    records = []

    print("\n[run.py] Running sweep over parameter values:", SWEEP_VALUES)

    # 1. Fixed threshold sweep
    for c in SWEEP_VALUES:
        print(f"  Encoding Fixed c={c}...")
        enc = encode_fixed(signal, fs, c=c)
        evals = evaluate_reconstruction(signal, enc["xh"], enc["events"], fs)

        bg = evals["background"]
        sz = evals["seizure"]
        ov = evals["overall"]

        rec = {
            "method": "Fixed",
            "param": c,
            "overall_events_per_sec": ov["events_per_sec"],
            "overall_reduction_pct": ov["duration_s"] and (1.0 - ov["events_per_sec"] / fs) * 100.0,
            "overall_mse": ov["mse"],
            "overall_mean_band_error_pct": ov["mean_band_error_pct"],
            "bg_events_per_sec": bg["events_per_sec"],
            "bg_reduction_pct": (1.0 - bg["events_per_sec"] / fs) * 100.0,
            "bg_mse": bg["mse"],
            "bg_mean_band_error_pct": bg["mean_band_error_pct"],
            "sz_events_per_sec": sz["events_per_sec"],
            "sz_reduction_pct": (1.0 - sz["events_per_sec"] / fs) * 100.0,
            "sz_mse": sz["mse"],
            "sz_mean_band_error_pct": sz["mean_band_error_pct"],
        }
        for band in BANDS:
            rec[f"bg_{band}_err_pct"] = bg["rel_band_errors"][band] * 100.0
            rec[f"sz_{band}_err_pct"] = sz["rel_band_errors"][band] * 100.0

        records.append(rec)

    # 2. Adaptive threshold sweep
    for k in SWEEP_VALUES:
        print(f"  Encoding Adaptive k={k}...")
        enc = encode_adaptive(signal, fs, k=k)
        evals = evaluate_reconstruction(signal, enc["xh"], enc["events"], fs)

        bg = evals["background"]
        sz = evals["seizure"]
        ov = evals["overall"]

        rec = {
            "method": "Adaptive",
            "param": k,
            "overall_events_per_sec": ov["events_per_sec"],
            "overall_reduction_pct": (1.0 - ov["events_per_sec"] / fs) * 100.0,
            "overall_mse": ov["mse"],
            "overall_mean_band_error_pct": ov["mean_band_error_pct"],
            "bg_events_per_sec": bg["events_per_sec"],
            "bg_reduction_pct": (1.0 - bg["events_per_sec"] / fs) * 100.0,
            "bg_mse": bg["mse"],
            "bg_mean_band_error_pct": bg["mean_band_error_pct"],
            "sz_events_per_sec": sz["events_per_sec"],
            "sz_reduction_pct": (1.0 - sz["events_per_sec"] / fs) * 100.0,
            "sz_mse": sz["mse"],
            "sz_mean_band_error_pct": sz["mean_band_error_pct"],
        }
        for band in BANDS:
            rec[f"bg_{band}_err_pct"] = bg["rel_band_errors"][band] * 100.0
            rec[f"sz_{band}_err_pct"] = sz["rel_band_errors"][band] * 100.0

        records.append(rec)

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[run.py] Saved sweep results to {csv_path}")
    return df


def generate_figures(signal: np.ndarray, fs: float, df: pd.DataFrame):
    """Generate the 3 required figures at 300 dpi with labelled axes and units."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})

    fixed_df = df[df["method"] == "Fixed"].sort_values(by="param")
    adapt_df = df[df["method"] == "Adaptive"].sort_values(by="param")

    # -------------------------------------------------------------
    # Figure 1: Pareto Trade-off
    # x = events/s, y = mean band-power error %, two panels (background, seizure)
    # -------------------------------------------------------------
    fig1, (ax_bg, ax_sz) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # Panel 1: Background Slice
    ax_bg.plot(
        fixed_df["bg_events_per_sec"],
        fixed_df["bg_mean_band_error_pct"],
        "o--",
        color="#2b5c8f",
        label="Fixed Threshold",
        linewidth=2,
        markersize=7,
    )
    ax_bg.plot(
        adapt_df["bg_events_per_sec"],
        adapt_df["bg_mean_band_error_pct"],
        "s-",
        color="#d95f02",
        label="Adaptive Threshold",
        linewidth=2,
        markersize=7,
    )
    for _, row in fixed_df.iterrows():
        ax_bg.annotate(
            f"c={row['param']}",
            (row["bg_events_per_sec"], row["bg_mean_band_error_pct"]),
            textcoords="offset points",
            xytext=(-5, 8),
            fontsize=9,
            color="#2b5c8f",
        )
    for _, row in adapt_df.iterrows():
        ax_bg.annotate(
            f"k={row['param']}",
            (row["bg_events_per_sec"], row["bg_mean_band_error_pct"]),
            textcoords="offset points",
            xytext=(-5, -14),
            fontsize=9,
            color="#d95f02",
        )
    ax_bg.set_title("Background Slice (0–250 s)", fontweight="bold")
    ax_bg.set_xlabel("Transmission Rate (events/s)")
    ax_bg.set_ylabel("Mean Band-Power Error (%)")
    ax_bg.grid(True, linestyle=":", alpha=0.6)
    ax_bg.legend(frameon=True)

    # Panel 2: Seizure Slice
    ax_sz.plot(
        fixed_df["sz_events_per_sec"],
        fixed_df["sz_mean_band_error_pct"],
        "o--",
        color="#2b5c8f",
        label="Fixed Threshold",
        linewidth=2,
        markersize=7,
    )
    ax_sz.plot(
        adapt_df["sz_events_per_sec"],
        adapt_df["sz_mean_band_error_pct"],
        "s-",
        color="#d95f02",
        label="Adaptive Threshold",
        linewidth=2,
        markersize=7,
    )
    for _, row in fixed_df.iterrows():
        ax_sz.annotate(
            f"c={row['param']}",
            (row["sz_events_per_sec"], row["sz_mean_band_error_pct"]),
            textcoords="offset points",
            xytext=(-5, 8),
            fontsize=9,
            color="#2b5c8f",
        )
    for _, row in adapt_df.iterrows():
        ax_sz.annotate(
            f"k={row['param']}",
            (row["sz_events_per_sec"], row["sz_mean_band_error_pct"]),
            textcoords="offset points",
            xytext=(-5, -14),
            fontsize=9,
            color="#d95f02",
        )
    ax_sz.set_title("Seizure Slice (296–336 s)", fontweight="bold")
    ax_sz.set_xlabel("Transmission Rate (events/s)")
    ax_sz.set_ylabel("Mean Band-Power Error (%)")
    ax_sz.grid(True, linestyle=":", alpha=0.6)
    ax_sz.legend(frameon=True)

    plt.suptitle("Rate-Distortion Pareto Frontier: Fixed vs Adaptive Send-on-Delta", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "pareto_tradeoff.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"[run.py] Saved Figure 1 to {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Raw vs reconstructed seizure segment (10 s of the seizure slice), adaptive k=0.2
    # -------------------------------------------------------------
    # Seizure slice is 296-336s. Take 10s from 296s to 306s (onset/core ictal segment)
    seg_start_s = SEIZURE_WINDOW_START_S
    seg_end_s = seg_start_s + 10.0
    idx_start = int(round(seg_start_s * fs))
    idx_end = int(round(seg_end_s * fs))

    enc_k02 = encode_adaptive(signal, fs, k=0.2)
    t_axis = np.linspace(0, 10.0, idx_end - idx_start)
    raw_seg = signal[idx_start:idx_end]
    recon_seg = enc_k02["xh"][idx_start:idx_end]
    events_seg = enc_k02["events"][idx_start:idx_end]
    event_times = t_axis[events_seg]
    event_amplitudes = recon_seg[events_seg]

    fig2, ax2 = plt.subplots(figsize=(13, 5))
    ax2.plot(t_axis, raw_seg, color="#4a5568", linewidth=1.2, alpha=0.7, label="Raw EEG signal (x)")
    ax2.step(t_axis, recon_seg, where="post", color="#e53e3e", linewidth=1.8, label="Reconstructed staircase (xh, ZOH)")
    ax2.scatter(event_times, event_amplitudes, color="#2b6cb0", s=18, zorder=4, label=f"Events ({len(event_times)} events / 10 s)")

    ax2.set_title("Raw vs Reconstructed Seizure Segment (10 s, Adaptive k=0.2)", fontweight="bold")
    ax2.set_xlabel("Time within segment (s)")
    ax2.set_ylabel("EEG Amplitude (µV)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "seizure_reconstruction.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"[run.py] Saved Figure 2 to {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Per-band error bar chart at k = 0.2, seizure vs background
    # -------------------------------------------------------------
    row_k02 = adapt_df[adapt_df["param"] == 0.2].iloc[0]
    band_names = list(BANDS.keys())
    bg_errors = [row_k02[f"bg_{b}_err_pct"] for b in band_names]
    sz_errors = [row_k02[f"sz_{b}_err_pct"] for b in band_names]

    x_pos = np.arange(len(band_names))
    bar_width = 0.35

    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    bars1 = ax3.bar(x_pos - bar_width / 2, bg_errors, bar_width, label="Background (0–250 s)", color="#3182ce")
    bars2 = ax3.bar(x_pos + bar_width / 2, sz_errors, bar_width, label="Seizure (296–336 s)", color="#dd6b20")

    # Add values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax3.set_title("Relative Band-Power Error per EEG Band (Adaptive k = 0.2)", fontweight="bold")
    ax3.set_xlabel("Frequency Band (Hz)")
    ax3.set_ylabel("Relative Band-Power Error (%)")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"{b.capitalize()}\n({BANDS[b][0]}–{BANDS[b][1]} Hz)" for b in band_names])
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    ax3.legend(frameon=True)

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "band_errors_k02.png"
    fig3.savefig(fig3_path, dpi=300)
    plt.close(fig3)
    print(f"[run.py] Saved Figure 3 to {fig3_path}")


def print_summary_table(df: pd.DataFrame):
    """Print an unvarnished summary table of all computed results."""
    print("\n" + "=" * 105)
    print("EXPERIMENTAL SUMMARY TABLE (REAL COMPUTATION)")
    print("=" * 105)
    headers = [
        "Method", "Param", "Ov Rate (ev/s)", "Ov Red %", "BG Rate (ev/s)", "BG Err %",
        "SZ Rate (ev/s)", "SZ Err %", "SZ Delta %", "SZ Beta %"
    ]
    format_row = "{:<9} {:<6} {:>14} {:>9} {:>14} {:>10} {:>14} {:>10} {:>11} {:>11}"
    print(format_row.format(*headers))
    print("-" * 105)

    for _, r in df.iterrows():
        print(format_row.format(
            r["method"],
            f"{r['param']:.2f}",
            f"{r['overall_events_per_sec']:.1f}",
            f"{r['overall_reduction_pct']:.1f}%",
            f"{r['bg_events_per_sec']:.1f}",
            f"{r['bg_mean_band_error_pct']:.2f}%",
            f"{r['sz_events_per_sec']:.1f}",
            f"{r['sz_mean_band_error_pct']:.2f}%",
            f"{r['sz_delta_err_pct']:.2f}%",
            f"{r['sz_beta_err_pct']:.2f}%",
        ))
    print("=" * 105)

    # Detailed comparative analysis
    print("\nCOMPARATIVE ANALYSIS (Fixed vs Adaptive):")
    for val in SWEEP_VALUES:
        f_row = df[(df["method"] == "Fixed") & (df["param"] == val)].iloc[0]
        a_row = df[(df["method"] == "Adaptive") & (df["param"] == val)].iloc[0]
        bg_rate_diff = a_row["bg_events_per_sec"] - f_row["bg_events_per_sec"]
        sz_rate_diff = a_row["sz_events_per_sec"] - f_row["sz_events_per_sec"]
        sz_err_diff = a_row["sz_mean_band_error_pct"] - f_row["sz_mean_band_error_pct"]
        print(
            f"Param={val:.2f}: "
            f"BG Rate: Adapt {a_row['bg_events_per_sec']:.1f} vs Fixed {f_row['bg_events_per_sec']:.1f} ({bg_rate_diff:+.1f} ev/s) | "
            f"SZ Rate: Adapt {a_row['sz_events_per_sec']:.1f} vs Fixed {f_row['sz_events_per_sec']:.1f} ({sz_rate_diff:+.1f} ev/s) | "
            f"SZ Band Err: Adapt {a_row['sz_mean_band_error_pct']:.2f}% vs Fixed {f_row['sz_mean_band_error_pct']:.2f}% ({sz_err_diff:+.2f}%)"
        )
    print("=" * 105)


def main():
    signal, fs = load_dataset()
    df = run_sweep(signal, fs)
    generate_figures(signal, fs, df)
    print_summary_table(df)


if __name__ == "__main__":
    main()
