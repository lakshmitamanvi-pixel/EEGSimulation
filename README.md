# eeg-event-tx: Event-Triggered EEG Transmission (Fixed vs Adaptive Threshold)

This project investigates **event-triggered (send-on-delta) transmission** for power-constrained wearable and implantable EEG monitoring systems. Instead of continuously streaming samples at a fixed sampling frequency ($f_s = 256$ Hz), an event is transmitted only when the signal deviation $|x[n] - \text{last}| \ge \Delta$. The receiver reconstructs the signal via Zero-Order Hold (ZOH).

We compare a **Fixed Threshold** policy against an **Adaptive Threshold** policy that scales dynamically with signal volatility using strictly past reconstructed samples.

---

## Key Features & Protocol Design

### 1. Fixed Threshold Encoder
$$\Delta = c \cdot \sigma_x$$
- Fixed threshold scaled by nominal signal standard deviation $\sigma_x = \text{std}(x)$.
- Transmitter sends an event whenever $|x[n] - \text{last}| \ge \Delta$.
- Receiver updates its hold register: $\hat{x}[n] = \text{last}$.

### 2. Adaptive Threshold Encoder
$$\Delta[n] = \max\left(k \cdot \text{std}\left(\hat{x}[n-f_s : n]\right),\, 0.02 \cdot \sigma_x\right)$$
- Evaluated **strictly on past reconstructed samples** $\hat{x}[n-f_s:n-1]$ over a 1-second causal sliding window.
- **Zero look-ahead & zero side-information**: The receiver tracks the identical threshold locally from its own reconstruction stream without requiring dedicated metadata packets.
- A noise-floor clamp ($0.02 \cdot \sigma_x$) prevents threshold collapse during quiescent EEG periods.

### 3. Frozen Dataset & Annotations
- **Source**: PhysioNet CHB-MIT Scalp EEG Database (`chb01_03.edf`, patient `chb01`, Channel 0 `FP1-F7`).
- **Window**: 2700–3300 s (600 s total, 153,600 samples at 256 Hz).
- **Seizure Slice**: Window seconds 296–336 s (file time 2996–3036 s, verified against `chb01-summary.txt`).
- **Background Slice**: Window seconds 0–250 s.
- **Parameter Sweep**: Multipliers $\in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]$.
- **Spectral Bands**:
  - Delta: 0.5 – 4 Hz
  - Theta: 4 – 8 Hz
  - Alpha: 8 – 13 Hz
  - Beta: 13 – 30 Hz
- **Welch Spectral Estimation**: `nperseg = 512`.

---

## Repository Structure

```text
eeg-event-tx/
├── assets/
│   └── segment.npz           # Processed 600s EEG segment (committed for offline use)
├── figures/
│   ├── pareto_tradeoff.png      # Rate-distortion Pareto curves (Background vs Seizure)
│   ├── seizure_reconstruction.png # 10s raw vs reconstructed seizure snippet (k=0.2)
│   └── band_errors_k02.png      # Per-band relative error bar chart (k=0.2)
├── results/
│   └── results.csv           # Full experimental sweep dataset
├── app.py                    # Streamlit interactive application
├── config.py                 # Frozen constants and directory configurations
├── data.py                   # Ingestion, annotation verification, MNE preprocessing
├── encoders.py               # Fixed and Adaptive send-on-delta encoders
├── metrics.py                # MSE, Welch band power, and relative error calculations
├── run.py                    # Sweep runner, figure generator, and table reporting
├── tests.py                  # Encoder sanity validation suite
├── requirements.txt          # Pinned clean dependencies
└── README.md                 # Project documentation
```

---

## Installation & Usage

### 1. Environment Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Sanity Tests
The unit tests verify:
1. Constant signal produces exactly 0 events after the initial reference sample.
2. For any fixed delta, the absolute error $\max |x - \hat{x}| < \Delta + 10^{-9}$ across all samples.
3. Monotonicity: smaller threshold never decreases the total event count.

```bash
python tests.py
```

### 3. Ingest Data (Optional if `assets/segment.npz` is present)
```bash
python data.py
```
*Note: `assets/segment.npz` is committed directly to the repository so the pipeline and app can run fully offline without re-downloading.*

### 4. Run the Full Experiment
Executes the parameter sweep across both encoders, generates 300 DPI figures in `figures/`, writes metrics to `results/results.csv`, and outputs an unvarnished summary table:
```bash
python run.py
```

### 5. Launch the Streamlit App
Runs offline using only `assets/segment.npz`:
```bash
streamlit run app.py
```

---

## Assumptions & Engineering Limitations

1. **Cohort & Channel Scope**: Evaluated on 1 patient (`chb01`), 1 recording (`chb01_03.edf`), and 1 bipolar channel (`FP1-F7`). Inter-subject variability and multi-channel spatial correlations are not captured.
2. **Energy Proxy**: Transmission energy is modeled as proportional to transmitted events ($E \propto \text{events} \times \text{bits}$). In real radio transceivers (e.g., Bluetooth Low Energy or IEEE 802.15.4), per-packet overhead, preamble energy, state transition penalties, and packet framing affect effective energy savings.
3. **ZOH Spectral Distortion**: Zero-Order Hold produces discontinuous step transitions in the reconstructed waveform $\hat{x}(t)$. In the frequency domain, step discontinuities introduce high-frequency sinc sidelobes that artificially elevate reconstructed power in higher bands, particularly the **Beta band (13–30 Hz)**. Linear or spline interpolation at the receiver would attenuate this effect at the cost of additional compute.
