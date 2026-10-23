# 📡 BurstPulse — Network Traffic Burst Analysis Using Poisson Distribution
**Applied Mathematics Micro Project — Scientific Anomaly Detection & Unified Traffic Analyzer**

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![SciPy](https://img.shields.io/badge/scipy-1.11+-8CAAE6.svg)](https://scipy.org/)
[![Plotly](https://img.shields.io/badge/plotly-5.20+-3F4F75.svg)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/pytest-52%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

**BurstPulse** is an interactive, scientific network traffic analysis platform and statistical anomaly detection engine. It models discrete network packet arrival queues as a Poisson arrival process and classifies statistical bursts in real time.

Built for university academic evaluation and research demonstrations, the application provides:
1. **Mathematical Rigor:** Typeset mathematical formulas rendered natively with KaTeX (`st.latex`), numerical stability via SciPy's survival function `scipy.stats.poisson.sf`, and zero baseline data leakage.
2. **5 Built-in Traffic Scenarios:** Ready-to-analyze traffic captures and reproducible synthetic simulations with truthful provenance disclosures.
3. **Multi-Schema CSV Ingestion:** Support for user-provided interval counts, raw packet timestamps, and aggregated flow logs.
4. **Interactive Visualizations:** Responsive Plotly time series with calibration window shading, baseline arrival rate lines, detected burst markers, and animated antenna status.
5. **Reproducible Test Coverage:** 52 comprehensive unit and integration tests passing across macOS and Windows.

---

## 🧮 Mathematical Foundation

### 1. Poisson Arrival Process (PMF)
Packet arrivals over discrete, fixed observation intervals (e.g., 1-second bins) are modeled using the Poisson Probability Mass Function (PMF):

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k \in \{0, 1, 2, \dots\}$$

- **$X$:** Discrete random variable representing packet arrivals in a fixed observation interval.
- **$k$:** Observed nonnegative integer packet count ($k \ge 0$).
- **$\lambda$:** Expected arrival rate parameter ($\mathbb{E}[X] = \operatorname{Var}(X) = \lambda$).

### 2. Baseline Rate Estimation ($\hat{\lambda}$)
Given an initial baseline partition containing $m$ observations $x_1, x_2, \dots, x_m$:

$$\hat{\lambda} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

- **Zero Contamination:** The baseline arrival rate $\hat{\lambda}$ is estimated **strictly from the baseline partition** (e.g., initial 30% to 90% of intervals) and completely isolated from the evaluation segment.
- **Validation:** Datasets with non-positive baseline rates ($\hat{\lambda} \le 0$) are rejected to prevent degenerate models.

### 3. Upper-Tail Cumulative Survival Probability
Rather than assessing point probability $P(X = k)$ (which naturally decays for any single count), burst detection evaluates the **cumulative probability of observing a count at least as large as $k$**:

$$p = P(X \ge k) = 1 - P(X \le k - 1) = \sum_{j=k}^{\infty} \frac{\lambda^j e^{-\lambda}}{j!}$$

Computed stably without numerical overflow via SciPy's survival function:
```python
import scipy.stats as stats

# k: observed count, lambda_hat: estimated baseline mean
p_value = stats.poisson.sf(k - 1, lambda_hat)
```

### 4. Classification Decision Rule
Given a user-configured significance threshold $\alpha \in \{0.001, 0.005, 0.01, 0.025, 0.05\}$:
- **🚨 Potential Burst:** If $P(X \ge k) < \alpha$ (arrival volume is statistically extreme under the fitted baseline $\hat{\lambda}$).
- **✅ Normal Variation:** If $P(X \ge k) \ge \alpha$ (count is consistent with ordinary Poisson random fluctuations).

### 5. Verified Reference Example ($\lambda = 20$)
Under a calibrated baseline with $\lambda = 20$ evaluated at $\alpha = 0.01$:
- **At $k = 30$ (Normal Fluctuation):**
  - Point PMF: $P(X = 30) \approx 0.00834$ (0.83%)
  - Upper-Tail Survival: $P(X \ge 30) = \text{poisson.sf}(29, 20) \approx 0.02182$ (2.18%)
  - Decision: $0.02182 \ge 0.01 \implies$ **NOT FLAGGED** (within 1% threshold).
- **At $k = 48$ (Volumetric Burst):**
  - Point PMF: $P(X = 48) \approx 7.15 \times 10^{-8}$
  - Upper-Tail Survival: $P(X \ge 48) = \text{poisson.sf}(47, 20) \approx 7.83 \times 10^{-8} \ll 0.01 \implies$ 🚨 **FLAGGED AS A BURST**.

---

## 🗂️ Built-in Datasets

The repository includes five verified, locally bundled datasets in `data/` that require zero external downloads:

| Dataset Name | Observations | Characteristics | Description |
| :--- | :--- | :--- | :--- |
| **Normal Network Traffic** | 250 intervals | $\lambda \approx 20, D \approx 1.05$ | Stable Poisson traffic representing ordinary background network activity. |
| **Network Traffic with Bursts** | 250 intervals | Baseline + 4 Spikes | Calibrated baseline with 4 injected burst spikes strictly after the 90% partition. |
| **Port Scan Simulation** | 250 intervals | Baseline + 3 Surges | Simulated TCP SYN port reconnaissance activity with ground-truth labels. |
| **DDoS-Style Traffic Surge** | 300 intervals | $10\times$ Volume Flood | Multi-interval sustained volumetric surge simulating denial-of-service conditions. |
| **Overdispersed Traffic** | 250 intervals | Negative Binomial ($D \approx 11.6$) | Exhibits heavy clustering and variance exceeding the mean ($\operatorname{Var}(X) > \mathbb{E}[X]$). |

---

## 📁 Repository Structure

```
burstpulse/
├── app.py                      # Main application controller & single-screen analyzer dashboard
├── landing_page.py             # Product landing page with typeset mathematical foundation & gallery
├── antenna_component.py        # Isolated iframe-based radio antenna signal animation
├── burst_detector.py           # Core Poisson mathematics, baseline estimator & KPI engine
├── dataset_registry.py         # Metadata catalog for the 5 built-in datasets
├── case_studies.py             # Historical network threat reference notes (CAIDA, Mirai, CIC-IDS)
├── data_preprocessor.py        # Ingestion pipeline for timestamps, flow logs & interval counts
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules for Python, virtual environments, and caches
├── data/                       # Bundled CSV datasets (reproducible & verified)
│   ├── normal_network_traffic.csv
│   ├── network_traffic_with_bursts.csv
│   ├── port_scan_simulation.csv
│   ├── ddos_style_traffic_surge.csv
│   └── overdispersed_network_traffic.csv
└── tests/                      # Automated test suite (52 tests)
    ├── conftest.py
    ├── test_app_integration.py
    ├── test_burst_detector.py
    ├── test_data_preprocessor.py
    ├── test_dataset_registry.py
    └── test_poisson_logic.py
```

---

## 🚀 Setup & Execution Guide

Follow the instructions below for your operating system.

### 🍎 macOS Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/jeedijoshua-art/burstpulse.git
cd burstpulse
```

#### 2. Create and Activate Virtual Environment
```bash
# Ensure Python 3.10+ is installed
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### 5. Run the Automated Tests
```bash
pytest -v
```

---

### 🪟 Windows Setup (Command Prompt or PowerShell)

#### 1. Clone the Repository
```cmd
git clone https://github.com/jeedijoshua-art/burstpulse.git
cd burstpulse
```

#### 2. Create and Activate Virtual Environment

**Using Command Prompt (`cmd.exe`):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Using PowerShell (`powershell.exe`):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
*(Note: If PowerShell throws an execution policy error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` once).*

#### 3. Install Dependencies
```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the Application
```cmd
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

#### 5. Run the Automated Tests
```cmd
pytest -v
```

---

## 🧪 Testing & Verification

The test suite validates every layer of the platform, including:
- **Statistical Invariants:** Baseline isolation, zero data leakage, and exact Poisson survival function bounds.
- **Data Validation:** Non-negative integer counts, schema detection, flow aggregation, and timestamp binning.
- **UI & Routing:** Streamlit `AppTest` coverage for bidirectional page navigation, query parameters (`?page=analyzer&dataset=...`), and native KaTeX `st.latex` formula rendering.

To execute tests with detailed output:
```bash
pytest -v
```
Expected output:
```
============================== 52 passed in 3.80s ==============================
```

---

## 💡 Usage Tips for Teammates

- **Navigation:** Start on the **Landing Page** to review the mathematical foundation and scenario overview, or jump directly to the **Analyzer** via the `🚀 Open Analyzer` button.
- **Direct Dataset Links:** You can link directly to specific datasets using query parameters, e.g.:
  - `http://localhost:8501/?page=analyzer&dataset=port_scan_simulation`
  - `http://localhost:8501/?page=analyzer&dataset=ddos_style_traffic_surge`
- **Custom CSV Upload:** Switch the radio selector to **Upload Your Own CSV** to test your own packet captures. Supported schemas include interval counts (`interval,packet_count`), timestamped packet logs (1 row per packet), and network flow logs.
- **Exporting Results:** Click **⬇️ Download Complete Analysis CSV** at the bottom of the analyzer to export intervals, counts, baseline $\hat{\lambda}$, $p$-values, and burst status flags.

---

## 📄 License & Attribution

Developed as an Applied Mathematics Micro Project.  
Code and documentation distributed under the [MIT License](LICENSE).  
Repository maintained by [jeedijoshua-art](https://github.com/jeedijoshua-art).
