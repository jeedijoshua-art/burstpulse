#!/usr/bin/env python3
"""
BurstPulse synthetic dataset pack generator (v1.0.0)

Generates five SYNTHETIC, research-inspired interval-count datasets for testing
BurstPulse (Poisson upper-tail burst detection). Nothing here is real captured
traffic and nothing originates from CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15,
BoT-IoT or TON_IoT. Folder names only indicate the schema/scenario *style*.

Usage:  python generate_datasets.py [output_dir]
Deterministic: fixed seeds per dataset (numpy default_rng / PCG64).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

GENERATOR_VERSION = "1.0.0"
N_ROWS = 10_000
BASELINE_ROWS = 3_000          # first 3,000 intervals (30%, the BurstPulse app's minimum baseline slider): clean baseline (no attacks / no injected events)
ALPHA = 1e-3                   # demo significance level used only for the support check below

DISCLAIMER = (
    "SYNTHETIC DATA: generated programmatically for software development, demos, statistical "
    "testing and education. Not genuine captured traffic and not a copy of any named research "
    "dataset. Attack labels describe simulated scenarios only; a statistical burst detection is "
    "not proof of an attack."
)

COLUMNS = [
    "interval", "timestamp", "packet_count", "flow_count", "measurement_unit",
    "has_attack_label", "attack_type", "scenario", "scenario_label", "period",
    "source_dataset_style", "is_synthetic",
]

COLUMN_DOCS = {
    "interval": "1-based consecutive interval index; unique per dataset (one row = one fixed-length time interval).",
    "timestamp": "ISO 8601 start time of the interval, UTC (+00:00).",
    "packet_count": "Non-negative integer: packets observed in the interval.",
    "flow_count": "Non-negative integer: flows started in the interval (never exceeds packet_count).",
    "measurement_unit": "Explicit unit of packet_count and flow_count, e.g. packets_and_flows_per_1s_interval.",
    "has_attack_label": "True when the interval belongs to a simulated attack-like scenario (attack_type != BENIGN).",
    "attack_type": "BENIGN or a SIM_* simulated scenario name.",
    "scenario": "Traffic scenario category for filtering/analysis (benign high-volume, recovery, overdispersed, etc.).",
    "scenario_label": "Exact copy of 'scenario'; kept because the BurstPulse app displays a column of this name.",
    "period": "'baseline' (first 3,000 clean intervals = 30%, for estimating lambda) or 'evaluation'.",
    "source_dataset_style": "Research-inspired schema/scenario style this synthetic file imitates (NOT its origin).",
    "is_synthetic": "Always True.",
}


# ----------------------------------------------------------------------------- helpers
def _shape(kind, length):
    if kind == "step":
        return np.ones(length)
    if kind == "tri":      # gradual rise then gradual fall
        h = length // 2
        return np.concatenate([np.linspace(0, 1, h, endpoint=False), np.linspace(1, 0, length - h)])
    if kind == "ramp":     # gradual rise, abrupt end
        return np.linspace(0.05, 1, length)
    if kind == "decay":    # exponential recovery tail
        return np.exp(-np.arange(length) / (length / 4.0))
    raise ValueError(kind)


def E(start, length, kind, mult, scenario, attack=None, phi=0.0, rmult=1.0):
    """Injected event: multiplies the local base rate by `mult` (shaped), flow ratio by `rmult`."""
    return dict(start=start, length=length, kind=kind, mult=mult, scenario=scenario,
                attack=attack, phi=phi, rmult=rmult)


def _wander(rng, n, sd=0.05, rho=0.995):
    w = np.zeros(n)
    inn = rng.normal(0, sd * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        w[i] = rho * w[i - 1] + inn[i]
    return np.exp(w)


# ----------------------------------------------------------------------------- dataset specs
def spec_cic2017(rng):
    n = N_ROWS
    lam = np.full(n, 60.0)
    scen = np.full(n, "stable_benign", dtype=object)
    ev = [
        E(3200, 900, "tri", 1.6, "gradual_variation"),
        E(3050, 40, "step", 3.0, "attack_window", "SIM_DDoS_Window", rmult=1.5),
        E(4100, 10, "step", 5.0, "attack_burst", "SIM_DoS_Burst", rmult=0.6),
        E(4600, 12, "step", 4.5, "attack_burst", "SIM_DoS_Burst", rmult=0.6),
        E(5200, 60, "step", 2.2, "benign_high_volume"),
        E(5900, 150, "step", 2.0, "attack_window", "SIM_BruteForce_Window", rmult=2.5),
        E(6700, 8, "step", 5.5, "attack_burst", "SIM_DoS_Burst", rmult=0.6),
        E(7600, 45, "step", 2.6, "benign_high_volume"),
        E(8300, 200, "step", 1.7, "attack_window", "SIM_PortScan_Window", rmult=4.0),
        E(9100, 15, "step", 4.0, "attack_burst", "SIM_DoS_Burst", rmult=0.6),
        E(9500, 30, "step", 3.2, "benign_high_volume"),
    ]
    return dict(lam=lam, scen=scen, phi=np.zeros(n), ratio=0.10, events=ev,
                step=1, start="2026-01-05T08:00:00+00:00", seed=20170001,
                unit="packets_and_flows_per_1s_interval",
                style="CIC-IDS2017-style", display="CIC-IDS2017-style synthetic benchmark",
                method="Poisson packet counts around a stable lambda=60 baseline, shaped multiplicative events, "
                       "flows ~ Binomial(packets, flow_ratio).")


def spec_cse2018(rng):
    n = N_ROWS
    lam = np.empty(n)
    scen = np.empty(n, dtype=object)
    phi = np.zeros(n)
    lam[:3000], scen[:3000] = 120.0, "regime_a_baseline"
    lam[3000:5000], scen[3000:5000] = 200.0, "regime_b_business_hours"
    lam[5000:7000], scen[5000:7000] = 90.0, "regime_c_night"
    lam[7000:], scen[7000:] = 150.0, "overdispersed_background"
    phi[7000:] = 0.15
    ev = [
        E(3100, 150, "step", 0.4, "quiet_interval"),
        E(3300, 20, "step", 4.0, "attack_surge", "SIM_DoS_Surge", rmult=0.7),
        E(3900, 120, "tri", 1.5, "gradual_variation"),
        E(5200, 120, "step", 0.35, "quiet_interval"),
        E(5400, 400, "step", 2.5, "attack_sustained", "SIM_DDoS_Sustained", rmult=1.4),
        E(5800, 25, "decay", 3.0, "recovery"),
        E(6300, 60, "step", 2.8, "benign_high_volume"),
        E(7400, 100, "step", 0.4, "quiet_interval"),
        E(7600, 150, "step", 2.0, "attack_window", "SIM_Botnet_Window", phi=0.15, rmult=1.8),
        E(8600, 25, "step", 4.5, "attack_burst", "SIM_BruteForce_Burst", rmult=2.0),
        E(9000, 80, "step", 2.2, "benign_high_volume", phi=0.15),
        E(9400, 60, "step", 3.0, "attack_window", "SIM_WebAttack_Window", rmult=1.6),
    ]
    return dict(lam=lam, scen=scen, phi=phi, ratio=0.09, events=ev,
                step=5, start="2026-02-12T00:00:00+00:00", seed=20180002,
                unit="packets_and_flows_per_5s_interval",
                style="CSE-CIC-IDS2018-style", display="CSE-CIC-IDS2018-style synthetic benchmark",
                method="Four background regimes (lambda 120/200/90/150), last regime negative-binomial "
                       "(gamma-Poisson, phi=0.15); 5-second intervals (~13.9 h); abrupt + sustained surges, quiet gaps.")


def spec_unsw(rng):
    n = N_ROWS
    t = np.arange(n)
    lam = 80.0 * (1 + 0.15 * np.sin(2 * np.pi * t / 1800.0)) * _wander(rng, n, 0.05)
    scen = np.full(n, "varying_normal", dtype=object)
    ev = [
        E(3000, 150, "step", 1.3, "recon_increase", "SIM_Recon_Scan", rmult=6.0),
        E(3500, 8, "step", 5.0, "attack_burst", "SIM_Fuzzers_Burst", rmult=0.8),
        E(4000, 300, "ramp", 2.5, "gradual_anomaly", "SIM_DoS_Gradual", rmult=1.2),
        E(4900, 80, "step", 2.0, "benign_high_volume"),
        E(5100, 60, "step", 1.4, "recon_increase", "SIM_Recon_Scan", rmult=7.0),   # overlaps the benign period
        E(6000, 10, "step", 4.5, "attack_burst", "SIM_Exploit_Burst", rmult=0.9),
        E(6500, 120, "step", 1.25, "recon_increase", "SIM_Recon_Scan", rmult=5.0),
        E(7400, 50, "step", 2.2, "benign_high_volume"),
        E(7800, 200, "ramp", 2.0, "gradual_anomaly", "SIM_Backdoor_Gradual", rmult=1.5),
        E(8800, 12, "step", 4.0, "attack_burst", "SIM_Exploit_Burst", rmult=0.9),
        E(9300, 40, "step", 1.9, "benign_high_volume"),
    ]
    return dict(lam=lam, scen=scen, phi=np.zeros(n), ratio=0.12, events=ev,
                step=1, start="2026-03-20T06:00:00+00:00", seed=20150003,
                unit="packets_and_flows_per_1s_interval",
                style="UNSW-NB15-style", display="UNSW-NB15-style synthetic benchmark",
                method="Slowly varying normal rate (sinusoid + AR(1) wander); recon events inflate flow ratio far more "
                       "than packet counts; ramps for gradual anomalies; partial overlap of recon with benign high volume.")


def spec_botiot(rng):
    n = N_ROWS
    t = np.arange(n)
    bump = (t % 60) < 3                                  # periodic IoT telemetry burst every 60 s
    lam = np.where(bump, 30.0, 12.0)
    scen = np.where(bump, "periodic_iot_activity", "iot_baseline").astype(object)
    ev = [
        E(3000, 100, "step", 25.0, "attack_flood", "SIM_UDP_Flood", rmult=0.5),
        E(3100, 40, "decay", 5.0, "recovery"),
        E(4200, 500, "step", 12.0, "attack_sustained", "SIM_TCP_Flood_Sustained", rmult=0.8),
        E(4700, 60, "decay", 4.0, "recovery"),
        E(5600, 10, "step", 30.0, "attack_burst", "SIM_HTTP_Flood_Burst", rmult=0.4),
        E(5610, 40, "decay", 5.0, "recovery"),
        E(6500, 200, "step", 6.0, "attack_flood", "SIM_Service_Scan_Flood", rmult=3.0),
        E(6700, 40, "decay", 3.0, "recovery"),
        E(7500, 8, "step", 28.0, "attack_burst", "SIM_UDP_Flood", rmult=0.5),
        E(7508, 40, "decay", 5.0, "recovery"),
        E(8400, 600, "step", 15.0, "attack_sustained", "SIM_TCP_Flood_Sustained", rmult=0.8),
        E(9000, 60, "decay", 4.0, "recovery"),
        E(9300, 50, "step", 3.5, "benign_high_volume"),
    ]
    return dict(lam=lam, scen=scen, phi=np.zeros(n), ratio=0.15, events=ev,
                step=1, start="2026-04-08T00:00:00+00:00", seed=20180004,
                unit="packets_and_flows_per_1s_interval",
                style="BoT-IoT-style", display="BoT-IoT-style synthetic benchmark",
                method="Low IoT baseline (lambda 12) with a 3-second telemetry burst every 60 s (lambda 30), strong "
                       "multiplicative flood events (x6..x30) followed by exponential recovery tails.")


def spec_toniot(rng):
    n = N_ROWS
    t = np.arange(n)
    sensor = 1 + 0.20 * np.sin(2 * np.pi * t / 300.0)
    base = np.empty(n)
    scen = np.empty(n, dtype=object)
    base[:3000], scen[:3000] = 35.0, "device_regime_a"
    base[3000:4500], scen[3000:4500] = 55.0, "device_regime_b"
    base[4500:6000], scen[4500:6000] = 35.0, "device_regime_a"
    base[6000:7500], scen[6000:7500] = 20.0, "device_regime_c_idle"
    base[7500:], scen[7500:] = 35.0, "device_regime_a"
    lam = base * sensor
    phi = np.zeros(n)
    phi[8200:9200] = 0.30
    scen[8200:9200] = "overdispersed_device_noise"
    ev = [E(5000 + 40 * k, 6, "step", 4.0, "attack_intermittent", "SIM_DDoS_Intermittent", rmult=0.7) for k in range(6)]
    ev += [
        E(3200, 30, "step", 1.7, "benign_fluctuation"),
        E(3800, 120, "step", 2.3, "attack_window", "SIM_Backdoor_Window", rmult=1.5),
        E(4300, 25, "step", 1.8, "benign_fluctuation"),
        E(6300, 100, "step", 3.0, "attack_window", "SIM_Scan_Window", rmult=4.0),
        E(7000, 20, "step", 1.9, "benign_fluctuation"),
        E(7800, 12, "step", 5.0, "attack_burst", "SIM_Injection_Burst", rmult=0.8),
        E(8500, 40, "step", 1.7, "benign_fluctuation", phi=0.30),
        E(9400, 150, "step", 2.2, "attack_window", "SIM_Ransomware_Traffic_Window", rmult=1.3),
    ]
    return dict(lam=lam, scen=scen, phi=phi, ratio=0.11, events=ev,
                step=2, start="2026-05-14T00:00:00+00:00", seed=20190005,
                unit="packets_and_flows_per_2s_interval",
                style="TON_IoT-style", display="TON_IoT-style synthetic benchmark",
                method="Sensor-like sinusoidal device cycle (period 300 intervals, +/-20%), three traffic-rate regimes, "
                       "intermittent surges (6 x 6-interval bursts every 40), gamma-Poisson noise segment (phi=0.30); 2-second intervals.")


SPECS = {
    "CIC-IDS2017": spec_cic2017,
    "CSE-CIC-IDS2018": spec_cse2018,
    "UNSW-NB15": spec_unsw,
    "BoT-IoT": spec_botiot,
    "TON_IoT": spec_toniot,
}


# ----------------------------------------------------------------------------- builder
def build_dataset(folder):
    n = N_ROWS
    # seed is looked up from a throwaway spec call; real rng is created per dataset below
    seed = SPECS[folder](np.random.default_rng(0))["seed"]
    rng = np.random.default_rng(seed)
    sp = SPECS[folder](rng)

    base_lam = np.asarray(sp["lam"], dtype=float)
    lam = base_lam.copy()
    ratio = np.full(n, sp["ratio"])
    phi = np.asarray(sp["phi"], dtype=float).copy()
    scen = np.asarray(sp["scen"], dtype=object).copy()
    atk = np.full(n, "BENIGN", dtype=object)

    for ev in sp["events"]:
        s, L = ev["start"], ev["length"]
        assert s >= BASELINE_ROWS and s + L <= n, f"{folder}: event outside evaluation period {ev}"
        sl = slice(s, s + L)
        sh = _shape(ev["kind"], L)
        lam[sl] = base_lam[sl] * (1 + (ev["mult"] - 1) * sh)
        ratio[sl] = sp["ratio"] * (1 + (ev["rmult"] - 1) * sh)
        if ev["phi"] > 0:
            phi[sl] = ev["phi"]
        scen[sl] = ev["scenario"]
        atk[sl] = ev["attack"] if ev["attack"] else "BENIGN"

    # Poisson, or gamma-Poisson (negative binomial) where phi>0: Var = mu + phi*mu^2
    g = np.ones(n)
    od = phi > 0
    g[od] = rng.gamma(1.0 / phi[od], phi[od])
    packets = rng.poisson(lam * g).astype(np.int64)
    p_flow = np.clip(ratio, 0.02, 0.9)
    flows = rng.binomial(packets, p_flow).astype(np.int64)
    flows = np.where(packets > 0, np.maximum(flows, 1), 0).astype(np.int64)

    ts = pd.date_range(sp["start"], periods=n, freq=f"{sp['step']}s")
    df = pd.DataFrame({
        "interval": np.arange(1, n + 1),
        "timestamp": [x.isoformat() for x in ts],
        "packet_count": packets,
        "flow_count": flows,
        "measurement_unit": sp["unit"],
        "has_attack_label": atk != "BENIGN",
        "attack_type": atk,
        "scenario": scen,
        "scenario_label": scen,
        "period": np.where(np.arange(n) < BASELINE_ROWS, "baseline", "evaluation"),
        "source_dataset_style": sp["style"],
        "is_synthetic": True,
    })[COLUMNS]
    return df, sp


def generate_all_frames():
    return {f: build_dataset(f) for f in SPECS}


def csv_bytes(df):
    return df.to_csv(index=False, lineterminator="\n").encode("utf-8")


# ----------------------------------------------------------------------------- statistics
def _moments(x):
    m, v = float(np.mean(x)), float(np.var(x, ddof=1))
    return dict(mean=round(m, 4), variance=round(v, 4),
                dispersion_index=round(v / m, 4) if m > 0 else None,
                min=int(np.min(x)), max=int(np.max(x)))


def _runs(mask):
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def poisson_support_check(df):
    """Independent check of how well the data supports the intended demo: upper-tail test with
    lambda estimated on the baseline period only. Reported, never guaranteed."""
    base = df[df.period == "baseline"]
    ev = df[df.period == "evaluation"].reset_index(drop=True)
    out = {"alpha": ALPHA, "test": "P(X >= k | Poisson(lambda_hat)); lambda_hat = baseline mean"}
    for col in ("packet_count", "flow_count"):
        lam0 = float(base[col].mean())
        p = poisson.sf(ev[col].to_numpy() - 1, lam0)
        flag = p < ALPHA
        atk = ev.has_attack_label.to_numpy()
        stable = (ev.scenario.isin(["stable_benign", "iot_baseline", "varying_normal", "periodic_iot_activity",
                                    "regime_a_baseline", "device_regime_a"])).to_numpy() & ~atk
        hv = ev.scenario.isin(["benign_high_volume", "benign_fluctuation"]).to_numpy()
        runs = _runs(atk)
        det = [bool(flag[a:b + 1].any()) for a, b in runs]
        res = {
            "baseline_lambda_hat": round(lam0, 4),
            "attack_interval_recall": round(float(flag[atk].mean()), 4) if atk.any() else None,
            "attack_interval_precision_among_flagged": round(float(atk[flag].mean()), 4) if flag.any() else None,
            "attack_events": len(runs),
            "attack_events_with_any_flag": int(sum(det)),
            "flags_total": int(flag.sum()),
            "stable_benign_false_positive_rate": round(float(flag[stable].mean()), 4) if stable.any() else None,
            "stable_benign_intervals": int(stable.sum()),
            "benign_high_volume_flag_rate": round(float(flag[hv].mean()), 4) if hv.any() else None,
            "benign_high_volume_intervals": int(hv.sum()),
            "flag_rate_by_scenario": {s: round(float(flag[(ev.scenario == s).to_numpy()].mean()), 4)
                                      for s in sorted(ev.scenario.unique())},
        }
        out[col] = res
    return out


def dataset_info(folder, df, sp):
    base = df[df.period == "baseline"]
    atk = df.has_attack_label
    od_rows = df[df.scenario.str.startswith("overdispersed")]
    od = None
    if len(od_rows) > 50:
        od = {"scenario_rows": int(len(od_rows)),
              "packet_count_dispersion_index": _moments(od_rows.packet_count)["dispersion_index"],
              "note": "variance/mean >> 1 within a constant-base-rate segment (gamma-Poisson mixing)"}
    events = [{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in e.items()}
              for e in sp["events"]]
    return {
        "dataset_folder": folder,
        "display_name": sp["display"],
        "source_dataset_style": sp["style"],
        "is_synthetic": True,
        "disclaimer": DISCLAIMER,
        "generator_version": GENERATOR_VERSION,
        "generation_method": sp["method"],
        "random_seed": sp["seed"],
        "rng": "numpy.random.default_rng (PCG64)",
        "row_count": int(len(df)),
        "interval_seconds": sp["step"],
        "measurement_unit": sp["unit"],
        "row_semantics": f"each row = one {sp['step']}-second interval; packet_count/flow_count are counts within it",
        "timestamp_start": df.timestamp.iloc[0],
        "timestamp_end": df.timestamp.iloc[-1],
        "timezone": "UTC (+00:00)",
        "baseline_period": {"first_interval": 1, "last_interval": BASELINE_ROWS, "rows": BASELINE_ROWS,
                            "contains_attack_labels": bool(base.has_attack_label.any()),
                            "timestamp_end": base.timestamp.iloc[-1]},
        "evaluation_period": {"first_interval": BASELINE_ROWS + 1, "last_interval": len(df),
                              "rows": int(len(df) - BASELINE_ROWS)},
        "packet_count_stats": _moments(df.packet_count),
        "flow_count_stats": _moments(df.flow_count),
        "baseline_packet_count_stats": _moments(base.packet_count),
        "baseline_flow_count_stats": _moments(base.flow_count),
        "attack_label_intervals": int(atk.sum()),
        "attack_label_percentage": round(float(atk.mean() * 100), 3),
        "scenario_counts": {k: int(v) for k, v in df.scenario.value_counts().sort_index().items()},
        "attack_type_counts": {k: int(v) for k, v in df.attack_type.value_counts().sort_index().items()},
        "overdispersed_scenario_check": od,
        "injected_events": events,
        "poisson_demo_support_check": poisson_support_check(df),
        "dispersion_note": ("Whole-file dispersion index is inflated by deliberate non-stationarity (regimes, events). "
                            "Baseline dispersion is ~1 where the base rate is constant (CIC-IDS2017-, CSE-CIC-IDS2018-style) and "
                            ">1 where the baseline deliberately includes smooth variation or periodic activity "
                            "(UNSW-NB15-, BoT-IoT-, TON_IoT-style), so a constant-lambda Poisson baseline is only approximate there."),
        "label_semantics": "attack labels describe the simulated scenario; a Poisson detection is not proof of an attack.",
        "columns": COLUMN_DOCS,
    }


# ----------------------------------------------------------------------------- writer
def write_all(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    catalog = []
    for folder, (df, sp) in generate_all_frames().items():
        d = root / folder
        d.mkdir(exist_ok=True)
        (d / "network_traffic.csv").write_bytes(csv_bytes(df))
        info = dataset_info(folder, df, sp)
        (d / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        catalog.append({
            "dataset_folder": folder,
            "display_name": sp["display"],
            "file_path": f"{folder}/network_traffic.csv",
            "row_count": len(df),
            "timestamp_start": info["timestamp_start"],
            "timestamp_end": info["timestamp_end"],
            "measurement_unit": sp["unit"],
            "scenarios_included": ";".join(sorted(df.scenario.unique())),
            "random_seed": sp["seed"],
            "is_synthetic": True,
            "provenance_description": (f"Synthetic {sp['style']} test dataset generated by generate_datasets.py "
                                       f"v{GENERATOR_VERSION}; not genuine captured data."),
        })
    pd.DataFrame(catalog).to_csv(root / "DATASET_CATALOG.csv", index=False, lineterminator="\n")

    prov = {
        "generator": "generate_datasets.py", "generator_version": GENERATOR_VERSION,
        "is_synthetic": True,
        "statements": [
            "All five datasets were generated programmatically with fixed random seeds.",
            "They are NOT genuine downloaded research datasets.",
            "Folder names indicate research-inspired test profiles only.",
            "No claim is made that any record originated from the named institutions, testbeds or datasets "
            "(CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, BoT-IoT, TON_IoT).",
            "The data is intended for software development, demos, statistical testing and educational use.",
            "Genuine research-data validation requires obtaining and processing the original datasets.",
            "Attack labels describe simulated scenarios; a statistical burst detection is not proof of an attack.",
        ],
        "datasets": {c["dataset_folder"]: {"seed": c["random_seed"], "rows": c["row_count"]} for c in catalog},
    }
    (root / "DATA_PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")
    write_readme(root, catalog)
    return catalog


def write_readme(root, catalog):
    rows = []
    for c in catalog:
        info = json.loads((root / c["dataset_folder"] / "dataset_info.json").read_text())
        ps = info["poisson_demo_support_check"]["packet_count"]
        rows.append(
            f"| {c['dataset_folder']} | {c['row_count']:,} | {info['interval_seconds']} s | "
            f"{info['attack_label_intervals']:,} ({info['attack_label_percentage']}%) | "
            f"{info['baseline_packet_count_stats']['mean']} | {info['baseline_packet_count_stats']['dispersion_index']} | "
            f"{ps['attack_events_with_any_flag']}/{ps['attack_events']} | {ps['attack_interval_recall']} | "
            f"{ps['stable_benign_false_positive_rate'] if ps['stable_benign_false_positive_rate'] is not None else 'n/a'} |")
    txt = f"""# BurstPulse synthetic dataset pack

> **SYNTHETIC DATA.** Every file here was generated programmatically by `generate_datasets.py`
> (v{GENERATOR_VERSION}, fixed seeds). These are *research-inspired test datasets*, **not** genuine
> CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, BoT-IoT or TON_IoT captures. Folder names indicate the
> schema/scenario style only. Attack labels describe simulated scenarios; a Poisson burst detection is
> not proof of an attack. Use for development, demos, statistical testing and education. Genuine
> research validation requires the original datasets.

## Contents
- `<DATASET>/network_traffic.csv`, `<DATASET>/dataset_info.json` for five datasets (10,000 interval rows each)
- `DATASET_CATALOG.csv`, `DATA_PROVENANCE.json`
- `generate_datasets.py` (regenerates everything), `validate_datasets.py` (validation + packaging), `VALIDATION_REPORT.txt`

## Schema (one row = one fixed-length interval)
`interval, timestamp, packet_count, flow_count, measurement_unit, has_attack_label, attack_type, scenario,
scenario_label, period, source_dataset_style, is_synthetic`

`period` marks the first 3,000 intervals (30%) as a clean `baseline` (no attacks, no injected events) for estimating
lambda; the rest is `evaluation`. In the BurstPulse app set **Baseline Partition = 30%** (its minimum) so the app's lambda estimate uses only the clean baseline; the app default of 70% would pull attacks into the baseline. `scenario_label` duplicates `scenario` because the BurstPulse app displays a
column with that name. Full column docs are in each `dataset_info.json`.

## Summary (packet_count; Poisson upper-tail test at alpha=0.001, lambda from baseline only)
| Dataset | Rows | Interval | Attack-labelled | Baseline mean | Baseline var/mean | Attack events flagged | Attack-interval recall | Stable-benign FP rate |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + """

The last three columns are an independent check of how well the generating parameters support the demo; they
are measurements, not guarantees. Non-stationary benign regimes, benign high-volume periods and overdispersed
segments are *intended* to trigger flags (false-positive and model-misfit demonstrations).

## Using with BurstPulse
Upload any `network_traffic.csv`. It is interval-level (`interval` + `packet_count`), so the app's interval-count
path is used; `attack_type == BENIGN` is treated as non-attack. Set the interval duration to the value in
`dataset_info.json` (1, 5 or 2 seconds).

## Regenerate / validate
```
python generate_datasets.py .
python validate_datasets.py .
```
"""
    (root / "README.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    cat = write_all(out)
    for c in cat:
        print(c["dataset_folder"], c["row_count"])
