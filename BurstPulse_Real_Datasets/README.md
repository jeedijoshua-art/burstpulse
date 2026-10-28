# BurstPulse synthetic dataset pack

> **SYNTHETIC DATA.** Every file here was generated programmatically by `generate_datasets.py`
> (v1.0.0, fixed seeds). These are *research-inspired test datasets*, **not** genuine
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
| CIC-IDS2017 | 10,000 | 1 s | 435 (4.35%) | 60.0043 | 1.0174 | 7/7 | 0.9747 | 0.0009 |
| CSE-CIC-IDS2018 | 10,000 | 5 s | 655 (6.55%) | 120.042 | 1.0425 | 5/5 | 0.9802 | n/a |
| UNSW-NB15 | 10,000 | 1 s | 860 (8.6%) | 81.3793 | 2.0827 | 7/8 | 0.5465 | 0.0104 |
| BoT-IoT | 10,000 | 1 s | 1,418 (14.18%) | 12.8417 | 2.2499 | 6/6 | 1.0 | 0.0367 |
| TON_IoT | 10,000 | 2 s | 418 (4.18%) | 34.8733 | 1.6871 | 10/10 | 0.9617 | 0.0071 |

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
