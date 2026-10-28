"""
tests/test_new_datasets_integration.py
Comprehensive integration tests for the five research-inspired synthetic benchmark datasets.
Covers:
- Loading of all five datasets from BurstPulse_Real_Datasets
- Packet count vs flow count selection
- Baseline isolation (zero contamination from evaluation intervals)
- Alpha threshold behavior (configurable significance)
- Attack label parsing and scenario preservation
- Malformed CSV handling (missing columns, negative/fractional counts)
- Empty and constant traffic edge cases
- Export CSV structure and data integrity
- Synthetic provenance disclosures and measurement intervals
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from scipy.stats import poisson

from dataset_registry import (
    BUILTIN_DATASETS,
    DATASET_ALIASES,
    get_dataset,
    list_builtin_datasets,
    ProvenanceCategory
)
from burst_detector import (
    normalize_traffic_data,
    calculate_baseline_lambda,
    compute_upper_tail_probability,
    analyze_traffic,
    prepare_export_dataframe
)
from data_preprocessor import (
    inspect_schema,
    prepare_interval_dataset,
    parse_attack_label,
    SchemaType
)


DATASET_KEYS = [
    "cic_ids2017_style",
    "cse_cic_ids2018_style",
    "unsw_nb15_style",
    "bot_iot_style",
    "ton_iot_style"
]


# =========================================================================
# 1. SUCCESSFUL LOADING OF ALL FIVE DATASETS
# =========================================================================
@pytest.mark.parametrize("d_id", DATASET_KEYS)
def test_successful_loading_of_all_five_datasets(d_id: str):
    """Verify each of the 5 datasets loads, has 10,000 rows, and validates required columns."""
    entry = get_dataset(d_id)
    assert entry is not None, f"Dataset {d_id} not found in registry"
    assert entry.is_analyzable is True

    csv_path = entry.resolved_path
    assert csv_path.exists(), f"CSV file {csv_path} does not exist"

    df = pd.read_csv(csv_path)
    assert len(df) == 10000, f"Expected 10,000 rows for {d_id}, got {len(df)}"

    required_cols = [
        "interval", "timestamp", "packet_count", "flow_count",
        "measurement_unit", "has_attack_label", "attack_type",
        "scenario", "scenario_label", "source_dataset_style", "is_synthetic"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column '{col}' in {d_id}"

    # Counts must be non-negative integers
    assert (df["packet_count"] >= 0).all()
    assert (df["packet_count"] % 1 == 0).all()
    assert (df["flow_count"] >= 0).all()
    assert (df["flow_count"] % 1 == 0).all()
    assert (df["flow_count"] <= df["packet_count"]).all(), "Flow count should never exceed packet count"


# =========================================================================
# 2. CORRECT SELECTION OF PACKET AND FLOW COUNTS
# =========================================================================
@pytest.mark.parametrize("d_id", DATASET_KEYS)
def test_selection_between_packet_and_flow_counts(d_id: str):
    """Verify analyzer correctly switches between packet_count and flow_count."""
    entry = get_dataset(d_id)
    df = pd.read_csv(entry.resolved_path)

    # 1. Analyze packet counts
    res_pkts = analyze_traffic(df, baseline_pct=30, alpha=0.001, count_col="packet_count", measurement_unit="packets")
    assert res_pkts.measurement_unit == "packets"
    assert res_pkts.baseline_count == 3000
    assert res_pkts.evaluated_count == 7000
    assert res_pkts.lambda_hat > 0

    # 2. Analyze flow counts
    res_flows = analyze_traffic(df, baseline_pct=30, alpha=0.001, count_col="flow_count", measurement_unit="flows")
    assert res_flows.measurement_unit == "flows"
    assert res_flows.baseline_count == 3000
    assert res_flows.evaluated_count == 7000
    assert res_flows.lambda_hat > 0

    # Rates should differ (flows are fewer than packets)
    assert res_pkts.lambda_hat > res_flows.lambda_hat, "Packet arrival rate should exceed flow arrival rate"
    assert res_pkts.total_packets == int(df["packet_count"].sum())
    assert res_flows.total_packets == int(df["flow_count"].sum())


# =========================================================================
# 3. BASELINE ISOLATION (ZERO LEAKAGE)
# =========================================================================
@pytest.mark.parametrize("d_id", DATASET_KEYS)
def test_baseline_isolation_zero_leakage(d_id: str):
    """Verify lambda_hat is calculated strictly from baseline partition with zero leakage from evaluation."""
    entry = get_dataset(d_id)
    df = pd.read_csv(entry.resolved_path)

    # Run analysis at 30% baseline
    original_analysis = analyze_traffic(df, baseline_pct=30, alpha=0.001)

    # Clean baseline verification (intervals 1 to 3,000 must have has_attack_label == False)
    baseline_segment = df.iloc[:3000]
    assert not baseline_segment["has_attack_label"].any(), f"Dataset {d_id} contains attack labels inside baseline!"

    # Perturb the evaluated window drastically (multiply counts by 100)
    perturbed_df = df.copy()
    perturbed_df.loc[3000:, "packet_count"] = perturbed_df.loc[3000:, "packet_count"] * 100

    perturbed_analysis = analyze_traffic(perturbed_df, baseline_pct=30, alpha=0.001)

    # Baseline lambda and variance MUST remain exactly identical
    assert perturbed_analysis.lambda_hat == pytest.approx(original_analysis.lambda_hat, abs=1e-9)
    assert perturbed_analysis.baseline_variance == pytest.approx(original_analysis.baseline_variance, abs=1e-9)
    assert perturbed_analysis.baseline_dispersion == pytest.approx(original_analysis.baseline_dispersion, abs=1e-9)


# =========================================================================
# 4. ALPHA THRESHOLD BEHAVIOR
# =========================================================================
@pytest.mark.parametrize("d_id", DATASET_KEYS)
def test_alpha_threshold_monotonicity_and_correctness(d_id: str):
    """Verify that more permissive alpha thresholds flag a superset of intervals."""
    entry = get_dataset(d_id)
    df = pd.read_csv(entry.resolved_path)

    alphas = [0.001, 0.005, 0.01, 0.025, 0.05]
    burst_counts = []

    for a in alphas:
        res = analyze_traffic(df, baseline_pct=30, alpha=a)
        burst_counts.append(res.burst_count)
        # Every flagged interval MUST have p < a
        flagged = res.evaluated_data[res.evaluated_data["is_burst"]]
        if not flagged.empty:
            assert (flagged["upper_tail_probability"] < a).all()
        # Every non-flagged interval MUST have p >= a
        non_flagged = res.evaluated_data[~res.evaluated_data["is_burst"]]
        if not non_flagged.empty:
            assert (non_flagged["upper_tail_probability"] >= a).all()

    # Monotonicity: higher alpha (more permissive) must flag at least as many bursts
    for i in range(len(burst_counts) - 1):
        assert burst_counts[i] <= burst_counts[i + 1]


# =========================================================================
# 5. ATTACK-LABEL PARSING AND SCENARIO PRESERVATION
# =========================================================================
def test_attack_label_parsing_rules():
    """Verify strict attack label parsing behavior."""
    assert parse_attack_label("BENIGN") == (False, "BENIGN")
    assert parse_attack_label("normal") == (False, "BENIGN")
    assert parse_attack_label("Unlabeled") == (False, "Unlabeled")
    assert parse_attack_label(False) == (False, "BENIGN")
    assert parse_attack_label(True) == (True, "Attack")
    assert parse_attack_label("SIM_DDoS_Window") == (True, "SIM_DDoS_Window")
    assert parse_attack_label("SIM_DoS_Burst") == (True, "SIM_DoS_Burst")


@pytest.mark.parametrize("d_id", DATASET_KEYS)
def test_scenario_names_preserved_in_analysis(d_id: str):
    """Verify scenario and scenario_label are preserved throughout analysis and export."""
    entry = get_dataset(d_id)
    df = pd.read_csv(entry.resolved_path)

    res = analyze_traffic(df, baseline_pct=30, alpha=0.001)
    assert "scenario_label" in res.full_data.columns
    assert "attack_type" in res.full_data.columns
    assert "has_attack_label" in res.full_data.columns

    export = prepare_export_dataframe(res)
    assert "scenario_label" in export.columns
    assert "attack_type" in export.columns
    assert len(export) == 10000


# =========================================================================
# 6. MALFORMED CSV HANDLING
# =========================================================================
def test_malformed_csv_negative_counts():
    """Verify negative counts raise descriptive ValueError."""
    bad_df = pd.DataFrame({
        "interval": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "packet_count": [10, 12, -4, 15, 14, 11, 10, 12, 13, 14]
    })
    with pytest.raises(ValueError, match="cannot be negative"):
        analyze_traffic(bad_df, baseline_pct=30, alpha=0.001)


def test_malformed_csv_fractional_counts():
    """Verify non-integer fractional counts raise descriptive ValueError."""
    bad_df = pd.DataFrame({
        "interval": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "packet_count": [10, 12, 14.5, 15, 14, 11, 10, 12, 13, 14]
    })
    with pytest.raises(ValueError, match="whole non-negative integers|whole integer"):
        analyze_traffic(bad_df, baseline_pct=30, alpha=0.001)


def test_missing_count_column_raises_error():
    """Verify CSV lacking any count column raises helpful ValueError."""
    bad_df = pd.DataFrame({
        "src_ip": ["10.0.0.1", "10.0.0.2"],
        "dst_ip": ["10.0.0.3", "10.0.0.4"]
    })
    with pytest.raises(ValueError, match="Could not find any numeric"):
        analyze_traffic(bad_df, baseline_pct=30, alpha=0.001)


# =========================================================================
# 7. EMPTY OR CONSTANT TRAFFIC EDGE CASES
# =========================================================================
def test_empty_dataset_raises_error():
    """Verify empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="dataset is empty"):
        analyze_traffic(pd.DataFrame(), baseline_pct=30, alpha=0.001)


def test_constant_zero_baseline_raises_error():
    """Verify zero baseline mean raises ValueError as Poisson rate must be positive."""
    zero_df = pd.DataFrame({
        "interval": list(range(1, 21)),
        "packet_count": [0] * 10 + [25] * 10
    })
    with pytest.raises(ValueError, match="strictly positive baseline rate"):
        analyze_traffic(zero_df, baseline_pct=50, alpha=0.001)


def test_constant_nonzero_traffic_yields_zero_bursts():
    """Verify constant steady traffic yields D_base = 0 and underdispersion advisory."""
    steady_df = pd.DataFrame({
        "interval": list(range(1, 101)),
        "packet_count": [20] * 100
    })
    res = analyze_traffic(steady_df, baseline_pct=30, alpha=0.001)
    assert res.lambda_hat == pytest.approx(20.0)
    assert res.baseline_variance == pytest.approx(0.0)
    assert res.baseline_dispersion == pytest.approx(0.0)
    assert res.burst_count == 0
    assert any("Underdispersion detected" in w for w in res.suitability_warnings)


# =========================================================================
# 8. EXPORT CSV OUTPUT INTEGRITY
# =========================================================================
@pytest.mark.parametrize("d_id", DATASET_KEYS)
def test_export_csv_output(d_id: str):
    """Verify exported DataFrame contains all required analysis and metadata columns."""
    entry = get_dataset(d_id)
    df = pd.read_csv(entry.resolved_path)

    res = analyze_traffic(df, baseline_pct=30, alpha=0.001)
    export_df = prepare_export_dataframe(res)

    expected_cols = [
        "interval", "timestamp", "packet_count", "flow_count",
        "measurement_unit", "segment", "baseline_lambda", "baseline_variance",
        "baseline_dispersion", "upper_tail_probability", "is_burst",
        "status", "attack_type", "has_attack_label", "scenario_label", "alpha_threshold"
    ]
    for c in expected_cols:
        assert c in export_df.columns, f"Missing export column '{c}'"

    assert len(export_df) == 10000
    assert export_df["segment"].value_counts()["Baseline"] == 3000
    assert export_df["segment"].value_counts()["Evaluated"] == 7000
    assert (export_df["alpha_threshold"] == 0.001).all()


# =========================================================================
# 9. DATASET PROVENANCE AND SYNTHETIC DISCLOSURE
# =========================================================================
@pytest.mark.parametrize("d_id", DATASET_KEYS)
def test_dataset_provenance_display(d_id: str):
    """Verify each dataset is strictly marked as synthetic with explicit disclaimers."""
    entry = get_dataset(d_id)
    assert entry.is_synthetic is True
    assert entry.category == ProvenanceCategory.SYNTHETIC_SIMULATION
    assert "Synthetic" in entry.name
    assert any("NOT genuine" in s for s in entry.provenance_statements)
    assert any("programmatically" in s for s in entry.provenance_statements)
    assert any("not proof of an attack" in s for s in entry.provenance_statements)
