"""
tests/test_hardening_diagnostics.py
Comprehensive regression tests for dataset compatibility, schema hardening,
parameter validation, statistical diagnostics, model suitability, and label integrity.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

from data_preprocessor import (
    parse_attack_label,
    summarize_interval_labels,
    parse_timestamp_series,
    aggregate_packet_stream,
    aggregate_flow_records,
    prepare_interval_dataset,
    inspect_schema,
    SchemaType
)
from burst_detector import (
    normalize_traffic_data,
    calculate_baseline_lambda,
    compute_upper_tail_probability,
    analyze_traffic,
    prepare_export_dataframe,
    AnalysisResult
)


# =====================================================================
# 1. Strict boolean and categorical attack label parsing
# =====================================================================
@pytest.mark.parametrize("benign_input", [
    False, 0, 0.0, "0", "0.0", "False", "false", "FALSE", "f", "F",
    "no", "NO", "none", "NONE", "None", "nan", "NaN", "null", "NULL",
    "BENIGN", "benign", "Benign", "NORMAL", "normal", "Normal",
    "normal background", "normal baseline", "background",
    "unlabeled", "Unlabeled", "unknown", "n/a", "N/A", "", "   "
])
def test_benign_and_false_labels_not_accidentally_attacks(benign_input):
    """Strings like False, 0, BENIGN, NORMAL, empty values must NOT become true attack labels."""
    is_attack, label_str = parse_attack_label(benign_input)
    assert is_attack is False, f"Failed for benign input: {benign_input!r}"


@pytest.mark.parametrize("attack_input,expected_name", [
    (True, "Attack"),
    (1, "Attack"),
    ("1", "Attack"),
    ("True", "True"),
    ("true", "true"),
    ("PortScan", "PortScan"),
    ("DDoS", "DDoS"),
    ("ddos_flood", "ddos_flood"),
    ("Botnet-C2", "Botnet-C2"),
    ("bruteforce", "bruteforce"),
    ("Traffic Burst", "Traffic Burst"),
])
def test_valid_attack_labels_parsed_correctly(attack_input, expected_name):
    """Genuine attack and anomaly markers must be parsed as is_attack=True with clean label name."""
    is_attack, label_str = parse_attack_label(attack_input)
    assert is_attack is True, f"Failed for attack input: {attack_input!r}"
    assert label_str == expected_name


def test_summarize_interval_labels():
    """Verify aggregation across an interval identifies attacks if present, or benign if all benign."""
    s_benign = pd.Series(["BENIGN", "NORMAL", "False", "0", ""])
    is_att, label = summarize_interval_labels(s_benign)
    assert is_att is False
    assert label == "BENIGN"

    s_mixed = pd.Series(["BENIGN", "PortScan", "NORMAL"])
    is_att, label = summarize_interval_labels(s_mixed)
    assert is_att is True
    assert label == "PortScan"


# =====================================================================
# 2. Ambiguous numeric columns & explicit count mapping
# =====================================================================
def test_ambiguous_numeric_columns_requires_explicit_mapping():
    """Multiple numeric columns without canonical 'packet_count' header must raise ValueError."""
    df_ambiguous = pd.DataFrame({
        "flow_duration_ms": [100, 150, 120] * 4,
        "src_port": [80, 443, 8080] * 4,
        "bytes_transferred": [1500, 3000, 4500] * 4,
        "packet_total": [10, 20, 30] * 4
    })
    with pytest.raises(ValueError, match="Multiple numeric columns detected"):
        normalize_traffic_data(df_ambiguous)

    # With explicit count_col, mapping must succeed cleanly
    norm = normalize_traffic_data(df_ambiguous, count_col="packet_total")
    assert "packet_count" in norm.columns
    assert (norm["packet_count"] == [10, 20, 30] * 4).all()


# =====================================================================
# 3. Nonnegative and integer count validation
# =====================================================================
def test_count_validation_rejections():
    """Verify rejection of negative, fractional, and missing counts."""
    # Negative counts
    df_neg = pd.DataFrame({"packet_count": [10, 20, -5] + [15] * 8})
    with pytest.raises(ValueError, match="cannot be negative"):
        normalize_traffic_data(df_neg)

    # Fractional counts
    df_frac = pd.DataFrame({"packet_count": [10, 20, 15.7] + [15] * 8})
    with pytest.raises(ValueError, match="whole non-negative integers"):
        normalize_traffic_data(df_frac)

    # Missing counts
    df_nan = pd.DataFrame({"packet_count": [10, 20, np.nan] + [15] * 8})
    with pytest.raises(ValueError, match="missing .* or non-numeric"):
        normalize_traffic_data(df_nan)


# =====================================================================
# 4. Timestamp parsing across supported ISO and Unix formats
# =====================================================================
def test_timestamp_parsing_across_formats():
    """Handles ISO-8601, Unix seconds, milliseconds, microseconds, and nanoseconds."""
    # Unix seconds (e.g. 1.69e9)
    s_sec = pd.Series([1697378400, 1697378405])
    p_sec = parse_timestamp_series(s_sec)
    assert p_sec.dt.year.iloc[0] == 2023

    # Unix milliseconds (e.g. 1.69e12)
    s_ms = pd.Series([1697378400000, 1697378405000])
    p_ms = parse_timestamp_series(s_ms)
    assert p_ms.dt.year.iloc[0] == 2023

    # Unix microseconds (e.g. 1.69e15)
    s_us = pd.Series([1697378400000000, 1697378405000000])
    p_us = parse_timestamp_series(s_us)
    assert p_us.dt.year.iloc[0] == 2023

    # ISO-8601 strings
    s_iso = pd.Series(["2023-10-15T14:00:00Z", "2023-10-15T14:00:05Z"])
    p_iso = parse_timestamp_series(s_iso)
    assert p_iso.dt.year.iloc[0] == 2023


# =====================================================================
# 5. Chronological sorting for out-of-order timestamp streams
# =====================================================================
def test_out_of_order_packet_events_sorted_chronologically():
    """Packet records provided in reverse or mixed temporal order must be sorted chronologically."""
    df_unsorted = pd.DataFrame({
        "timestamp": [
            "2023-10-15T14:00:05Z",
            "2023-10-15T14:00:01Z",
            "2023-10-15T14:00:03Z",
            "2023-10-15T14:00:01Z"
        ]
    })
    res = aggregate_packet_stream(df_unsorted, timestamp_col="timestamp", interval_sec=1, fill_empty=True)
    out = res.dataframe

    # Total duration is 14:00:01 to 14:00:05 = 5 seconds (5 intervals)
    assert len(out) == 5
    # Second 1 had 2 packets
    assert out.loc[out["interval"] == 1, "packet_count"].iloc[0] == 2
    # Second 2 had 0 packets (filled)
    assert out.loc[out["interval"] == 2, "packet_count"].iloc[0] == 0
    # Second 3 had 1 packet
    assert out.loc[out["interval"] == 3, "packet_count"].iloc[0] == 1
    # Second 4 had 0 packets (filled)
    assert out.loc[out["interval"] == 4, "packet_count"].iloc[0] == 0
    # Second 5 had 1 packet
    assert out.loc[out["interval"] == 5, "packet_count"].iloc[0] == 1


# =====================================================================
# 6. Duplicate timestamps and missing interval policies
# =====================================================================
def test_duplicate_timestamps_and_gap_filling():
    """Duplicate timestamps accumulate counts; missing intervals are filled with 0."""
    df = pd.DataFrame({
        "timestamp": [
            "2023-10-15T14:00:00Z",
            "2023-10-15T14:00:00Z",
            "2023-10-15T14:00:00Z",
            "2023-10-15T14:00:03Z"
        ]
    })
    res = aggregate_packet_stream(df, timestamp_col="timestamp", interval_sec=1, fill_empty=True)
    out = res.dataframe

    # Intervals: 14:00:00 (3 packets), 14:00:01 (0), 14:00:02 (0), 14:00:03 (1)
    assert len(out) == 4
    assert res.zero_count_intervals == 2
    assert out["packet_count"].tolist() == [3, 0, 0, 1]


# =====================================================================
# 7. Flow-level aggregation correctly sums packets across flows
# =====================================================================
def test_flow_record_packet_summation():
    """Sums forward and backward packets across flows and preserves flow count."""
    df_flows = pd.DataFrame({
        "time": ["2023-10-15 14:00:01", "2023-10-15 14:00:02", "2023-10-15 14:00:07"],
        "fwd_pkts": [10, 20, 100],
        "bwd_pkts": [5, 10, 50],
        "label": ["NORMAL", "NORMAL", "DDoS"]
    })
    res = aggregate_flow_records(
        df_flows,
        timestamp_col="time",
        packet_cols=["fwd_pkts", "bwd_pkts"],
        interval_sec=5,
        label_col="label",
        fill_empty=True
    )
    out = res.dataframe

    # 14:00:00 - 14:00:05 has 2 flows, (10+5) + (20+10) = 45 packets
    assert out.loc[out["interval"] == 1, "packet_count"].iloc[0] == 45
    assert out.loc[out["interval"] == 1, "flow_count"].iloc[0] == 2
    assert out.loc[out["interval"] == 1, "has_attack_label"].iloc[0] == False

    # 14:00:05 - 14:00:10 has 1 flow, 150 packets, labeled DDoS
    assert out.loc[out["interval"] == 2, "packet_count"].iloc[0] == 150
    assert out.loc[out["interval"] == 2, "flow_count"].iloc[0] == 1
    assert out.loc[out["interval"] == 2, "has_attack_label"].iloc[0] == True
    assert out.loc[out["interval"] == 2, "attack_type"].iloc[0] == "DDoS"


# =====================================================================
# 8. Zero and invalid baseline handling
# =====================================================================
def test_zero_baseline_rate_rejected():
    """If baseline window contains zero total arrivals, lambda_hat=0 must be rejected."""
    df_zero = pd.DataFrame({"packet_count": [0] * 15})
    with pytest.raises(ValueError, match="strictly positive"):
        analyze_traffic(df_zero, baseline_pct=70, alpha=0.01)


def test_too_few_baseline_observations():
    """Baseline partition with fewer than 5 observations must raise ValueError."""
    baseline_few = pd.DataFrame({"packet_count": [10, 20, 30]})
    with pytest.raises(ValueError, match="At least 5 baseline observations"):
        calculate_baseline_lambda(baseline_few)


# =====================================================================
# 9. Parameter validation: Alpha and Baseline Percentage
# =====================================================================
@pytest.mark.parametrize("invalid_alpha", [-0.01, 0.0, 1.0, 1.5])
def test_invalid_alpha_rejected(invalid_alpha):
    """Significance threshold alpha must be strictly in (0, 1)."""
    df = pd.DataFrame({"packet_count": [20] * 20})
    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        analyze_traffic(df, baseline_pct=70, alpha=invalid_alpha)


@pytest.mark.parametrize("invalid_pct", [0, 4, 96, 100])
def test_invalid_baseline_pct_rejected(invalid_pct):
    """Baseline percentage must be between 5% and 95%."""
    df = pd.DataFrame({"packet_count": [20] * 20})
    with pytest.raises(ValueError, match="Baseline partition percentage must be between 5 and 95"):
        analyze_traffic(df, baseline_pct=invalid_pct, alpha=0.01)


# =====================================================================
# 10. Poisson Survival Probabilities & Decision Boundary
# =====================================================================
def test_poisson_tail_probabilities_and_decision_rule():
    """
    Verify:
    - P(X >= 0) = 1.0 (never flagged as upper-tail burst).
    - Boundary decision rule: strictly p < alpha.
    """
    lam = 20.0
    # k = 0
    assert compute_upper_tail_probability(0, lam) == 1.0
    assert compute_upper_tail_probability(-3, lam) == 1.0

    # For lam = 20:
    # At k = 30, p ~ 0.021818 >= 0.01 -> NOT FLAGGED at alpha = 0.01
    p_30 = compute_upper_tail_probability(30, lam)
    assert p_30 >= 0.01

    # At k = 48, p ~ 6.64e-8 < 0.01 -> FLAGGED at alpha = 0.01
    p_48 = compute_upper_tail_probability(48, lam)
    assert p_48 < 0.01

    # Extreme high count numerical stability
    p_extreme = compute_upper_tail_probability(500, lam)
    assert 0.0 <= p_extreme < 1e-100


# =====================================================================
# 11. Baseline / Evaluation Isolation and Zero Leakage
# =====================================================================
def test_zero_leakage_from_evaluation_into_baseline():
    """
    Ensure massive spikes or attack labels in the evaluation window
    have ZERO effect on baseline mean lambda_hat.
    """
    base_counts = [20] * 70
    eval_calm = [20] * 30
    eval_surge = [500] * 30

    df_calm = pd.DataFrame({"packet_count": base_counts + eval_calm})
    df_surge = pd.DataFrame({"packet_count": base_counts + eval_surge})

    res_calm = analyze_traffic(df_calm, baseline_pct=70, alpha=0.01)
    res_surge = analyze_traffic(df_surge, baseline_pct=70, alpha=0.01)

    # Baseline lambda must be identical down to machine precision
    assert res_calm.lambda_hat == pytest.approx(20.0, rel=1e-9)
    assert res_surge.lambda_hat == pytest.approx(20.0, rel=1e-9)
    assert res_calm.lambda_hat == res_surge.lambda_hat


# =====================================================================
# 12. Statistical Diagnostics, Overdispersion, and Negative Binomial
# =====================================================================
def test_overdispersed_data_diagnostics_and_suitability_warnings():
    """
    Evaluate on overdispersed data (Negative Binomial, seed=404).
    Verifies:
    - baseline variance and dispersion index D > 1.5.
    - overdispersion suitability warning generated.
    - experimental Negative Binomial comparison computed.
    """
    df = pd.read_csv("data/overdispersed_network_traffic.csv")
    result = analyze_traffic(df, baseline_pct=70, alpha=0.01)

    assert result.baseline_variance > result.lambda_hat
    assert result.baseline_dispersion > 1.5
    assert any("Overdispersion" in w for w in result.suitability_warnings)

    # Experimental Negative Binomial comparison must be populated
    assert result.experimental_nb is not None
    nb = result.experimental_nb
    assert "dispersion_r" in nb
    assert nb["dispersion_r"] > 0
    assert "burst_count" in nb


# =====================================================================
# 13. Datasets with no ground-truth labels
# =====================================================================
def test_unlabeled_data_produces_no_fabricated_contingency_metrics():
    """When evaluating unlabeled data, has_labels is False and label_contingency is None."""
    df_unlabeled = pd.DataFrame({
        "interval": np.arange(1, 101),
        "packet_count": [20] * 100
    })
    result = analyze_traffic(df_unlabeled, baseline_pct=70, alpha=0.01)
    assert result.has_labels is False
    assert result.label_contingency is None


# =====================================================================
# 14. CSV export format, headers, and metadata
# =====================================================================
def test_prepare_export_dataframe_schema():
    """Verify downloadable export dataframe contains all required fields including measurement unit."""
    df = pd.DataFrame({
        "interval": np.arange(1, 21),
        "packet_count": [20] * 19 + [50]
    })
    result = analyze_traffic(df, baseline_pct=70, alpha=0.01, measurement_unit="packets")
    export_df = prepare_export_dataframe(result)

    expected_cols = [
        "interval", "packet_count", "measurement_unit", "segment",
        "baseline_lambda", "baseline_variance", "baseline_dispersion",
        "upper_tail_probability", "is_burst", "status", "alpha_threshold"
    ]
    for col in expected_cols:
        assert col in export_df.columns, f"Missing export column: {col}"

    assert export_df["measurement_unit"].iloc[0] == "packets"
    assert export_df["baseline_lambda"].iloc[0] == pytest.approx(20.0, rel=1e-4)


# =====================================================================
# 15. Real sample traffic dataset compatibility
# =====================================================================
def test_real_sample_traffic_dataset():
    """Verify data/sample_traffic.csv processes cleanly through the complete pipeline."""
    sample_df = pd.read_csv("data/sample_traffic.csv")
    result = analyze_traffic(sample_df, baseline_pct=70, alpha=0.01)

    assert result.lambda_hat > 0
    assert result.baseline_count == int(len(sample_df) * 0.7)
    assert result.evaluated_count == len(sample_df) - result.baseline_count
    assert len(result.full_data) == len(sample_df)
