"""
Unit and statistical tests for burst_detector module.
Covers all requirements from Rule 2, Rule 3, and Rule 6.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import poisson

from burst_detector import (
    generate_synthetic_data,
    normalize_traffic_data,
    calculate_baseline_lambda,
    compute_upper_tail_probability,
    analyze_traffic,
    prepare_export_dataframe,
)


# 1. Correct lambda calculation
def test_calculate_baseline_lambda():
    df = pd.DataFrame({"packet_count": [10, 20, 30, 40, 50]})
    lam = calculate_baseline_lambda(df)
    assert lam == pytest.approx(30.0, rel=1e-5)


# 2. Correct upper-tail probability for lambda = 20, k = 30
def test_poisson_reference_lambda_20_k_30():
    # Exact point probability P(X = 30) is approx 0.0083435
    exact_pmf = poisson.pmf(30, 20)
    assert exact_pmf == pytest.approx(0.0083435, abs=0.0001)

    # Upper-tail probability P(X >= 30) = P(X > 29) is approx 0.021818
    upper_tail = compute_upper_tail_probability(30, 20.0)
    assert upper_tail == pytest.approx(0.021818, abs=0.0002)


# 3. Correct threshold behavior at alpha = 0.01
def test_threshold_behavior_alpha_0_01():
    # For lambda = 20, k = 30 has p ~ 0.0218 > 0.01, so NOT flagged
    p_30 = compute_upper_tail_probability(30, 20.0)
    assert p_30 >= 0.01

    # For lambda = 20, k = 48 has p ~ 6.64e-8 < 0.01, so FLAGGED
    p_48 = compute_upper_tail_probability(48, 20.0)
    assert p_48 < 0.01


# 4. High-count observation being flagged under stated baseline
def test_high_count_observation_flagged():
    # Create dataset with baseline mean 20, evaluated count 48
    base_counts = [20] * 70
    eval_counts = [20] * 29 + [48]
    df = pd.DataFrame({"packet_count": base_counts + eval_counts})
    result = analyze_traffic(df, baseline_pct=70, alpha=0.01)

    assert result.lambda_hat == pytest.approx(20.0, rel=1e-5)
    assert result.burst_count == 1
    flagged = result.evaluated_data[result.evaluated_data["is_burst"]]
    assert len(flagged) == 1
    assert flagged.iloc[0]["packet_count"] == 48


# 5. Zero and invalid baseline handling
def test_zero_baseline_handling():
    df_zero = pd.DataFrame({"packet_count": [0] * 20})
    with pytest.raises(ValueError, match="strictly positive"):
        analyze_traffic(df_zero, baseline_pct=70, alpha=0.01)


# 6. Negative packet counts rejected
def test_negative_packet_counts_rejected():
    df_neg = pd.DataFrame({"packet_count": [10, 15, -2, 20, 25, 18, 19, 22, 21, 20]})
    with pytest.raises(ValueError, match="cannot be negative"):
        normalize_traffic_data(df_neg)


# 7. Fractional packet counts rejected
def test_fractional_packet_counts_rejected():
    df_frac = pd.DataFrame({"packet_count": [10, 15, 20.5, 20, 25, 18, 19, 22, 21, 20]})
    with pytest.raises(ValueError, match="whole non-negative integers"):
        normalize_traffic_data(df_frac)


# 8. Missing or non-numeric values rejected
def test_missing_or_non_numeric_values_rejected():
    df_missing = pd.DataFrame({"packet_count": [10, 15, None, 20, 25, 18, 19, 22, 21, 20]})
    with pytest.raises(ValueError, match="missing .* or non-numeric"):
        normalize_traffic_data(df_missing)

    df_str = pd.DataFrame({"packet_count": [10, 15, "invalid", 20, 25, 18, 19, 22, 21, 20]})
    with pytest.raises(ValueError, match="missing .* or non-numeric"):
        normalize_traffic_data(df_str)


# 9. Too few observations rejected
def test_too_few_observations_rejected():
    df_few = pd.DataFrame({"packet_count": [10, 15, 20, 25, 18]})
    with pytest.raises(ValueError, match="At least 10 interval observations"):
        normalize_traffic_data(df_few)


# 10. CSV upload and column validation
def test_column_validation_and_normalization():
    # 1. Explicit packet_count column
    df1 = pd.DataFrame({"interval": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "packet_count": [20]*10})
    norm1 = normalize_traffic_data(df1)
    assert "interval" in norm1.columns and "packet_count" in norm1.columns

    # 2. Timestamp column with packet_count
    df2 = pd.DataFrame({"timestamp": [f"12:0{i}" for i in range(10)], "packet_count": [15]*10})
    norm2 = normalize_traffic_data(df2)
    assert norm2["interval"].iloc[0] == "12:00"

    # 3. Single numeric column with arbitrary name
    df3 = pd.DataFrame({"counts": [18]*10})
    norm3 = normalize_traffic_data(df3)
    assert "packet_count" in norm3.columns
    assert (norm3["packet_count"] == 18).all()

    # 4. Ambiguous multiple numeric columns without packet_count name
    df4 = pd.DataFrame({"col_a": [1]*10, "col_b": [2]*10})
    with pytest.raises(ValueError, match="Multiple numeric columns detected"):
        normalize_traffic_data(df4)


# 11. Baseline/evaluation separation at default 70%
def test_baseline_evaluation_separation_default():
    df = generate_synthetic_data(seed=42, n_total=200, base_lambda=20)
    result = analyze_traffic(df, baseline_pct=70, alpha=0.01)

    # Baseline rows must have is_burst == False, status == 'Baseline', and NaN probability
    baseline_rows = result.full_data[result.full_data["segment"] == "Baseline"]
    assert len(baseline_rows) == 140
    assert not baseline_rows["is_burst"].any()
    assert (baseline_rows["status"] == "Baseline").all()
    assert baseline_rows["upper_tail_probability"].isna().all()

    # Evaluated rows should evaluate to 60 rows with valid probabilities
    eval_rows = result.full_data[result.full_data["segment"] == "Evaluated"]
    assert len(eval_rows) == 60
    assert eval_rows["upper_tail_probability"].notna().all()


# 12. Rule 2 Fix: Zero baseline contamination at maximum supported 90% baseline
def test_spikes_strictly_outside_maximum_90_pct_baseline():
    """
    Ensure all injected spikes (intervals 183, 188, 193, 197) lie strictly beyond
    observation 180 (the 90% partition mark of a 200-row dataset).
    """
    df = generate_synthetic_data(seed=42, n_total=200, base_lambda=20)
    
    # 90% baseline partition contains first 180 observations
    baseline_cutoff = 180
    spike_indices = [182, 187, 192, 196]  # 0-based indices

    # 1. Assert all spikes are strictly after observation 180
    for idx in spike_indices:
        assert idx >= baseline_cutoff, f"Spike at index {idx} violates baseline isolation (> 180)"

    # 2. Run analysis at 90% baseline
    result = analyze_traffic(df, baseline_pct=90, alpha=0.01)
    assert result.baseline_count == 180
    assert result.evaluated_count == 20

    # 3. Check baseline mean is uncontaminated (~19.95, completely unaffected by spikes 48, 55, 44, 62)
    assert result.lambda_hat == pytest.approx(19.95, rel=1e-3)
    assert not result.baseline_data["packet_count"].isin([48, 55, 44, 62]).any()
    baseline_in_full = result.full_data[result.full_data["segment"] == "Baseline"]
    assert not baseline_in_full["is_burst"].any()

    # 4. All 4 spikes must be in the evaluated partition and detected as bursts
    assert result.burst_count == 4
    assert result.peak_evaluated_count == 62
    flagged = result.evaluated_data[result.evaluated_data["is_burst"]]
    assert len(flagged) == 4
    assert set(flagged["packet_count"].values) == {48, 55, 44, 62}


# 13. Reproducibility of synthetic data with fixed seed
def test_reproducibility_synthetic_data():
    df1 = generate_synthetic_data(seed=42, n_total=200, base_lambda=20)
    df2 = generate_synthetic_data(seed=42, n_total=200, base_lambda=20)
    pd.testing.assert_frame_equal(df1, df2)

    # Injected bursts must be present at expected indices (intervals 183, 188, 193, 197)
    assert df1.loc[182, "packet_count"] == 48
    assert df1.loc[187, "packet_count"] == 55
    assert df1.loc[192, "packet_count"] == 44
    assert df1.loc[196, "packet_count"] == 62


# 14. Agreement between AnalysisResult KPIs, full_data, evaluated_data, and export
def test_chart_kpi_table_export_calculations_agree():
    df = generate_synthetic_data(seed=42, n_total=200, base_lambda=20)
    result = analyze_traffic(df, baseline_pct=70, alpha=0.01)
    export_df = prepare_export_dataframe(result)

    # Check total packets consistency
    assert result.total_packets == int(df["packet_count"].sum())
    assert result.total_packets == int(export_df["packet_count"].sum())

    # Check burst counts consistency
    assert result.burst_count == int(result.evaluated_data["is_burst"].sum())
    assert result.burst_count == int(export_df["is_burst"].sum())

    # Check burst percentage formula: (burst_count / evaluated_count) * 100
    expected_pct = (result.burst_count / result.evaluated_count) * 100.0
    assert result.burst_percentage == pytest.approx(expected_pct, rel=1e-5)

    # Check export fields and types
    assert "interval" in export_df.columns
    assert "packet_count" in export_df.columns
    assert "baseline_lambda" in export_df.columns
    assert "alpha_threshold" in export_df.columns
    assert export_df["baseline_lambda"].iloc[0] == round(result.lambda_hat, 4)
    assert export_df["alpha_threshold"].iloc[0] == 0.01


# 15. Strict isolation: No accidental use of evaluation observations when estimating lambda
def test_no_contamination_of_baseline_lambda():
    # 70 baseline observations at 20, 30 evaluated observations at 100
    df = pd.DataFrame({"packet_count": [20]*70 + [100]*30})
    result = analyze_traffic(df, baseline_pct=70, alpha=0.01)

    # If contaminated, lambda would be (70*20 + 30*100)/100 = 44.0
    # Strictly baseline lambda must be exactly 20.0
    assert result.lambda_hat == pytest.approx(20.0, rel=1e-5)
