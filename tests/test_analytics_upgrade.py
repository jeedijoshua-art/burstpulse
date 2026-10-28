"""
tests/test_analytics_upgrade.py
Unit and integration tests for BurstPulse statistical analytics upgrade:
- Descriptive statistics computation
- Critical burst threshold (k_crit)
- Garwood Exact vs Wald Confidence Intervals for lambda
- Chi-Square Goodness of Fit with Cochran's pooling
- Benjamini-Hochberg FDR multiple testing adjustment
- Poisson vs Binomial convergence computation
- Point-level explanation narratives
"""

import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson, binom

from burst_detector import (
    compute_descriptive_stats,
    compute_critical_burst_threshold,
    compute_lambda_confidence_interval,
    compute_benjamini_hochberg,
    compute_poisson_goodness_of_fit,
    compute_poisson_binomial_comparison,
    generate_point_explanation,
    analyze_traffic,
    compute_upper_tail_probability
)


def test_compute_descriptive_stats_accuracy():
    """Verify sample variance uses ddof=1, and all percentiles/quartiles are correct."""
    data = pd.Series([10, 20, 30, 40, 50])
    stats = compute_descriptive_stats(data)

    assert stats["count"] == 5
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    # Sample variance: sum((x - 30)^2)/4 = (400 + 100 + 0 + 100 + 400)/4 = 1000/4 = 250.0
    assert stats["sample_variance"] == 250.0
    assert stats["sample_std"] == pytest.approx(math.sqrt(250.0), rel=1e-5)
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["q25"] == 20.0
    assert stats["q75"] == 40.0
    assert stats["iqr"] == 20.0
    assert stats["p90"] == 46.0
    assert stats["p95"] == 48.0
    assert stats["p99"] == 49.6
    assert stats["dispersion_index"] == pytest.approx(250.0 / 30.0, rel=1e-4)


def test_compute_critical_burst_threshold():
    """Verify k_crit is the exact minimum integer where P(X >= k_crit) < alpha."""
    lambda_hat = 15.0
    alpha = 0.01

    k_crit = compute_critical_burst_threshold(lambda_hat, alpha)

    # Upper tail at k_crit must be strictly < alpha
    p_at_crit = poisson.sf(k_crit - 1, lambda_hat)
    assert p_at_crit < alpha

    # Upper tail at k_crit - 1 must be >= alpha
    p_before_crit = poisson.sf(k_crit - 2, lambda_hat)
    assert p_before_crit >= alpha


def test_compute_critical_burst_threshold_various_lambdas():
    """Verify threshold logic holds across different rates and significance levels."""
    for lam in [1.5, 10.0, 50.0, 100.0]:
        for alp in [0.05, 0.01, 0.001]:
            k_crit = compute_critical_burst_threshold(lam, alp)
            assert k_crit > lam
            assert poisson.sf(k_crit - 1, lam) < alp
            assert poisson.sf(k_crit - 2, lam) >= alp


def test_compute_lambda_confidence_interval():
    """Verify Garwood exact and Wald asymptotic intervals contain lambda_hat."""
    lambda_hat = 20.0
    m_baseline = 100

    ci = compute_lambda_confidence_interval(lambda_hat, m_baseline, confidence=0.95)

    assert ci["confidence_level"] == 0.95
    assert ci["lambda_hat"] == 20.0
    garwood_low, garwood_high = ci["exact_garwood"]
    wald_low, wald_high = ci["wald_asymptotic"]

    assert garwood_low < lambda_hat < garwood_high
    assert wald_low < lambda_hat < wald_high
    assert garwood_low > 0
    assert "Exact Garwood method" in ci["interpretation"]


def test_compute_poisson_goodness_of_fit():
    """Verify Chi-Square Goodness of Fit pools bins so that all expected counts >= 5."""
    np.random.seed(42)
    # Generate 500 samples from Poisson(12)
    counts = np.random.poisson(lam=12.0, size=500)
    lambda_hat = float(np.mean(counts))

    gof = compute_poisson_goodness_of_fit(counts, lambda_hat, min_expected=5.0)

    assert gof["valid"] is True
    assert gof["df"] >= 1
    # Check that expected frequencies in pooled bins >= 5.0
    for exp_val in gof["expected_frequencies"]:
        assert exp_val >= 4.99  # allow tiny float margin
    # P-value must be a valid probability
    assert 0.0 <= gof["p_value"] <= 1.0
    assert gof["chi2_stat"] >= 0.0


def test_compute_benjamini_hochberg():
    """Verify Benjamini-Hochberg controls False Discovery Rate properly."""
    # 5 test p-values: two very small, three large
    p_values = np.array([0.0001, 0.0005, 0.04, 0.15, 0.80])
    alpha = 0.01

    fdr = compute_benjamini_hochberg(p_values, alpha=alpha)

    assert fdr["num_discoveries"] >= 2
    assert fdr["significant"][0] is True or fdr["significant"][0] == 1
    assert fdr["significant"][1] is True or fdr["significant"][1] == 1
    assert fdr["significant"][4] is False or fdr["significant"][4] == 0
    # Adjusted p-values must be within [0, 1]
    assert np.all(fdr["adjusted_p"] >= 0.0)
    assert np.all(fdr["adjusted_p"] <= 1.0)


def test_compute_poisson_binomial_comparison():
    """Verify Poisson vs Binomial convergence model."""
    lambda_hat = 10.0
    comp = compute_poisson_binomial_comparison(lambda_hat=lambda_hat)

    assert comp["lambda_hat"] == 10.0
    assert len(comp["k_values"]) > 0
    assert len(comp["poisson_pmf"]) == len(comp["k_values"])
    assert len(comp["scenarios"]) == 3

    # PMFs sum to approximately 1
    assert pytest.approx(sum(comp["poisson_pmf"]), abs=1e-2) == 1.0
    for sc in comp["scenarios"]:
        assert pytest.approx(sum(sc["pmf"]), abs=1e-2) == 1.0
        # As n increases, variance approaches lambda
        assert "n=" in sc["name"]


def test_generate_point_explanation_burst_point():
    """Verify explanation generation for a flagged burst observation."""
    exp = generate_point_explanation(
        interval=350,
        count=45,
        lambda_hat=12.5,
        p_val=1.2e-7,
        alpha=0.001,
        is_burst=True,
        is_baseline=False,
        unit="packets",
        k_crit=23
    )

    assert exp["interval"] == 350
    assert exp["count"] == 45
    assert exp["is_burst"] is True
    assert exp["is_baseline"] is False
    assert "Reject H₀" in exp["decision_label"]
    assert "P(X ≥ 45)" in exp["narrative"]
    assert "strictly less than the significance threshold" in exp["narrative"]
    assert "NOT independent proof of a cyberattack" in exp["caveat"]


def test_generate_point_explanation_normal_point():
    """Verify explanation generation for an observation within threshold."""
    exp = generate_point_explanation(
        interval=120,
        count=14,
        lambda_hat=12.5,
        p_val=0.36,
        alpha=0.001,
        is_burst=False,
        is_baseline=False,
        unit="packets",
        k_crit=23
    )

    assert exp["interval"] == 120
    assert exp["count"] == 14
    assert exp["is_burst"] is False
    assert "Fail to Reject H₀" in exp["decision_label"]
    assert "consistent with ordinary random fluctuations" in exp["narrative"]
    assert "NOT independent proof of a cyberattack" in exp["caveat"]


def test_analyze_traffic_returns_all_upgraded_statistical_fields():
    """Verify analyze_traffic populates all new statistical fields on actual data."""
    df = pd.DataFrame({
        "interval": list(range(1, 101)),
        "packet_count": [10] * 30 + [10] * 65 + [45, 50, 60, 55, 48]
    })

    res = analyze_traffic(df, baseline_pct=30, alpha=0.001)

    assert hasattr(res, "k_crit")
    assert res.k_crit > res.lambda_hat
    assert hasattr(res, "descriptive_full")
    assert res.descriptive_full["count"] == 100
    assert hasattr(res, "descriptive_base")
    assert res.descriptive_base["count"] == 30
    assert hasattr(res, "lambda_ci")
    assert res.lambda_ci["lambda_hat"] == res.lambda_hat
    assert hasattr(res, "goodness_of_fit")
    assert hasattr(res, "bh_fdr")
    assert res.burst_count >= 5
