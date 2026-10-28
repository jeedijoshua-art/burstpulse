"""
burst_detector.py
Core statistical logic for Poisson-based Network Traffic Burst Analysis.
Hardened for data correctness, zero label leakage, statistical diagnostics,
and numerical stability across diverse network CSV schemas.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom, binom, chi2, norm

from data_preprocessor import parse_attack_label


@dataclass
class AnalysisResult:
    """Container for burst analysis outputs, descriptive statistics, and diagnostics."""
    full_data: pd.DataFrame
    baseline_data: pd.DataFrame
    evaluated_data: pd.DataFrame
    lambda_hat: float
    alpha: float
    baseline_pct: int
    baseline_count: int
    evaluated_count: int
    total_packets: int
    peak_evaluated_count: int
    burst_count: int
    burst_percentage: float
    has_labels: bool = False
    label_contingency: Optional[Dict[str, Any]] = None
    baseline_variance: float = 0.0
    baseline_dispersion: float = 1.0
    overall_dispersion: float = 1.0
    suitability_warnings: List[str] = field(default_factory=list)
    measurement_unit: str = "packets"
    experimental_nb: Optional[Dict[str, Any]] = None
    k_crit: int = 0
    descriptive_full: Dict[str, Any] = field(default_factory=dict)
    descriptive_base: Dict[str, Any] = field(default_factory=dict)
    lambda_ci: Dict[str, Any] = field(default_factory=dict)
    goodness_of_fit: Dict[str, Any] = field(default_factory=dict)
    bh_fdr: Dict[str, Any] = field(default_factory=dict)


def generate_synthetic_data(
    seed: int = 42,
    n_total: int = 200,
    base_lambda: int = 20,
    burst_indices: Optional[Dict[int, int]] = None
) -> pd.DataFrame:
    """
    Generate reproducible synthetic interval packet counts using a fixed seed.
    
    Spikes are deliberately injected strictly after observation 180 (the maximum
    supported 90% baseline partition). 0-based indices 182, 187, 192, 196
    correspond to 1-based observation intervals 183, 188, 193, 197.
    This guarantees zero baseline contamination across all supported baseline percentages (30% to 90%).
    """
    if burst_indices is None:
        burst_indices = {182: 48, 187: 55, 192: 44, 196: 62}

    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam=base_lambda, size=n_total).astype(int)

    for idx, val in burst_indices.items():
        if 0 <= idx < n_total:
            counts[idx] = val

    return pd.DataFrame({
        "interval": np.arange(1, n_total + 1),
        "packet_count": counts
    })


def normalize_traffic_data(
    raw_df: pd.DataFrame,
    count_col: Optional[str] = None,
    measurement_unit: str = "packets"
) -> pd.DataFrame:
    """
    Validate and normalize input DataFrame into canonical interval schema.
    
    Ensures:
    - Non-empty DataFrame.
    - Identification of count column without silent column selection on ambiguity.
    - No missing or NaN values in counts.
    - All counts are non-negative whole integers.
    - At least 10 observations for meaningful baseline estimation.
    - Preserves timestamp and label columns without leakage.
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("The dataset is empty. Please provide a CSV with interval packet counts.")

    df = raw_df.copy()
    cols_clean = {str(c).strip().lower(): c for c in df.columns}

    # Determine packet count column
    selected_count_col = None
    if count_col and count_col in df.columns:
        selected_count_col = count_col
    elif "packet_count" in df.columns:
        selected_count_col = "packet_count"
    elif "packet_count" in cols_clean:
        selected_count_col = cols_clean["packet_count"]
    elif "packets" in cols_clean:
        selected_count_col = cols_clean["packets"]
    elif "count" in cols_clean:
        selected_count_col = cols_clean["count"]
    else:
        # Check numeric candidates
        numeric_candidates = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_candidates.append(c)
            else:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() == len(df):
                    df[c] = converted
                    numeric_candidates.append(c)

        if len(numeric_candidates) == 1:
            selected_count_col = numeric_candidates[0]
        elif len(numeric_candidates) == 0:
            raise ValueError(
                "Could not find any numeric packet count column. "
                "Ensure your CSV has a column named 'packet_count' containing integer counts."
            )
        else:
            raise ValueError(
                f"Multiple numeric columns detected: {numeric_candidates}. "
                "Please name the target column 'packet_count' to avoid ambiguity."
            )

    # Validate counts
    raw_counts = df[selected_count_col]
    counts_numeric = pd.to_numeric(raw_counts, errors="coerce")

    if counts_numeric.isna().any():
        raise ValueError(
            f"Column '{selected_count_col}' contains missing (NaN) or non-numeric values. "
            "All packet counts must be valid integers."
        )

    if (counts_numeric < 0).any():
        raise ValueError(
            f"Negative packet counts found in '{selected_count_col}'. Packet counts cannot be negative."
        )

    if not (counts_numeric % 1 == 0).all():
        raise ValueError(
            f"Fractional packet counts found in '{selected_count_col}'. "
            "Packet counts must be whole non-negative integers."
        )

    if len(counts_numeric) < 10:
        raise ValueError(
            f"Dataset has only {len(counts_numeric)} observations. "
            "At least 10 interval observations are required for a meaningful baseline."
        )

    out = pd.DataFrame()

    # Normalize interval or timestamp column
    if "interval" in df.columns:
        out["interval"] = df["interval"].values
    elif "timestamp" in df.columns:
        out["interval"] = df["timestamp"].astype(str).values
    elif "interval" in cols_clean:
        out["interval"] = df[cols_clean["interval"]].values
    elif "timestamp" in cols_clean:
        out["interval"] = df[cols_clean["timestamp"]].astype(str).values
    else:
        out["interval"] = np.arange(1, len(counts_numeric) + 1)

    out["packet_count"] = counts_numeric.astype(int).values
    out["measurement_unit"] = measurement_unit

    # Preserve metadata fields
    ts_key = next((c for c in ["timestamp", "time", "date_time"] if c in cols_clean), None)
    if ts_key and "interval" in df.columns:
        out["timestamp"] = df[cols_clean[ts_key]].astype(str).values

    if "flow_count" in cols_clean:
        out["flow_count"] = pd.to_numeric(df[cols_clean["flow_count"]], errors="coerce").fillna(0).astype(int).values

    # Strict label parsing: only attach label fields if genuine attack markers exist
    label_key = next((c for c in ["attack_type", "label", "attack", "scenario_label"] if c in cols_clean), None)
    if "has_attack_label" in cols_clean:
        raw_vals = df[cols_clean["has_attack_label"]]
        if pd.api.types.is_bool_dtype(raw_vals):
            is_attack_list = raw_vals.tolist()
        else:
            is_attack_list = [bool(parse_attack_label(x)[0]) for x in raw_vals]
        out["has_attack_label"] = is_attack_list
        if "attack_type" in cols_clean:
            out["attack_type"] = df[cols_clean["attack_type"]].astype(str).tolist()
        else:
            out["attack_type"] = ["Attack" if is_att else "BENIGN" for is_att in is_attack_list]
    elif label_key:
        parsed_labels = [parse_attack_label(x) for x in df[cols_clean[label_key]]]
        is_attack_list = [is_att for is_att, _ in parsed_labels]
        if any(is_attack_list):
            out["has_attack_label"] = is_attack_list
            out["attack_type"] = [name if is_att else "BENIGN" for is_att, name in parsed_labels]

    if "scenario" in cols_clean:
        out["scenario"] = df[cols_clean["scenario"]].astype(str).values
    if "scenario_label" in cols_clean:
        out["scenario_label"] = df[cols_clean["scenario_label"]].astype(str).values
    elif "scenario" in cols_clean:
        out["scenario_label"] = df[cols_clean["scenario"]].astype(str).values

    if "source_dataset_style" in cols_clean:
        out["source_dataset_style"] = df[cols_clean["source_dataset_style"]].astype(str).values
    if "is_synthetic" in cols_clean:
        out["is_synthetic"] = df[cols_clean["is_synthetic"]].astype(bool).values

    return out.reset_index(drop=True)


def compute_label_contingency(evaluated_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute overlap and contingency metrics between statistical Poisson burst flags
    and ground-truth attack labels for evaluated observations.
    
    Academic Note: A statistical burst indicates volume anomaly, not intent.
    Precision and recall measure alignment between volume spikes and attack periods.
    """
    if "has_attack_label" not in evaluated_data.columns:
        return {}

    is_burst = evaluated_data["is_burst"].astype(bool)
    is_attack = evaluated_data["has_attack_label"].astype(bool)

    tp = int((is_burst & is_attack).sum())
    fp = int((is_burst & (~is_attack)).sum())
    fn = int(((~is_burst) & is_attack).sum())
    tn = int(((~is_burst) & (~is_attack)).sum())

    total_eval = len(evaluated_data)
    total_flagged = int(is_burst.sum())
    total_attacks = int(is_attack.sum())

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "total_evaluated": total_eval,
        "total_flagged": total_flagged,
        "total_attacks": total_attacks,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "attack_types_present": evaluated_data["attack_type"].unique().tolist() if "attack_type" in evaluated_data.columns else []
    }


def calculate_baseline_lambda(baseline_df: pd.DataFrame) -> float:
    """
    Calculate baseline mean lambda_hat = (1/m) * sum(x_i).
    
    Raises ValueError if baseline is empty, has < 5 observations, or lambda_hat <= 0.
    """
    if baseline_df is None or len(baseline_df) == 0:
        raise ValueError("Baseline dataset is empty.")

    m = len(baseline_df)
    if m < 5:
        raise ValueError(f"Baseline has only {m} observations. At least 5 baseline observations are required.")

    total = float(baseline_df["packet_count"].sum())
    lambda_hat = total / m

    if lambda_hat <= 0:
        raise ValueError(
            f"Estimated baseline mean λ is {lambda_hat:.4f}. "
            "A Poisson burst model requires a strictly positive baseline rate (λ > 0)."
        )

    return lambda_hat


def compute_upper_tail_probability(k: int, lambda_hat: float) -> float:
    """
    Compute upper-tail probability P(X >= k) = P(X > k - 1) for X ~ Poisson(lambda_hat).
    
    Uses numerically stable scipy.stats.poisson.sf(k - 1, lambda_hat).
    Note: For k <= 0, P(X >= k) = 1.0 since Poisson is supported on {0, 1, 2, ...}.
    """
    if lambda_hat <= 0:
        raise ValueError("lambda_hat must be strictly positive.")
    if k <= 0:
        return 1.0
    return float(poisson.sf(k - 1, lambda_hat))


def compute_descriptive_stats(series: pd.Series) -> Dict[str, Any]:
    """
    Compute comprehensive descriptive statistics for an observed arrival count series.
    
    Includes count, mean, median, sample variance (ddof=1), sample standard deviation (ddof=1),
    min, max, quartiles (Q1, Q3), IQR, 90th/95th/99th percentiles, skewness, and excess kurtosis.
    Safely handles empty or degenerate series.
    """
    if series is None or len(series) == 0:
        return {}
    clean = pd.to_numeric(series, errors="coerce").dropna()
    n = len(clean)
    if n == 0:
        return {}

    mean_val = float(clean.mean())
    median_val = float(clean.median())
    var_val = float(clean.var(ddof=1)) if n > 1 else 0.0
    std_val = float(clean.std(ddof=1)) if n > 1 else 0.0
    min_val = float(clean.min())
    max_val = float(clean.max())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    iqr = q75 - q25
    p90 = float(clean.quantile(0.90))
    p95 = float(clean.quantile(0.95))
    p99 = float(clean.quantile(0.99))
    skew_val = float(clean.skew()) if n > 2 else 0.0
    kurt_val = float(clean.kurtosis()) if n > 3 else 0.0
    dispersion = (var_val / mean_val) if mean_val > 0 else 1.0

    return {
        "count": n,
        "mean": mean_val,
        "median": median_val,
        "sample_std": std_val,
        "sample_variance": var_val,
        "min": min_val,
        "max": max_val,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "dispersion_index": dispersion
    }


def compute_critical_burst_threshold(lambda_hat: float, alpha: float) -> int:
    """
    Compute the exact integer cutoff k_crit for a one-sided upper-tail Poisson test:
    k_crit = min { k in Z_{>=0} : P(X >= k) < alpha }.
    
    Any interval count k >= k_crit satisfies P(X >= k) < alpha and is flagged as a potential burst.
    """
    if lambda_hat <= 0:
        raise ValueError("lambda_hat must be strictly positive.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be strictly between 0 and 1.")

    k_guess = int(poisson.ppf(1.0 - alpha, lambda_hat)) + 1
    k_crit = max(1, k_guess)

    while poisson.sf(k_crit - 1, lambda_hat) >= alpha:
        k_crit += 1

    while k_crit > 1 and poisson.sf(k_crit - 2, lambda_hat) < alpha:
        k_crit -= 1

    return k_crit


def compute_lambda_confidence_interval(
    lambda_hat: float,
    m_baseline: int,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate confidence intervals for the Poisson rate parameter lambda.
    
    1. Exact Garwood CI based on the relationship between Poisson and Chi-Square distributions:
       [ chi2(2*T, alpha/2) / (2*m),  chi2(2*(T+1), 1 - alpha/2) / (2*m) ]
       where T = sum(x_i) = m * lambda_hat.
    2. Large-Sample Asymptotic Wald CI:
       lambda_hat +/- z * sqrt(lambda_hat / m).
    """
    if lambda_hat <= 0 or m_baseline <= 0:
        return {}

    alpha_ci = 1.0 - confidence
    T = int(round(m_baseline * lambda_hat))

    ci_exact_low = float(chi2.ppf(alpha_ci / 2.0, 2 * T) / (2.0 * m_baseline)) if T > 0 else 0.0
    ci_exact_high = float(chi2.ppf(1.0 - alpha_ci / 2.0, 2 * (T + 1)) / (2.0 * m_baseline))

    z_score = float(norm.ppf(1.0 - alpha_ci / 2.0))
    se_wald = float(np.sqrt(lambda_hat / m_baseline))
    ci_wald_low = max(0.0, float(lambda_hat - z_score * se_wald))
    ci_wald_high = float(lambda_hat + z_score * se_wald)

    return {
        "confidence_level": confidence,
        "lambda_hat": lambda_hat,
        "sample_size": m_baseline,
        "total_arrivals": T,
        "exact_garwood": (ci_exact_low, ci_exact_high),
        "wald_asymptotic": (ci_wald_low, ci_wald_high),
        "standard_error": se_wald,
        "interpretation": (
            f"With {int(confidence*100)}% confidence, the true stationary baseline arrival rate λ "
            f"lies between {ci_exact_low:.4f} and {ci_exact_high:.4f} (Exact Garwood method)."
        )
    }


def compute_benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.001
) -> Dict[str, Any]:
    """
    Perform Benjamini-Hochberg (BH) False Discovery Rate (FDR) procedure
    for multi-interval hypothesis testing.
    
    Guarantees E[FDR] <= alpha across the evaluated time series.
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return {"significant": np.array([]), "adjusted_p": np.array([]), "num_discoveries": 0}

    sort_idx = np.argsort(p_arr)
    sorted_p = p_arr[sort_idx]

    bh_thresholds = (np.arange(1, m + 1) / m) * alpha
    significant_ranks = sorted_p <= bh_thresholds

    k_star = -1
    for i in range(m - 1, -1, -1):
        if significant_ranks[i]:
            k_star = i
            break

    rejected_sorted = np.zeros(m, dtype=bool)
    if k_star >= 0:
        rejected_sorted[:k_star + 1] = True

    significant = np.zeros(m, dtype=bool)
    significant[sort_idx] = rejected_sorted

    adj_p_sorted = np.zeros(m)
    curr_min = 1.0
    for i in range(m - 1, -1, -1):
        val = (m / (i + 1)) * sorted_p[i]
        curr_min = min(curr_min, val)
        adj_p_sorted[i] = min(1.0, curr_min)

    adjusted_p = np.zeros(m)
    adjusted_p[sort_idx] = adj_p_sorted

    return {
        "significant": significant,
        "adjusted_p": adjusted_p,
        "num_discoveries": int(np.sum(significant)),
        "max_significant_p": float(sorted_p[k_star]) if k_star >= 0 else 0.0
    }


def compute_poisson_goodness_of_fit(
    counts: np.ndarray,
    lambda_hat: float,
    min_expected: float = 5.0
) -> Dict[str, Any]:
    """
    Chi-Square Goodness-of-Fit Test with sparse tail bin pooling.
    
    H0: The baseline data follows Poisson(lambda_hat).
    H1: The baseline data significantly deviates from Poisson(lambda_hat).
    """
    arr = np.asarray(counts, dtype=int)
    n = len(arr)
    if n < 20 or lambda_hat <= 0:
        return {
            "valid": False,
            "reason": "Insufficient baseline sample size (minimum 20 observations required for Chi-Square GoF)."
        }

    observed_counts = Counter(arr)
    max_c = int(np.max(arr))

    pooled_bins = []
    obs_list = []
    exp_list = []

    curr_obs = 0
    curr_exp = 0.0
    curr_labels = []

    for k in range(0, max_c + 15):
        p_k = float(poisson.pmf(k, lambda_hat))
        e_k = n * p_k
        o_k = observed_counts.get(k, 0)

        curr_obs += o_k
        curr_exp += e_k
        curr_labels.append(k)

        if curr_exp >= min_expected:
            label = f"{curr_labels[0]}-{curr_labels[-1]}" if len(curr_labels) > 1 else str(curr_labels[0])
            pooled_bins.append(label)
            obs_list.append(curr_obs)
            exp_list.append(curr_exp)
            curr_obs = 0
            curr_exp = 0.0
            curr_labels = []

    if curr_exp > 0 and len(exp_list) > 0:
        obs_list[-1] += curr_obs
        exp_list[-1] += curr_exp
        if curr_labels:
            first_part = pooled_bins[-1].split("-")[0]
            pooled_bins[-1] = f"{first_part}-{curr_labels[-1]}"

    k_bins = len(obs_list)
    df = k_bins - 1 - 1

    if df < 1:
        return {
            "valid": False,
            "reason": f"Count variability yielded only {k_bins} pooled bins, resulting in df = {df} <= 0."
        }

    obs_arr = np.array(obs_list, dtype=float)
    exp_arr = np.array(exp_list, dtype=float)

    chi2_stat = float(np.sum(((obs_arr - exp_arr) ** 2) / exp_arr))
    p_value = float(1.0 - chi2.cdf(chi2_stat, df))

    return {
        "valid": True,
        "chi2_stat": chi2_stat,
        "df": df,
        "p_value": p_value,
        "num_bins": k_bins,
        "bins": pooled_bins,
        "observed_frequencies": obs_list,
        "expected_frequencies": [round(e, 2) for e in exp_list],
        "reject_h0": p_value < 0.05,
        "conclusion": (
            f"Chi-Square Goodness-of-Fit test yields χ² = {chi2_stat:.2f} (df = {df}, p = {p_value:.4f}). "
            + ("The null hypothesis of a Poisson distribution is REJECTED (p < 0.05), indicating non-Poisson variance clustering or overdispersion."
               if p_value < 0.05 else
               "Fail to reject H₀ (p >= 0.05). The baseline distribution does not significantly deviate from a Poisson model.")
        )
    }


def compute_poisson_binomial_comparison(
    lambda_hat: float,
    max_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Pedagogical & statistical comparison between Poisson(lambda_hat) and Binomial(n, p = lambda_hat / n).
    
    Demonstrates the Poisson Limit Theorem (Law of Rare Events):
    As n -> infty and p -> 0 with n*p = lambda, Binomial(n, p) converges to Poisson(lambda).
    """
    if lambda_hat <= 0:
        return {}

    if max_k is None or max_k <= 0:
        max_k = max(int(np.ceil(lambda_hat * 2.2)), int(lambda_hat + 25))

    k_values = np.arange(0, max_k + 1)
    poisson_pmf = [float(poisson.pmf(k, lambda_hat)) for k in k_values]

    n1 = max(int(np.ceil(lambda_hat * 1.5)), int(lambda_hat + 5))
    p1 = min(0.9, lambda_hat / n1)

    n2 = max(int(round(lambda_hat * 5)), 20)
    p2 = lambda_hat / n2

    n3 = max(int(round(lambda_hat * 50)), 100)
    p3 = lambda_hat / n3

    binom1_pmf = [float(binom.pmf(k, n1, p1)) if k <= n1 else 0.0 for k in k_values]
    binom2_pmf = [float(binom.pmf(k, n2, p2)) if k <= n2 else 0.0 for k in k_values]
    binom3_pmf = [float(binom.pmf(k, n3, p3)) if k <= n3 else 0.0 for k in k_values]

    return {
        "lambda_hat": lambda_hat,
        "k_values": k_values.tolist(),
        "poisson_pmf": poisson_pmf,
        "scenarios": [
            {
                "name": f"Binomial (n={n1}, p={p1:.3f})",
                "n": n1,
                "p": p1,
                "variance": n1 * p1 * (1.0 - p1),
                "pmf": binom1_pmf,
                "description": f"Small trial capacity (Var = {n1*p1*(1-p1):.2f} < λ={lambda_hat:.2f})"
            },
            {
                "name": f"Binomial (n={n2}, p={p2:.3f})",
                "n": n2,
                "p": p2,
                "variance": n2 * p2 * (1.0 - p2),
                "pmf": binom2_pmf,
                "description": f"Moderate trial capacity (Var = {n2*p2*(1-p2):.2f} ~ λ={lambda_hat:.2f})"
            },
            {
                "name": f"Binomial (n={n3}, p={p3:.4f})",
                "n": n3,
                "p": p3,
                "variance": n3 * p3 * (1.0 - p3),
                "pmf": binom3_pmf,
                "description": f"Large trial capacity (Var = {n3*p3*(1-p3):.2f} ≈ λ={lambda_hat:.2f})"
            }
        ]
    }


def generate_point_explanation(
    interval: int,
    count: int,
    lambda_hat: float,
    p_val: float,
    alpha: float,
    is_burst: bool,
    is_baseline: bool,
    label: Optional[str] = None,
    unit: str = "packets",
    k_crit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate an interpretable, mathematically rigorous point-by-point explanation
    for any selected observation interval.
    """
    diff = count - lambda_hat
    multiplier = (count / lambda_hat) if lambda_hat > 0 else 1.0

    if is_baseline:
        decision_label = "Baseline Observation (Calibration Window)"
        status_badge = "🔵 Baseline Reference"
        narrative = (
            f"Interval #{interval:,} belongs to the initial baseline partition "
            f"reserved exclusively for calibrating the reference arrival rate (λ̂ = {lambda_hat:.2f} {unit}/int). "
            f"It measured {count:,} {unit}. To eliminate data leakage, baseline intervals are not "
            f"evaluated against their own estimated mean."
        )
    elif is_burst:
        decision_label = "🚨 Potential Volume Burst (Reject H₀)"
        status_badge = "🚨 Statistically Significant Burst"
        crit_text = f" (exceeds critical cutoff k_crit = {k_crit:,})" if k_crit else ""
        narrative = (
            f"At interval #{interval:,}, the measured count is {count:,} {unit}, which is {diff:+.1f} {unit} "
            f"({multiplier:.2f}×) above the baseline rate λ̂ = {lambda_hat:.2f} {unit}/int{crit_text}. "
            f"Under the null hypothesis (H₀: arrival volume follows the baseline Poisson model), "
            f"the upper-tail probability of observing {count:,} or more arrivals strictly by chance is "
            f"P(X ≥ {count:,}) = {p_val:.2e}. Because this p-value is strictly less than the significance "
            f"threshold α = {alpha:g}, we reject H₀ and flag this observation as a statistically improbable volume surge."
        )
    else:
        decision_label = "✅ Within Expected Variation (Fail to Reject H₀)"
        status_badge = "✅ Within Threshold"
        narrative = (
            f"At interval #{interval:,}, the measured count is {count:,} {unit} "
            f"(deviation of {diff:+.1f} {unit} from baseline λ̂ = {lambda_hat:.2f} {unit}/int). "
            f"Under H₀, the probability of observing {count:,} or more arrivals is "
            f"P(X ≥ {count:,}) = {p_val:.4f} ≥ α = {alpha:g}. Because the survival probability exceeds the "
            f"decision threshold, we fail to reject H₀. This count is consistent with ordinary random fluctuations."
        )

    caveat = (
        "Statistical volume rarity identifies mathematical deviation from the baseline rate; "
        "it is NOT independent proof of a cyberattack or malicious intent. Normal batch synchronization, "
        "video streaming, or protocol bursts can trigger volume flags, while low-rate stealth attacks can evade detection."
    )

    return {
        "interval": interval,
        "count": count,
        "lambda_hat": lambda_hat,
        "expected_count": lambda_hat,
        "difference": diff,
        "multiplier": multiplier,
        "p_value": p_val,
        "alpha": alpha,
        "k_crit": k_crit,
        "is_burst": is_burst,
        "is_baseline": is_baseline,
        "decision_label": decision_label,
        "status_badge": status_badge,
        "narrative": narrative,
        "label": label,
        "caveat": caveat
    }


def analyze_traffic(
    df: pd.DataFrame,
    baseline_pct: int = 30,
    alpha: float = 0.001,
    count_col: Optional[str] = None,
    measurement_unit: str = "packets"
) -> AnalysisResult:
    """
    Perform full Poisson baseline estimation, burst classification,
    descriptive statistics, and diagnostics.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance threshold alpha must be strictly between 0 and 1, got {alpha}.")

    if not (5 <= baseline_pct <= 95):
        raise ValueError(f"Baseline partition percentage must be between 5 and 95, got {baseline_pct}%.")

    normalized = normalize_traffic_data(df, count_col=count_col, measurement_unit=measurement_unit)
    n = len(normalized)

    baseline_n = max(5, min(n - 1, int(np.floor(n * baseline_pct / 100.0))))
    if n - baseline_n < 1:
        raise ValueError(f"Dataset of length {n} leaves no evaluated observations with baseline partition {baseline_pct}%.")

    baseline_data = normalized.iloc[:baseline_n].copy()
    evaluated_data = normalized.iloc[baseline_n:].copy()

    lambda_hat = calculate_baseline_lambda(baseline_data)

    base_counts = baseline_data["packet_count"].values
    baseline_var = float(np.var(base_counts, ddof=1)) if len(base_counts) > 1 else 0.0
    baseline_dispersion = (baseline_var / lambda_hat) if lambda_hat > 0 else 1.0

    all_counts = normalized["packet_count"].values
    overall_mean = float(np.mean(all_counts))
    overall_var = float(np.var(all_counts, ddof=1)) if len(all_counts) > 1 else 0.0
    overall_dispersion = (overall_var / overall_mean) if overall_mean > 0 else 1.0

    # Critical integer cutoff threshold
    k_crit = compute_critical_burst_threshold(lambda_hat, alpha)

    # Descriptive statistics
    descriptive_full = compute_descriptive_stats(normalized["packet_count"])
    descriptive_base = compute_descriptive_stats(baseline_data["packet_count"])

    # Baseline Lambda Confidence Interval
    lambda_ci = compute_lambda_confidence_interval(lambda_hat, baseline_n)

    # Chi-Square Goodness of Fit on Baseline
    goodness_of_fit = compute_poisson_goodness_of_fit(base_counts, lambda_hat)

    # Model suitability warnings
    suitability_warnings = []
    if baseline_dispersion > 1.5:
        suitability_warnings.append(
            f"Overdispersion detected in baseline (D_base = {baseline_dispersion:.2f} > 1.5). "
            "Variance exceeds the Poisson equi-dispersion assumption (Var ≈ Mean), "
            "which may cause the Poisson detector to overestimate anomaly significance and inflate false alarms."
        )
    elif baseline_dispersion < 0.5:
        suitability_warnings.append(
            f"Underdispersion detected in baseline (D_base = {baseline_dispersion:.2f} < 0.5). "
            "Traffic exhibits sub-Poisson regular pacing."
        )

    if overall_dispersion > 2.0:
        suitability_warnings.append(
            f"Substantial overall overdispersion (D = {overall_dispersion:.2f} ≫ 1.0). "
            "Reflects natural heavy-tailed variance clustering or surge periods."
        )

    # Experimental Negative Binomial comparison
    experimental_nb = None
    if baseline_var > lambda_hat and lambda_hat > 0:
        r_nb = float((lambda_hat ** 2) / (baseline_var - lambda_hat))
        p_nb = float(lambda_hat / baseline_var)
        eval_counts = evaluated_data["packet_count"].values
        nb_upper_probs = [
            float(nbinom.sf(int(k) - 1, r_nb, p_nb)) if k > 0 else 1.0
            for k in eval_counts
        ]
        nb_bursts = [p < alpha for p in nb_upper_probs]
        experimental_nb = {
            "model_name": "Negative Binomial (Method-of-Moments Baseline)",
            "dispersion_r": r_nb,
            "success_prob_p": p_nb,
            "upper_tail_probabilities": nb_upper_probs,
            "burst_count": int(sum(nb_bursts)),
            "burst_flags": nb_bursts,
            "interpretation": (
                f"Negative Binomial fits baseline overdispersion (r={r_nb:.2f}, p={p_nb:.3f}). "
                f"Flags {sum(nb_bursts)} bursts under threshold α={alpha:g} (vs. Poisson's burst count)."
            )
        }

    # Upper-tail probabilities for evaluated segment
    eval_counts = evaluated_data["packet_count"].values
    upper_tail_probs = [compute_upper_tail_probability(int(k), lambda_hat) for k in eval_counts]

    evaluated_data["upper_tail_probability"] = upper_tail_probs
    evaluated_data["is_burst"] = evaluated_data["upper_tail_probability"] < alpha
    evaluated_data["status"] = np.where(evaluated_data["is_burst"], "Potential burst", "Within threshold")

    # Benjamini-Hochberg FDR calculation
    bh_fdr = compute_benjamini_hochberg(np.array(upper_tail_probs), alpha)
    evaluated_data["bh_adjusted_p"] = bh_fdr["adjusted_p"]
    evaluated_data["is_burst_fdr"] = bh_fdr["significant"]

    # Combine full dataset
    full_data = normalized.copy()
    full_data["segment"] = ["Baseline" if i < baseline_n else "Evaluated" for i in range(n)]
    full_data["upper_tail_probability"] = np.nan
    full_data["is_burst"] = False
    full_data["status"] = "Baseline"
    full_data["bh_adjusted_p"] = np.nan
    full_data["is_burst_fdr"] = False

    full_data.loc[evaluated_data.index, "upper_tail_probability"] = evaluated_data["upper_tail_probability"]
    full_data.loc[evaluated_data.index, "is_burst"] = evaluated_data["is_burst"]
    full_data.loc[evaluated_data.index, "status"] = evaluated_data["status"]
    full_data.loc[evaluated_data.index, "bh_adjusted_p"] = evaluated_data["bh_adjusted_p"]
    full_data.loc[evaluated_data.index, "is_burst_fdr"] = evaluated_data["is_burst_fdr"]

    total_packets = int(full_data["packet_count"].sum())
    peak_eval = int(evaluated_data["packet_count"].max()) if len(evaluated_data) > 0 else 0
    burst_count = int(evaluated_data["is_burst"].sum())
    burst_pct = (burst_count / len(evaluated_data) * 100.0) if len(evaluated_data) > 0 else 0.0

    has_labels = "has_attack_label" in normalized.columns and bool(normalized["has_attack_label"].any())
    contingency = compute_label_contingency(evaluated_data) if has_labels else None

    return AnalysisResult(
        full_data=full_data,
        baseline_data=baseline_data,
        evaluated_data=evaluated_data,
        lambda_hat=lambda_hat,
        alpha=alpha,
        baseline_pct=baseline_pct,
        baseline_count=baseline_n,
        evaluated_count=len(evaluated_data),
        total_packets=total_packets,
        peak_evaluated_count=peak_eval,
        burst_count=burst_count,
        burst_percentage=burst_pct,
        has_labels=has_labels,
        label_contingency=contingency,
        baseline_variance=baseline_var,
        baseline_dispersion=baseline_dispersion,
        overall_dispersion=overall_dispersion,
        suitability_warnings=suitability_warnings,
        measurement_unit=measurement_unit,
        experimental_nb=experimental_nb,
        k_crit=k_crit,
        descriptive_full=descriptive_full,
        descriptive_base=descriptive_base,
        lambda_ci=lambda_ci,
        goodness_of_fit=goodness_of_fit,
        bh_fdr=bh_fdr
    )


def prepare_export_dataframe(analysis: AnalysisResult) -> pd.DataFrame:
    """
    Prepare downloadable DataFrame with clean numeric fields and model metadata.
    """
    export = analysis.full_data.copy()
    export["baseline_lambda"] = round(analysis.lambda_hat, 4)
    export["baseline_variance"] = round(analysis.baseline_variance, 4)
    export["baseline_dispersion"] = round(analysis.baseline_dispersion, 4)
    export["alpha_threshold"] = analysis.alpha
    export["measurement_unit"] = analysis.measurement_unit

    columns_order = [
        "interval",
        "timestamp",
        "packet_count",
        "flow_count",
        "measurement_unit",
        "segment",
        "baseline_lambda",
        "baseline_variance",
        "baseline_dispersion",
        "upper_tail_probability",
        "is_burst",
        "status",
        "attack_type",
        "has_attack_label",
        "scenario_label",
        "scenario",
        "source_dataset_style",
        "is_synthetic",
        "alpha_threshold"
    ]
    return export[[c for c in columns_order if c in export.columns]]
