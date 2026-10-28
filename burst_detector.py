"""
burst_detector.py
Core statistical logic for Poisson-based Network Traffic Burst Analysis.
Hardened for data correctness, zero label leakage, statistical diagnostics,
and numerical stability across diverse network CSV schemas.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom

from data_preprocessor import parse_attack_label


@dataclass
class AnalysisResult:
    """Container for burst analysis outputs and statistical diagnostics."""
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


def analyze_traffic(
    df: pd.DataFrame,
    baseline_pct: int = 30,
    alpha: float = 0.001,
    count_col: Optional[str] = None,
    measurement_unit: str = "packets"
) -> AnalysisResult:
    """
    Perform full Poisson baseline estimation and burst classification.
    
    - Validates parameters (alpha strictly in (0, 1), baseline_pct in [5, 95]).
    - Strictly calculates lambda_hat from baseline partition only.
    - Evaluates upper-tail probabilities for evaluated segment only.
    - Computes statistical diagnostics (variance, dispersion, model suitability).
    - Prevents label leakage.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance threshold alpha must be strictly between 0 and 1, got {alpha}.")

    if not (5 <= baseline_pct <= 95):
        raise ValueError(f"Baseline partition percentage must be between 5 and 95, got {baseline_pct}%.")

    normalized = normalize_traffic_data(df, count_col=count_col, measurement_unit=measurement_unit)
    n = len(normalized)

    # Calculate baseline size: at least 5, at most n - 1
    baseline_n = max(5, min(n - 1, int(np.floor(n * baseline_pct / 100.0))))
    if n - baseline_n < 1:
        raise ValueError(f"Dataset of length {n} leaves no evaluated observations with baseline partition {baseline_pct}%.")

    baseline_data = normalized.iloc[:baseline_n].copy()
    evaluated_data = normalized.iloc[baseline_n:].copy()

    # Calculate lambda strictly from baseline
    lambda_hat = calculate_baseline_lambda(baseline_data)

    # Baseline variance & dispersion
    base_counts = baseline_data["packet_count"].values
    baseline_var = float(np.var(base_counts, ddof=1)) if len(base_counts) > 1 else 0.0
    baseline_dispersion = (baseline_var / lambda_hat) if lambda_hat > 0 else 1.0

    # Overall counts & dispersion
    all_counts = normalized["packet_count"].values
    overall_mean = float(np.mean(all_counts))
    overall_var = float(np.var(all_counts, ddof=1)) if len(all_counts) > 1 else 0.0
    overall_dispersion = (overall_var / overall_mean) if overall_mean > 0 else 1.0

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

    # Experimental Negative Binomial comparison (optional statistical contrast)
    experimental_nb = None
    if baseline_var > lambda_hat and lambda_hat > 0:
        # Method-of-moments estimation for Negative Binomial
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

    # Calculate upper-tail probabilities for evaluated segment
    eval_counts = evaluated_data["packet_count"].values
    upper_tail_probs = [compute_upper_tail_probability(int(k), lambda_hat) for k in eval_counts]

    evaluated_data["upper_tail_probability"] = upper_tail_probs
    evaluated_data["is_burst"] = evaluated_data["upper_tail_probability"] < alpha
    evaluated_data["status"] = np.where(evaluated_data["is_burst"], "Potential burst", "Within threshold")

    # Combine full dataset with segment labels and status
    full_data = normalized.copy()
    full_data["segment"] = ["Baseline" if i < baseline_n else "Evaluated" for i in range(n)]
    full_data["upper_tail_probability"] = np.nan
    full_data["is_burst"] = False
    full_data["status"] = "Baseline"

    full_data.loc[evaluated_data.index, "upper_tail_probability"] = evaluated_data["upper_tail_probability"]
    full_data.loc[evaluated_data.index, "is_burst"] = evaluated_data["is_burst"]
    full_data.loc[evaluated_data.index, "status"] = evaluated_data["status"]

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
        experimental_nb=experimental_nb
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
