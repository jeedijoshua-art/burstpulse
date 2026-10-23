"""
burst_detector.py
Core statistical logic for Poisson-based Network Traffic Burst Analysis.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import poisson


@dataclass
class AnalysisResult:
    """Container for burst analysis outputs."""
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
        # Strictly after observation 180 (intervals 183, 188, 193, 197)
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


def normalize_traffic_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize input DataFrame.
    
    Ensures:
    - Non-empty DataFrame.
    - Identification of packet count column ('packet_count' or single numeric column).
    - No missing or NaN values in counts.
    - All counts are non-negative.
    - All counts are integers (whole numbers).
    - At least 10 observations for meaningful analysis.
    - Generates 'interval' column if missing (using timestamp or sequential index).
    """
    if raw_df is None or raw_df.empty:
        raise ValueError("The dataset is empty. Please provide a CSV with interval packet counts.")

    df = raw_df.copy()

    # Determine packet count column
    if "packet_count" in df.columns:
        count_col = "packet_count"
    else:
        # Check for numeric or numeric-convertible columns
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
            count_col = numeric_candidates[0]
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
    raw_counts = df[count_col]
    counts_numeric = pd.to_numeric(raw_counts, errors="coerce")

    if counts_numeric.isna().any():
        raise ValueError(
            f"Column '{count_col}' contains missing (NaN) or non-numeric values. "
            "All packet counts must be valid integers."
        )

    if (counts_numeric < 0).any():
        raise ValueError(
            f"Negative packet counts found in '{count_col}'. Packet counts cannot be negative."
        )

    # Check for non-integers (fractional values)
    if not (counts_numeric % 1 == 0).all():
        raise ValueError(
            f"Fractional packet counts found in '{count_col}'. "
            "Packet counts must be whole non-negative integers."
        )

    if len(counts_numeric) < 10:
        raise ValueError(
            f"Dataset has only {len(counts_numeric)} observations. "
            "At least 10 interval observations are required for a meaningful baseline."
        )

    out = pd.DataFrame()

    # Normalize interval or timestamp column first
    if "interval" in df.columns:
        out["interval"] = df["interval"].values
    elif "timestamp" in df.columns:
        out["interval"] = df["timestamp"].astype(str).values
    else:
        out["interval"] = np.arange(1, len(counts_numeric) + 1)

    out["packet_count"] = counts_numeric.astype(int)

    # Preserve metadata fields if present
    if "timestamp" in df.columns and "interval" in df.columns:
        out["timestamp"] = df["timestamp"].astype(str).values
    if "scenario_label" in df.columns:
        out["scenario_label"] = df["scenario_label"].astype(str).values
    if "attack_type" in df.columns:
        out["attack_type"] = df["attack_type"].astype(str).values
    elif "scenario_label" in df.columns:
        out["attack_type"] = out["scenario_label"]
    if "has_attack_label" in df.columns:
        out["has_attack_label"] = df["has_attack_label"].astype(bool).values
    elif "scenario_label" in df.columns:
        out["has_attack_label"] = out["scenario_label"].apply(
            lambda s: str(s).strip().upper() not in ["BENIGN", "NORMAL", "NORMAL BACKGROUND", "NORMAL BASELINE", "0", "NONE", "FALSE", ""]
        ).astype(bool).values
    if "flow_count" in df.columns:
        out["flow_count"] = df["flow_count"].values

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

    tp = int((is_burst & is_attack).sum())       # Flagged burst during attack
    fp = int((is_burst & (~is_attack)).sum())    # Statistical false alarm (high benign volume)
    fn = int(((~is_burst) & is_attack).sum())    # Stealth / low-volume attack not flagged
    tn = int(((~is_burst) & (~is_attack)).sum()) # Normal benign within threshold

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
    
    Raises ValueError if baseline is empty or lambda_hat <= 0.
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
    baseline_pct: int = 70,
    alpha: float = 0.01
) -> AnalysisResult:
    """
    Perform full Poisson baseline estimation and burst classification.
    
    Strictly calculates lambda_hat from the baseline segment only.
    Evaluates upper-tail probabilities for evaluated segment only.
    """
    normalized = normalize_traffic_data(df)
    n = len(normalized)

    # Calculate baseline size: at least 5, at most n - 1
    baseline_n = max(5, min(n - 1, int(np.floor(n * baseline_pct / 100.0))))

    baseline_data = normalized.iloc[:baseline_n].copy()
    evaluated_data = normalized.iloc[baseline_n:].copy()

    # Calculate lambda strictly from baseline
    lambda_hat = calculate_baseline_lambda(baseline_data)

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
        label_contingency=contingency
    )


def prepare_export_dataframe(analysis: AnalysisResult) -> pd.DataFrame:
    """
    Prepare downloadable DataFrame with clean numeric fields and model metadata.
    """
    export = analysis.full_data.copy()
    export["baseline_lambda"] = round(analysis.lambda_hat, 4)
    export["alpha_threshold"] = analysis.alpha
    columns_order = [
        "interval",
        "timestamp",
        "packet_count",
        "segment",
        "baseline_lambda",
        "upper_tail_probability",
        "is_burst",
        "status",
        "attack_type",
        "has_attack_label",
        "alpha_threshold"
    ]
    return export[[c for c in columns_order if c in export.columns]]
