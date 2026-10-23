"""
data_preprocessor.py
Network Traffic Preprocessing Pipeline.
Converts packet timestamps, flow-level CSVs, and pre-aggregated logs into
verified equal-duration interval packet counts for Poisson burst analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class SchemaType(str, Enum):
    """Detected schema classification for incoming traffic data."""
    INTERVAL_COUNTS = "Interval-Level Counts"
    PACKET_TIMESTAMPS = "Packet-Level Timestamps"
    FLOW_RECORDS = "Network Flow Records"
    UNKNOWN = "Unknown / Unsupported Schema"


@dataclass
class PreprocessingResult:
    """Standardized output of the traffic preprocessing pipeline."""
    dataframe: pd.DataFrame
    schema_type: SchemaType
    interval_duration_sec: int
    source_records: int
    total_intervals: int
    zero_count_intervals: int
    has_labels: bool
    label_distribution: Dict[str, int] = field(default_factory=dict)
    summary_message: str = ""
    warnings: List[str] = field(default_factory=list)


def inspect_schema(df: pd.DataFrame) -> Tuple[SchemaType, Dict[str, Any]]:
    """
    Inspect an input DataFrame to determine its data representation.
    Distinguishes between interval counts, packet timestamps, and flow records.
    """
    cols_clean = {str(c).strip().lower(): c for c in df.columns}
    info: Dict[str, Any] = {"raw_columns": list(df.columns)}

    # 1. Check for Interval-Level Counts (standard burst detector format)
    if "packet_count" in cols_clean or "packets" in cols_clean or "count" in cols_clean:
        # Check if already indexed or aggregated
        if "interval" in cols_clean or len(df.columns) <= 5:
            count_col = cols_clean.get("packet_count") or cols_clean.get("packets") or cols_clean.get("count")
            info["packet_count_col"] = count_col
            info["interval_col"] = cols_clean.get("interval")
            info["has_labels"] = any(c in cols_clean for c in ["attack_type", "label", "has_attack_label"])
            return SchemaType.INTERVAL_COUNTS, info

    # 2. Check for Flow Records (e.g., CICFlowMeter / NetFlow formats)
    flow_packet_candidates = [
        "total fwd packets", "total backward packets", "tot fwd pkts", "tot bwd pkts",
        "flow packets/s", "packets", "packet_count"
    ]
    matched_flow_cols = [cols_clean[c] for c in flow_packet_candidates if c in cols_clean]
    has_timestamp = any(c in cols_clean for c in ["timestamp", "time", "date_time", "flow_start"])

    if matched_flow_cols and has_timestamp:
        info["timestamp_col"] = next(cols_clean[c] for c in ["timestamp", "time", "date_time", "flow_start"] if c in cols_clean)
        info["flow_packet_cols"] = matched_flow_cols
        info["label_col"] = next((cols_clean[c] for c in ["label", "attack", "attack_type"] if c in cols_clean), None)
        return SchemaType.FLOW_RECORDS, info

    # 3. Check for Packet-Level Timestamps (one row per packet)
    if has_timestamp:
        info["timestamp_col"] = next(cols_clean[c] for c in ["timestamp", "time", "date_time", "flow_start"] if c in cols_clean)
        info["label_col"] = next((cols_clean[c] for c in ["label", "attack", "attack_type"] if c in cols_clean), None)
        return SchemaType.PACKET_TIMESTAMPS, info

    # Single numeric column fallback: treat as interval counts
    if len(df.columns) == 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        info["packet_count_col"] = df.columns[0]
        return SchemaType.INTERVAL_COUNTS, info

    return SchemaType.UNKNOWN, info


def parse_timestamp_series(series: pd.Series) -> pd.Series:
    """
    Parse various timestamp representations and normalize to UTC-aware datetime.
    Handles Unix epochs, ISO-8601 strings, and typical log formats.
    """
    # Check if numeric epoch
    if pd.api.types.is_numeric_dtype(series):
        # Infer seconds vs milliseconds
        val_sample = series.dropna().iloc[0] if not series.dropna().empty else 0
        if val_sample > 1e11:  # milliseconds
            return pd.to_datetime(series, unit="ms", utc=True)
        return pd.to_datetime(series, unit="s", utc=True)

    # String format parsing with flexible mixed format parser
    parsed = pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    if parsed.isna().all():
        raise ValueError(
            "Failed to parse timestamp column. Ensure timestamps are in ISO-8601, "
            "standard datetime string, or Unix epoch numeric format."
        )
    return parsed


def aggregate_packet_stream(
    df: pd.DataFrame,
    timestamp_col: str,
    interval_sec: int = 1,
    fill_empty: bool = True,
    label_col: Optional[str] = None
) -> PreprocessingResult:
    """
    Aggregate a packet-level capture (one row per packet) into fixed-duration intervals.
    """
    if interval_sec <= 0:
        raise ValueError(f"Interval duration must be positive, got {interval_sec} seconds.")

    dt_series = parse_timestamp_series(df[timestamp_col])
    valid_mask = ~dt_series.isna()
    valid_df = df[valid_mask].copy()
    valid_df["_dt"] = dt_series[valid_mask]

    if valid_df.empty:
        raise ValueError("No valid timestamp records found in the dataset.")

    # Floor timestamps to interval boundary
    freq_str = f"{interval_sec}s"
    valid_df["_interval_start"] = valid_df["_dt"].dt.floor(freq_str)

    has_labels = label_col is not None and label_col in valid_df.columns

    if has_labels:
        # Group by interval and aggregate packet counts + attack labels
        def _summarize_labels(s):
            attacks = [str(x) for x in s if str(x).upper() not in ["BENIGN", "NORMAL", "0", "NONE", ""]]
            return attacks[0] if attacks else "BENIGN"

        agg_df = valid_df.groupby("_interval_start").agg(
            packet_count=("_dt", "count"),
            attack_type=(label_col, _summarize_labels),
            has_attack_label=(label_col, lambda s: any(str(x).upper() not in ["BENIGN", "NORMAL", "0", "NONE", ""] for x in s))
        ).reset_index()
    else:
        agg_df = valid_df.groupby("_interval_start").agg(
            packet_count=("_dt", "count")
        ).reset_index()
        agg_df["has_attack_label"] = False
        agg_df["attack_type"] = "Unlabeled"

    zero_count_filled = 0
    # Fill empty intervals if continuous timeline is expected
    if fill_empty and len(agg_df) > 1:
        full_range = pd.date_range(
            start=agg_df["_interval_start"].min(),
            end=agg_df["_interval_start"].max(),
            freq=freq_str,
            tz="UTC"
        )
        full_df = pd.DataFrame({"_interval_start": full_range})
        merged = pd.merge(full_df, agg_df, on="_interval_start", how="left")
        zero_count_filled = int(merged["packet_count"].isna().sum())
        merged["packet_count"] = merged["packet_count"].fillna(0).astype(int)
        merged["has_attack_label"] = merged["has_attack_label"].fillna(False).astype(bool)
        merged["attack_type"] = merged["attack_type"].fillna("BENIGN" if has_labels else "Unlabeled")
        agg_df = merged

    agg_df["interval"] = np.arange(1, len(agg_df) + 1)
    agg_df["timestamp"] = agg_df["_interval_start"].dt.strftime("%Y-%m-%d %H:%M:%S")

    label_dist = {}
    if has_labels:
        label_dist = agg_df["attack_type"].value_counts().to_dict()

    final_df = agg_df[["interval", "timestamp", "packet_count", "has_attack_label", "attack_type"]].copy()

    return PreprocessingResult(
        dataframe=final_df,
        schema_type=SchemaType.PACKET_TIMESTAMPS,
        interval_duration_sec=interval_sec,
        source_records=len(df),
        total_intervals=len(final_df),
        zero_count_intervals=zero_count_filled,
        has_labels=has_labels,
        label_distribution=label_dist,
        summary_message=(
            f"Successfully aggregated {len(df):,} packet records into {len(final_df)} "
            f"equal-duration intervals of {interval_sec}s ({zero_count_filled} empty intervals filled with 0)."
        )
    )


def aggregate_flow_records(
    df: pd.DataFrame,
    timestamp_col: str,
    packet_cols: List[str],
    interval_sec: int = 10,
    label_col: Optional[str] = None,
    fill_empty: bool = True
) -> PreprocessingResult:
    """
    Aggregate network flow records (where each row represents a flow with a packet total)
    into fixed-duration intervals. Sums packet counts across all concurrent flows.
    """
    if interval_sec <= 0:
        raise ValueError(f"Interval duration must be positive, got {interval_sec} seconds.")

    dt_series = parse_timestamp_series(df[timestamp_col])
    valid_mask = ~dt_series.isna()
    valid_df = df[valid_mask].copy()
    valid_df["_dt"] = dt_series[valid_mask]

    # Calculate total packets per flow record
    for c in packet_cols:
        valid_df[c] = pd.to_numeric(valid_df[c], errors="coerce").fillna(0)
    valid_df["_flow_packets"] = valid_df[packet_cols].sum(axis=1)

    freq_str = f"{interval_sec}s"
    valid_df["_interval_start"] = valid_df["_dt"].dt.floor(freq_str)

    has_labels = label_col is not None and label_col in valid_df.columns

    if has_labels:
        def _summarize_labels(s):
            attacks = [str(x) for x in s if str(x).upper() not in ["BENIGN", "NORMAL", "0", "NONE", ""]]
            return attacks[0] if attacks else "BENIGN"

        agg_df = valid_df.groupby("_interval_start").agg(
            packet_count=("_flow_packets", "sum"),
            flow_count=("_dt", "count"),
            attack_type=(label_col, _summarize_labels),
            has_attack_label=(label_col, lambda s: any(str(x).upper() not in ["BENIGN", "NORMAL", "0", "NONE", ""] for x in s))
        ).reset_index()
    else:
        agg_df = valid_df.groupby("_interval_start").agg(
            packet_count=("_flow_packets", "sum"),
            flow_count=("_dt", "count")
        ).reset_index()
        agg_df["has_attack_label"] = False
        agg_df["attack_type"] = "Unlabeled"

    zero_count_filled = 0
    if fill_empty and len(agg_df) > 1:
        full_range = pd.date_range(
            start=agg_df["_interval_start"].min(),
            end=agg_df["_interval_start"].max(),
            freq=freq_str,
            tz="UTC"
        )
        full_df = pd.DataFrame({"_interval_start": full_range})
        merged = pd.merge(full_df, agg_df, on="_interval_start", how="left")
        zero_count_filled = int(merged["packet_count"].isna().sum())
        merged["packet_count"] = merged["packet_count"].fillna(0).astype(int)
        merged["flow_count"] = merged["flow_count"].fillna(0).astype(int)
        merged["has_attack_label"] = merged["has_attack_label"].fillna(False).astype(bool)
        merged["attack_type"] = merged["attack_type"].fillna("BENIGN" if has_labels else "Unlabeled")
        agg_df = merged

    agg_df["interval"] = np.arange(1, len(agg_df) + 1)
    agg_df["timestamp"] = agg_df["_interval_start"].dt.strftime("%Y-%m-%d %H:%M:%S")

    label_dist = {}
    if has_labels:
        label_dist = agg_df["attack_type"].value_counts().to_dict()

    final_df = agg_df[["interval", "timestamp", "packet_count", "flow_count", "has_attack_label", "attack_type"]].copy()

    return PreprocessingResult(
        dataframe=final_df,
        schema_type=SchemaType.FLOW_RECORDS,
        interval_duration_sec=interval_sec,
        source_records=len(df),
        total_intervals=len(final_df),
        zero_count_intervals=zero_count_filled,
        has_labels=has_labels,
        label_distribution=label_dist,
        summary_message=(
            f"Successfully aggregated {len(df):,} network flow records into {len(final_df)} "
            f"equal-duration intervals of {interval_sec}s. Total packet count summed across flows."
        )
    )


def prepare_interval_dataset(
    df: pd.DataFrame,
    interval_sec: int = 1
) -> PreprocessingResult:
    """
    Standardize an already aggregated interval-level dataset, validating packet counts
    and preserving optional attack labels.
    """
    cols_clean = {c.strip().lower(): c for c in df.columns}
    
    count_col = next(cols_clean[c] for c in ["packet_count", "packets", "count"] if c in cols_clean)
    packet_counts = pd.to_numeric(df[count_col], errors="coerce")
    
    if packet_counts.isna().any():
        raise ValueError(f"Column '{count_col}' contains missing or non-numeric packet counts.")
    if (packet_counts < 0).any():
        raise ValueError(f"Packet counts in column '{count_col}' cannot be negative.")
    if not (packet_counts % 1 == 0).all():
        raise ValueError(f"Packet counts in column '{count_col}' must be whole integer values.")

    packet_counts = packet_counts.astype(int)
    n = len(df)
    
    res_df = pd.DataFrame({
        "interval": np.arange(1, n + 1),
        "packet_count": packet_counts.values
    })

    # Preserve timestamp if present
    ts_col = next((cols_clean[c] for c in ["timestamp", "time", "date_time"] if c in cols_clean), None)
    if ts_col:
        res_df["timestamp"] = df[ts_col].astype(str).values
    else:
        res_df["timestamp"] = [f"Interval {i}" for i in range(1, n + 1)]

    # Preserve attack labels if present
    label_col = next((cols_clean[c] for c in ["attack_type", "label", "attack"] if c in cols_clean), None)
    has_labels = False
    label_dist = {}
    
    if label_col:
        has_labels = True
        res_df["attack_type"] = df[label_col].astype(str).values
        res_df["has_attack_label"] = res_df["attack_type"].apply(
            lambda x: str(x).upper() not in ["BENIGN", "NORMAL", "0", "NONE", "FALSE", ""]
        )
        label_dist = res_df["attack_type"].value_counts().to_dict()
    elif "has_attack_label" in cols_clean:
        has_labels = True
        res_df["has_attack_label"] = df[cols_clean["has_attack_label"]].astype(bool).values
        res_df["attack_type"] = res_df["has_attack_label"].apply(lambda b: "Attack" if b else "BENIGN")
        label_dist = res_df["attack_type"].value_counts().to_dict()
    else:
        res_df["has_attack_label"] = False
        res_df["attack_type"] = "Unlabeled"

    zero_count = int((res_df["packet_count"] == 0).sum())

    return PreprocessingResult(
        dataframe=res_df,
        schema_type=SchemaType.INTERVAL_COUNTS,
        interval_duration_sec=interval_sec,
        source_records=n,
        total_intervals=n,
        zero_count_intervals=zero_count,
        has_labels=has_labels,
        label_distribution=label_dist,
        summary_message=f"Verified pre-aggregated interval dataset ({n} observations, {zero_count} zero-count intervals)."
    )
