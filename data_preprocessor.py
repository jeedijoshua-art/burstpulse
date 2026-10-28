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


BENIGN_LABEL_VALUES = {
    "benign", "normal", "normal background", "normal baseline", "background",
    "0", "0.0", "false", "f", "no", "n", "none", "nan", "null", "nil", "ok", "safe", "",
    "unlabeled", "unknown", "n/a", "na", "undefined"
}


def parse_attack_label(val: Any) -> Tuple[bool, str]:
    """
    Strictly parse boolean or categorical anomaly/attack labels.
    
    Guarantees:
    - Boolean False / 0 / empty / 'false' / 'benign' / 'normal' / 'unlabeled' evaluate to (False, 'BENIGN' or 'Unlabeled').
    - Boolean True / 1 / 'true' / specific attack names evaluate to (True, '<clean_name>').
    - Whitespace is stripped and case is normalized.
    """
    if val is None or pd.isna(val):
        return False, "BENIGN"
    if isinstance(val, bool):
        return (True, "Attack") if val else (False, "BENIGN")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return (False, "BENIGN") if val == 0 else (True, "Attack")

    s = str(val).strip()
    s_lower = s.lower()
    if s_lower in BENIGN_LABEL_VALUES:
        return False, "Unlabeled" if s_lower in {"unlabeled", "unknown", "n/a", "na", "undefined"} else "BENIGN"
    if s in {"1", "1.0"}:
        return True, "Attack"

    return True, s


def summarize_interval_labels(label_series: pd.Series) -> Tuple[bool, str]:
    """
    Summarize a collection of records within a single aggregated interval.
    If any record has a positive attack label, has_attack_label is True,
    and attack_type is set to the first distinct positive attack name.
    If all records are benign/unlabeled, returns (False, 'BENIGN').
    """
    attacks = []
    for x in label_series:
        is_attack, label_name = parse_attack_label(x)
        if is_attack:
            attacks.append(label_name)
    if attacks:
        return True, attacks[0]
    return False, "BENIGN"


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
    measurement_unit: str = "packets"


def inspect_schema(df: pd.DataFrame) -> Tuple[SchemaType, Dict[str, Any]]:
    """
    Inspect an input DataFrame to determine its data representation.
    Distinguishes between interval counts, packet timestamps, and flow records.
    """
    cols_clean = {str(c).strip().lower(): c for c in df.columns}
    info: Dict[str, Any] = {"raw_columns": list(df.columns)}

    # 1. Check for Interval-Level Counts (standard burst detector format)
    if "packet_count" in cols_clean or "packets" in cols_clean or "count" in cols_clean:
        if "interval" in cols_clean or len(df.columns) <= 5:
            count_col = cols_clean.get("packet_count") or cols_clean.get("packets") or cols_clean.get("count")
            info["packet_count_col"] = count_col
            info["interval_col"] = cols_clean.get("interval")
            info["has_labels"] = any(c in cols_clean for c in ["attack_type", "label", "has_attack_label", "attack", "scenario_label"])
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
    Parse various timestamp representations and normalize to UTC datetime.
    Supports Unix epochs (seconds, milliseconds, microseconds, nanoseconds),
    ISO-8601 strings, RFC 3339, and standard datetime formats.
    """
    if series.empty:
        return pd.to_datetime(pd.Series(dtype="datetime64[ns, UTC]"))

    # Check if numeric epoch
    if pd.api.types.is_numeric_dtype(series):
        non_na = series.dropna()
        if non_na.empty:
            return pd.to_datetime(series, utc=True)
        val_sample = float(non_na.iloc[0])
        # Distinguish seconds vs ms vs us vs ns
        if val_sample > 1e17:  # nanoseconds
            return pd.to_datetime(series, unit="ns", utc=True)
        elif val_sample > 1e14:  # microseconds
            return pd.to_datetime(series, unit="us", utc=True)
        elif val_sample > 1e11:  # milliseconds
            return pd.to_datetime(series, unit="ms", utc=True)
        else:  # seconds
            return pd.to_datetime(series, unit="s", utc=True)

    # String format parsing with flexible mixed format parser
    parsed = pd.to_datetime(series, format="mixed", errors="coerce", utc=True)
    if parsed.isna().all() and not series.isna().all():
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
    Sorts chronologically, validates interval duration, fills empty gaps if requested,
    and aggregates attack labels strictly using parse_attack_label.
    """
    if interval_sec <= 0:
        raise ValueError(f"Interval duration must be positive, got {interval_sec} seconds.")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    dt_series = parse_timestamp_series(df[timestamp_col])
    valid_mask = ~dt_series.isna()
    invalid_count = int((~valid_mask).sum())
    warnings: List[str] = []

    if invalid_count > 0:
        warnings.append(f"Dropped {invalid_count:,} records with invalid/unparseable timestamps.")

    valid_df = df[valid_mask].copy()
    valid_df["_dt"] = dt_series[valid_mask]

    if valid_df.empty:
        raise ValueError("No valid timestamp records found in the dataset.")

    # Chronological sort
    valid_df = valid_df.sort_values("_dt").reset_index(drop=True)

    # Floor timestamps to interval boundary
    freq_str = f"{interval_sec}s"
    valid_df["_interval_start"] = valid_df["_dt"].dt.floor(freq_str)

    has_labels = label_col is not None and label_col in valid_df.columns

    if has_labels:
        grouped = valid_df.groupby("_interval_start")
        agg_rows = []
        for interval_start, group in grouped:
            is_attack, att_name = summarize_interval_labels(group[label_col])
            agg_rows.append({
                "_interval_start": interval_start,
                "packet_count": len(group),
                "attack_type": att_name if is_attack else "BENIGN",
                "has_attack_label": is_attack
            })
        agg_df = pd.DataFrame(agg_rows)
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
        warnings=warnings,
        measurement_unit="packets",
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
    Aggregate network flow records (where each row represents a flow with packet totals)
    into fixed-duration intervals. Sums packet counts across all concurrent flows.
    """
    if interval_sec <= 0:
        raise ValueError(f"Interval duration must be positive, got {interval_sec} seconds.")

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    dt_series = parse_timestamp_series(df[timestamp_col])
    valid_mask = ~dt_series.isna()
    invalid_count = int((~valid_mask).sum())
    warnings: List[str] = []

    if invalid_count > 0:
        warnings.append(f"Dropped {invalid_count:,} records with invalid timestamps.")

    valid_df = df[valid_mask].copy()
    valid_df["_dt"] = dt_series[valid_mask]

    if valid_df.empty:
        raise ValueError("No valid timestamp records found in flow dataset.")

    # Chronological sort
    valid_df = valid_df.sort_values("_dt").reset_index(drop=True)

    # Calculate total packets per flow record
    for c in packet_cols:
        if c in valid_df.columns:
            valid_df[c] = pd.to_numeric(valid_df[c], errors="coerce").fillna(0)
    matched_packet_cols = [c for c in packet_cols if c in valid_df.columns]
    if not matched_packet_cols:
        raise ValueError(f"None of specified packet columns {packet_cols} found in DataFrame.")

    valid_df["_flow_packets"] = valid_df[matched_packet_cols].sum(axis=1)

    freq_str = f"{interval_sec}s"
    valid_df["_interval_start"] = valid_df["_dt"].dt.floor(freq_str)

    has_labels = label_col is not None and label_col in valid_df.columns

    if has_labels:
        grouped = valid_df.groupby("_interval_start")
        agg_rows = []
        for interval_start, group in grouped:
            is_attack, att_name = summarize_interval_labels(group[label_col])
            agg_rows.append({
                "_interval_start": interval_start,
                "packet_count": int(group["_flow_packets"].sum()),
                "flow_count": len(group),
                "attack_type": att_name if is_attack else "BENIGN",
                "has_attack_label": is_attack
            })
        agg_df = pd.DataFrame(agg_rows)
    else:
        agg_df = valid_df.groupby("_interval_start").agg(
            packet_count=("_flow_packets", "sum"),
            flow_count=("_dt", "count")
        ).reset_index()
        agg_df["packet_count"] = agg_df["packet_count"].astype(int)
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
        warnings=warnings,
        measurement_unit="packets",
        summary_message=(
            f"Successfully aggregated {len(df):,} network flow records into {len(final_df)} "
            f"equal-duration intervals of {interval_sec}s. Total packet count summed across flows."
        )
    )


def prepare_interval_dataset(
    df: pd.DataFrame,
    interval_sec: int = 1,
    count_col: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    label_col: Optional[str] = None,
    measurement_unit: str = "packets"
) -> PreprocessingResult:
    """
    Standardize an already aggregated interval-level dataset, validating packet counts,
    chronological ordering, and preserving optional attack labels.
    """
    if df is None or df.empty:
        raise ValueError("The dataset is empty. Please provide a CSV with interval packet counts.")

    cols_clean = {str(c).strip().lower(): c for c in df.columns}

    # Determine count column
    target_count_col = None
    if count_col and count_col in df.columns:
        target_count_col = count_col
    elif "packet_count" in df.columns:
        target_count_col = "packet_count"
    elif "packet_count" in cols_clean:
        target_count_col = cols_clean["packet_count"]
    elif "packets" in cols_clean:
        target_count_col = cols_clean["packets"]
    elif "count" in cols_clean:
        target_count_col = cols_clean["count"]
    else:
        # Check numeric candidates
        numeric_candidates = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_candidates.append(c)
            else:
                converted = pd.to_numeric(df[c], errors="coerce")
                if converted.notna().sum() == len(df):
                    numeric_candidates.append(c)

        if len(numeric_candidates) == 1:
            target_count_col = numeric_candidates[0]
        elif len(numeric_candidates) == 0:
            raise ValueError(
                "Could not find any numeric count column. "
                "Ensure your CSV has a column named 'packet_count' containing non-negative integers."
            )
        else:
            raise ValueError(
                f"Multiple numeric columns detected: {numeric_candidates}. "
                "Please specify the target column name or name it 'packet_count' to avoid ambiguity."
            )

    packet_counts = pd.to_numeric(df[target_count_col], errors="coerce")

    if packet_counts.isna().any():
        raise ValueError(f"Column '{target_count_col}' contains missing or non-numeric packet counts.")
    if (packet_counts < 0).any():
        raise ValueError(f"Packet counts in column '{target_count_col}' cannot be negative.")
    if not (packet_counts % 1 == 0).all():
        raise ValueError(f"Packet counts in column '{target_count_col}' must be whole integer values.")

    n = len(df)

    res_df = pd.DataFrame({
        "interval": np.arange(1, n + 1),
        "packet_count": packet_counts.astype(int).values
    })

    # Timestamp column
    actual_ts_col = timestamp_col if (timestamp_col and timestamp_col in df.columns) else next(
        (cols_clean[c] for c in ["timestamp", "time", "date_time", "datetime"] if c in cols_clean), None
    )
    if actual_ts_col:
        res_df["timestamp"] = df[actual_ts_col].astype(str).values
    elif "interval" in cols_clean and df[cols_clean["interval"]].dtype == object:
        res_df["timestamp"] = df[cols_clean["interval"]].astype(str).values
    else:
        res_df["timestamp"] = [f"Interval {i}" for i in range(1, n + 1)]

    # Preserve flow_count if present
    if "flow_count" in cols_clean:
        res_df["flow_count"] = pd.to_numeric(df[cols_clean["flow_count"]], errors="coerce").fillna(0).astype(int).values

    # Preserve metadata fields (scenarios, measurement units, provenance markers)
    if "scenario" in cols_clean:
        res_df["scenario"] = df[cols_clean["scenario"]].astype(str).values
    if "scenario_label" in cols_clean:
        res_df["scenario_label"] = df[cols_clean["scenario_label"]].astype(str).values
    elif "scenario" in cols_clean:
        res_df["scenario_label"] = df[cols_clean["scenario"]].astype(str).values

    if "measurement_unit" in cols_clean:
        res_df["measurement_unit"] = df[cols_clean["measurement_unit"]].astype(str).values
    if "source_dataset_style" in cols_clean:
        res_df["source_dataset_style"] = df[cols_clean["source_dataset_style"]].astype(str).values
    if "is_synthetic" in cols_clean:
        res_df["is_synthetic"] = df[cols_clean["is_synthetic"]].astype(bool).values
    if "period" in cols_clean:
        res_df["period"] = df[cols_clean["period"]].astype(str).values

    # Preserve attack labels only if explicit label column exists and is not purely 'Unlabeled'
    actual_label_col = label_col if (label_col and label_col in df.columns) else next(
        (cols_clean[c] for c in ["attack_type", "label", "attack", "scenario_label"] if c in cols_clean), None
    )
    has_labels = False
    label_dist = {}

    if "has_attack_label" in cols_clean:
        raw_vals = df[cols_clean["has_attack_label"]]
        if pd.api.types.is_bool_dtype(raw_vals):
            is_attack_list = raw_vals.tolist()
        else:
            is_attack_list = [bool(parse_attack_label(x)[0]) for x in raw_vals]
        has_labels = any(is_attack_list)
        res_df["has_attack_label"] = is_attack_list
        if "attack_type" in cols_clean:
            res_df["attack_type"] = df[cols_clean["attack_type"]].astype(str).tolist()
        else:
            res_df["attack_type"] = ["Attack" if b else "BENIGN" for b in is_attack_list]
        if has_labels:
            label_dist = pd.Series(res_df["attack_type"]).value_counts().to_dict()
    elif actual_label_col:
        parsed_labels = [parse_attack_label(x) for x in df[actual_label_col]]
        is_attack_list = [is_att for is_att, _ in parsed_labels]
        has_labels = any(is_attack_list)
        if has_labels:
            res_df["has_attack_label"] = is_attack_list
            res_df["attack_type"] = [name if is_att else "BENIGN" for is_att, name in parsed_labels]
            label_dist = res_df["attack_type"].value_counts().to_dict()

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
        measurement_unit=measurement_unit,
        summary_message=f"Verified pre-aggregated interval dataset ({n} observations, {zero_count} zero-count intervals)."
    )
