"""
tests/test_data_preprocessor.py
Unit tests for data_preprocessor.py covering schema detection, timestamp parsing,
interval binning, flow aggregation, empty interval filling, and label propagation.
"""

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytest

from data_preprocessor import (
    inspect_schema,
    parse_timestamp_series,
    aggregate_packet_stream,
    aggregate_flow_records,
    prepare_interval_dataset,
    SchemaType
)


def test_schema_detection_interval_counts():
    """Verify correct detection of pre-aggregated interval counts."""
    df1 = pd.DataFrame({"interval": [1, 2, 3], "packet_count": [10, 20, 15]})
    st1, info1 = inspect_schema(df1)
    assert st1 == SchemaType.INTERVAL_COUNTS
    assert info1["packet_count_col"] == "packet_count"

    # Single column fallback
    df2 = pd.DataFrame([10, 25, 30, 15])
    st2, _ = inspect_schema(df2)
    assert st2 == SchemaType.INTERVAL_COUNTS


def test_schema_detection_flow_records():
    """Verify detection of flow-level CSV schemas with packet totals."""
    df = pd.DataFrame({
        "Timestamp": ["2017-07-07 12:00:00", "2017-07-07 12:00:05"],
        "Total Fwd Packets": [15, 30],
        "Total Backward Packets": [10, 20],
        "Label": ["BENIGN", "PortScan"]
    })
    st, info = inspect_schema(df)
    assert st == SchemaType.FLOW_RECORDS
    assert info["timestamp_col"] == "Timestamp"
    assert "Total Fwd Packets" in info["flow_packet_cols"]


def test_schema_detection_packet_timestamps():
    """Verify detection of packet stream with timestamps."""
    df = pd.DataFrame({
        "timestamp": ["2023-10-15T14:00:01Z", "2023-10-15T14:00:02Z"],
        "src_ip": ["192.168.1.1", "192.168.1.2"]
    })
    st, info = inspect_schema(df)
    assert st == SchemaType.PACKET_TIMESTAMPS
    assert info["timestamp_col"] == "timestamp"


def test_parse_timestamp_series_formats():
    """Verify parsing ISO strings, standard logs, and numeric epochs."""
    # ISO string
    s_iso = pd.Series(["2023-10-15T14:00:00Z", "2023-10-15T14:00:01Z"])
    p_iso = parse_timestamp_series(s_iso)
    assert p_iso.dt.year.iloc[0] == 2023

    # Unix epoch in seconds
    s_sec = pd.Series([1697378400, 1697378401])
    p_sec = parse_timestamp_series(s_sec)
    assert p_sec.dt.year.iloc[0] == 2023

    # Unix epoch in milliseconds
    s_ms = pd.Series([1697378400000, 1697378401000])
    p_ms = parse_timestamp_series(s_ms)
    assert p_ms.dt.year.iloc[0] == 2023


def test_aggregate_packet_stream_with_zero_filling_and_labels():
    """Verify packet timestamp binning fills empty intervals and propagates attack labels."""
    # 4 packets: 2 at second 0 (BENIGN), 0 at second 1, 2 at second 2 (PortScan)
    df = pd.DataFrame({
        "timestamp": [
            "2023-10-15T14:00:00Z",
            "2023-10-15T14:00:00.5Z",
            "2023-10-15T14:00:02Z",
            "2023-10-15T14:00:02.5Z"
        ],
        "attack": ["BENIGN", "BENIGN", "PortScan", "PortScan"]
    })

    res = aggregate_packet_stream(df, timestamp_col="timestamp", interval_sec=1, fill_empty=True, label_col="attack")
    out = res.dataframe

    # Total 3 intervals: second 0, second 1 (empty, count=0), second 2
    assert len(out) == 3
    assert res.zero_count_intervals == 1
    assert out.loc[out["interval"] == 1, "packet_count"].iloc[0] == 2
    assert out.loc[out["interval"] == 2, "packet_count"].iloc[0] == 0
    assert bool(out.loc[out["interval"] == 3, "has_attack_label"].iloc[0]) is True
    assert out.loc[out["interval"] == 3, "attack_type"].iloc[0] == "PortScan"
    assert bool(out.loc[out["interval"] == 1, "has_attack_label"].iloc[0]) is False


def test_aggregate_flow_records_sums_packet_counts():
    """Verify flow record aggregation sums total packets across multiple flows."""
    df = pd.DataFrame({
        "flow_start": ["2017-07-07 12:00:01", "2017-07-07 12:00:05", "2017-07-07 12:00:15"],
        "fwd_pkts": [10, 20, 50],
        "bwd_pkts": [5, 10, 25],
        "label": ["BENIGN", "BENIGN", "DDoS"]
    })

    res = aggregate_flow_records(
        df,
        timestamp_col="flow_start",
        packet_cols=["fwd_pkts", "bwd_pkts"],
        interval_sec=10,
        label_col="label",
        fill_empty=True
    )
    out = res.dataframe
    assert len(out) == 2  # 12:00:00-12:00:10 (interval 1), 12:00:10-12:00:20 (interval 2)
    # Interval 1: 15 + 30 = 45 packets across 2 flows
    assert out.loc[out["interval"] == 1, "packet_count"].iloc[0] == 45
    assert out.loc[out["interval"] == 1, "flow_count"].iloc[0] == 2
    # Interval 2: 75 packets, labeled DDoS
    assert out.loc[out["interval"] == 2, "packet_count"].iloc[0] == 75
    assert bool(out.loc[out["interval"] == 2, "has_attack_label"].iloc[0]) is True
    assert out.loc[out["interval"] == 2, "attack_type"].iloc[0] == "DDoS"


def test_prepare_interval_dataset_validates_cleanly():
    """Verify interval dataset preparation preserves clean schema and throws on invalid counts."""
    valid_df = pd.DataFrame({
        "interval": [1, 2, 3],
        "packet_count": [10, 20, 30]
    })
    res = prepare_interval_dataset(valid_df)
    assert len(res.dataframe) == 3

    # Negative count rejection
    bad_df = pd.DataFrame({"interval": [1, 2], "packet_count": [10, -5]})
    with pytest.raises(ValueError, match="cannot be negative"):
        prepare_interval_dataset(bad_df)

    # Fractional count rejection
    frac_df = pd.DataFrame({"interval": [1, 2], "packet_count": [10, 5.5]})
    with pytest.raises(ValueError, match="must be whole integer"):
        prepare_interval_dataset(frac_df)
