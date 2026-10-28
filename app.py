"""
app.py
Network Traffic Burst Analysis Using Poisson Distribution
Applied Mathematics Micro Project — 5 Built-In Datasets + Working CSV Upload
"""

import math
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from burst_detector import (
    generate_synthetic_data,
    normalize_traffic_data,
    analyze_traffic,
    prepare_export_dataframe,
    compute_upper_tail_probability,
    AnalysisResult
)
from landing_page import render_landing_page
from dataset_registry import (
    BUILTIN_DATASETS,
    DATASET_ALIASES,
    DatasetEntry,
    ProvenanceCategory,
    AvailabilityStatus,
    get_dataset,
    list_builtin_datasets
)
from data_preprocessor import (
    inspect_schema,
    aggregate_packet_stream,
    aggregate_flow_records,
    prepare_interval_dataset,
    SchemaType
)
from antenna_component import get_antenna_signal_html, render_antenna_signal

# Page configuration
st.set_page_config(
    page_title="BurstPulse — Network Traffic Burst Analysis",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="auto"
)

# Shared Custom CSS for clean dark analytics theme
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.8rem;
    max-width: 1240px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
}

/* Metric card styling */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(23, 29, 45, 0.85) 0%, rgba(15, 20, 32, 0.95) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
    min-height: 98px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(99, 102, 241, 0.4);
}
div[data-testid="stMetric"] label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    color: #94a3b8 !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
    line-height: 1.25;
}

.header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(99, 102, 241, 0.14);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.3);
}

.mode-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.76rem;
    font-weight: 600;
    background-color: rgba(99, 102, 241, 0.15);
    color: #c4b5fd;
    border: 1px solid rgba(99, 102, 241, 0.35);
}

.provenance-card {
    background: linear-gradient(145deg, rgba(23, 29, 45, 0.9) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 18px;
    font-size: 0.88rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.contingency-card {
    background: linear-gradient(145deg, rgba(24, 28, 42, 0.95) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}

.caveat-banner {
    background: rgba(245, 158, 11, 0.08);
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 10px 16px;
    margin-top: 6px;
    margin-bottom: 18px;
    font-size: 0.88rem;
    color: #fcd34d;
    display: flex;
    align-items: center;
    gap: 8px;
}

.config-card {
    background: linear-gradient(145deg, rgba(23, 29, 45, 0.75) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    font-size: 0.86rem;
    color: #94a3b8;
    line-height: 1.5;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
}

.status-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 10px;
    padding: 10px 16px;
    margin-bottom: 16px;
    font-size: 0.88rem;
    color: #cbd5e1;
}

/* Cohesive Button Design System */
button[data-testid="baseButton-primary"],
button[kind="primary"] {
    background: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(244, 63, 94, 0.5) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    box-shadow: 0 4px 14px rgba(225, 29, 72, 0.28) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button[data-testid="baseButton-primary"]:hover,
button[kind="primary"]:hover {
    background: linear-gradient(135deg, #fb7185 0%, #f43f5e 100%) !important;
    border-color: rgba(251, 113, 133, 0.7) !important;
    box-shadow: 0 6px 20px rgba(225, 29, 72, 0.42) !important;
    transform: translateY(-1px) !important;
}
button[data-testid="baseButton-primary"]:active,
button[kind="primary"]:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 6px rgba(225, 29, 72, 0.25) !important;
}

button[data-testid="baseButton-secondary"],
button[kind="secondary"] {
    background: rgba(30, 41, 59, 0.75) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}
button[data-testid="baseButton-secondary"]:hover,
button[kind="secondary"]:hover {
    background: rgba(51, 65, 85, 0.9) !important;
    border-color: rgba(99, 102, 241, 0.5) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}
button[data-testid="baseButton-secondary"]:active,
button[kind="secondary"]:active {
    transform: translateY(0px) !important;
}
</style>
""", unsafe_allow_html=True)


def navigate_to(page: str, dataset_id: Optional[str] = None):
    """Update page state, dataset selection, and query parameter, then rerun."""
    st.session_state["current_page"] = page
    st.query_params["page"] = page
    if dataset_id is not None:
        st.session_state["selected_dataset_id"] = dataset_id
        st.query_params["dataset"] = dataset_id
    st.rerun()


def render_analyzer_page():
    """
    Render the streamlined single-screen analyzer dashboard with:
      A. Header & Status
      B. Data Source Switcher (Built-in Datasets vs Upload Your Own CSV)
      C. Built-in Dataset Selector (exactly 5 datasets)
      D. Compact Description Panel
      E. In-Page Analysis Configuration
      F. Complete Analysis Results (KPIs, Plotly, Contingency, Table, Interpretation, CSV Export)
    """
    # ---------------------------------------------------------
    # A. HEADER
    # ---------------------------------------------------------
    top_col1, top_col2 = st.columns([1.8, 4.2])
    with top_col1:
        if st.button("← Back to Product Overview", key="analyzer_top_back_btn"):
            navigate_to("landing")

    st.title("📡 BurstPulse — Network Traffic Burst Analysis")
    st.caption("Analyze network traffic, detect unusual bursts, and explore statistical evidence in real time using the Poisson distribution.")

    st.markdown("""
    <div class="caveat-banner">
        <b>🔬 Mathematical Foundation:</b> <span class="mode-badge">5 Built-in Datasets</span> &nbsp;
        Statistical burst detection identifies unusual traffic volume; it does not independently establish malicious intent.
    </div>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # B. DATA SOURCE SWITCHER
    # ---------------------------------------------------------
    # Check if pre-selected via navigation
    initial_source = "Built-in Datasets"
    if st.session_state.get("selected_dataset_id") == "custom_upload" or st.query_params.get("dataset") == "custom_upload":
        initial_source = "Upload Your Own CSV"

    data_source_mode = st.radio(
        "Data Source Selection",
        options=["Built-in Datasets", "Upload Your Own CSV"],
        index=0 if initial_source == "Built-in Datasets" else 1,
        horizontal=True,
        key="analyzer_source_mode",
        help="Choose one of five ready-to-use local datasets or upload your own network traffic CSV."
    )

    raw_dataframe = None
    source_label = ""
    is_synthetic = True
    export_filename = "traffic_burst_analysis.csv"
    active_dataset_entry = None

    # ---------------------------------------------------------
    # C & D. BUILT-IN DATASETS MODE
    # ---------------------------------------------------------
    if data_source_mode == "Built-in Datasets":
        builtin_keys = list(BUILTIN_DATASETS.keys())

        # Resolve pre-selected dataset from session state / query params
        active_id = st.session_state.get("selected_dataset_id") or st.query_params.get("dataset")
        if active_id in DATASET_ALIASES:
            active_id = DATASET_ALIASES[active_id]
        if not active_id or active_id not in builtin_keys:
            active_id = "cic_ids2017_style"

        default_idx = builtin_keys.index(active_id) if active_id in builtin_keys else 0

        col_sel, _ = st.columns([2.5, 1.5])
        with col_sel:
            selected_dataset_id = st.selectbox(
                "Select a Network Traffic Dataset",
                options=builtin_keys,
                index=default_idx,
                format_func=lambda d_id: BUILTIN_DATASETS[d_id].name,
                key="builtin_dataset_selector",
                help="Select one of the five verified research-inspired benchmark datasets."
            )

        # Synchronize selection
        if st.session_state.get("selected_dataset_id") != selected_dataset_id:
            st.session_state["selected_dataset_id"] = selected_dataset_id
            st.query_params["dataset"] = selected_dataset_id

        active_dataset_entry = BUILTIN_DATASETS[selected_dataset_id]
        source_label = active_dataset_entry.name
        is_synthetic = True
        export_filename = f"{selected_dataset_id}_results.csv"

        # Load local data immediately using robust project root resolution
        try:
            target_path = active_dataset_entry.resolved_path
            if not target_path.exists():
                raise FileNotFoundError(f"Dataset file `{active_dataset_entry.local_file_path}` was not found on disk.")
            raw_dataframe = pd.read_csv(target_path)
        except Exception as load_err:
            st.error(f"❌ Failed to load built-in dataset `{active_dataset_entry.name}`: {load_err}")
            st.info("💡 You can select another built-in dataset or upload your own CSV file using the controls above.")
            st.stop()

        # Antenna Signal Animation Triggering (State-Aware)
        current_signal_id = f"builtin_{selected_dataset_id}"
        prev_signal_id = st.session_state.get("last_analyzed_signal_id")
        is_new_signal = (prev_signal_id != current_signal_id)
        st.session_state["last_analyzed_signal_id"] = current_signal_id

        render_antenna_signal(
            status_title=f"Signal Synchronized: {active_dataset_entry.name}",
            status_detail=f"{len(raw_dataframe):,} observation intervals · Baseline λ̂ ready for calibration",
            is_animating=is_new_signal,
            signal_type="builtin",
            height=76
        )

        # D. Description Panel with Visible Provenance & Measurement Interval
        st.markdown(f"""
        <div class="provenance-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">
                    {active_dataset_entry.name}
                    &nbsp;<span class="mode-badge" style="background: rgba(239, 68, 68, 0.15); color: #fca5a5; border-color: rgba(239, 68, 68, 0.35);">🔬 Synthetic Research-Inspired Benchmark</span>
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; font-weight: 600;">
                    📊 {len(raw_dataframe):,} observation intervals &nbsp;·&nbsp; {active_dataset_entry.actual_time_resolution}
                </div>
            </div>
            <div style="background: rgba(245, 158, 11, 0.08); border-left: 3px solid #f59e0b; border-radius: 4px; padding: 8px 12px; margin-bottom: 10px; font-size: 0.82rem; color: #fde68a; line-height: 1.45;">
                <b>⚠️ Synthetic Dataset Disclosure & Provenance:</b> All five benchmark profiles were generated programmatically with fixed random seeds for software development, demos, statistical testing, and educational evaluation. They are <b>NOT</b> genuine downloaded research datasets from the named institutions. Attack labels describe simulated scenarios; statistical burst detection indicates volume rarity, not proof of an attack.
            </div>
            <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin-bottom: 8px;">
                <b>Scenario Overview:</b> {active_dataset_entry.description}
            </p>
            <div style="color: #a5b4fc; font-size: 0.84rem; line-height: 1.45; margin-bottom: 6px;">
                <b>Expected Analysis:</b> {active_dataset_entry.expected_findings}
            </div>
            <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.4; margin-bottom: 6px;">
                <b>Measurement Interval & Unit:</b> <code>{active_dataset_entry.measurement_unit}</code> ({active_dataset_entry.actual_time_resolution}) &nbsp;·&nbsp;
                <b>Clean Baseline:</b> First {active_dataset_entry.clean_baseline_intervals:,} intervals ({active_dataset_entry.clean_baseline_pct}%) reserved for calibration.
            </div>
            <div style="color: #fca5a5; font-size: 0.8rem; line-height: 1.4;">
                <b>Statistical Limitations:</b> {"; ".join(active_dataset_entry.limitations)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # WORKING USER CSV UPLOAD MODE
    # ---------------------------------------------------------
    else:
        st.subheader("📁 Upload Your Own Network Traffic CSV")
        st.markdown("""
        Upload any compatible network traffic CSV file for real-time Poisson burst analysis.
        Supported formats include **interval-level counts** (`packet_count`), **timestamped packet events** (1 row per packet),
        or **network flow records** (with forward/backward packet counts).
        """)

        uploaded_file = st.file_uploader(
            "Drag and drop your CSV here or Browse files",
            type=["csv"],
            key="user_csv_uploader",
            help="Upload an interval-count CSV, packet timestamp log, or flow-level CSV."
        )

        if uploaded_file is not None:
            # File size limit check (e.g. 50 MB)
            if uploaded_file.size > 50 * 1024 * 1024:
                st.error("❌ File size exceeds 50 MB limit. Please provide a smaller sample for interactive evaluation.")
                st.stop()

            try:
                uploaded_raw = pd.read_csv(uploaded_file)
            except Exception as read_err:
                st.error(f"❌ Failed to parse uploaded CSV: {read_err}")
                st.stop()

            if uploaded_raw.empty:
                st.error("❌ The uploaded CSV file is empty.")
                st.stop()

            # Check duplicate columns
            if len(uploaded_raw.columns) != len(set(uploaded_raw.columns)):
                st.error("❌ Duplicate column names detected in uploaded CSV. Please ensure column headers are unique.")
                st.stop()

            st.markdown(f"**Uploaded Data Preview:** `{uploaded_file.name}` ({len(uploaded_raw):,} records, {len(uploaded_raw.columns)} columns)")
            st.dataframe(uploaded_raw.head(5), width="stretch", hide_index=True)

            # Auto-inspect schema
            detected_schema, schema_info = inspect_schema(uploaded_raw)

            # Optional Column Mapping Overrides
            with st.expander("⚙️ Schema & Column Mapping Options", expanded=(detected_schema == SchemaType.UNKNOWN)):
                col_m1, col_m2, col_m3 = st.columns(3)
                all_cols = list(uploaded_raw.columns)

                with col_m1:
                    schema_choice = st.selectbox(
                        "Data Representation",
                        options=[SchemaType.INTERVAL_COUNTS, SchemaType.PACKET_TIMESTAMPS, SchemaType.FLOW_RECORDS],
                        index=0 if detected_schema == SchemaType.INTERVAL_COUNTS else (1 if detected_schema == SchemaType.PACKET_TIMESTAMPS else 2),
                        format_func=lambda s: s.value,
                        key="mapping_schema_choice"
                    )

                with col_m2:
                    default_count_col = schema_info.get("packet_count_col") or next((c for c in all_cols if "packet" in c.lower() or "count" in c.lower()), all_cols[0])
                    mapped_count_col = st.selectbox("Packet Count Column", options=all_cols, index=all_cols.index(default_count_col) if default_count_col in all_cols else 0, key="mapping_count_col")

                with col_m3:
                    ts_candidates = ["(None / Sequential Index)"] + all_cols
                    default_ts = schema_info.get("timestamp_col")
                    ts_idx = ts_candidates.index(default_ts) if default_ts in ts_candidates else 0
                    mapped_ts_col = st.selectbox("Timestamp / Time Column", options=ts_candidates, index=ts_idx, key="mapping_ts_col")

                agg_interval_sec = 1
                if schema_choice in [SchemaType.PACKET_TIMESTAMPS, SchemaType.FLOW_RECORDS]:
                    agg_interval_sec = st.select_slider(
                        "Aggregation Window Duration",
                        options=[1, 5, 10, 30, 60],
                        value=10 if schema_choice == SchemaType.FLOW_RECORDS else 1,
                        format_func=lambda s: f"{s} seconds",
                        key="mapping_agg_slider"
                    )

            # Process based on chosen/detected schema
            try:
                if schema_choice == SchemaType.FLOW_RECORDS:
                    ts_col = mapped_ts_col if mapped_ts_col != "(None / Sequential Index)" else schema_info.get("timestamp_col")
                    if not ts_col or ts_col not in uploaded_raw.columns:
                        st.error("❌ Flow-level aggregation requires a timestamp column.")
                        st.stop()
                    flow_packet_cols = schema_info.get("flow_packet_cols") if schema_info.get("flow_packet_cols") else ([mapped_count_col] if mapped_count_col in uploaded_raw.columns else [all_cols[0]])
                    prep_res = aggregate_flow_records(
                        uploaded_raw,
                        timestamp_col=ts_col,
                        packet_cols=flow_packet_cols,
                        interval_sec=agg_interval_sec,
                        label_col=schema_info.get("label_col")
                    )
                    raw_dataframe = prep_res.dataframe
                    source_label = f"Uploaded Flow CSV: {uploaded_file.name} ({agg_interval_sec}s intervals)"

                elif schema_choice == SchemaType.PACKET_TIMESTAMPS:
                    ts_col = mapped_ts_col if mapped_ts_col != "(None / Sequential Index)" else schema_info.get("timestamp_col")
                    if not ts_col or ts_col not in uploaded_raw.columns:
                        st.error("❌ Packet stream aggregation requires a timestamp column.")
                        st.stop()
                    prep_res = aggregate_packet_stream(
                        uploaded_raw,
                        timestamp_col=ts_col,
                        interval_sec=agg_interval_sec,
                        label_col=schema_info.get("label_col")
                    )
                    raw_dataframe = prep_res.dataframe
                    source_label = f"Uploaded Packet Stream: {uploaded_file.name} ({agg_interval_sec}s intervals)"

                else:
                    # Interval counts format
                    interval_df = uploaded_raw.copy()
                    if mapped_count_col != "packet_count" and mapped_count_col in interval_df.columns:
                        interval_df["packet_count"] = interval_df[mapped_count_col]
                    if mapped_ts_col != "(None / Sequential Index)" and mapped_ts_col in interval_df.columns:
                        interval_df["timestamp"] = interval_df[mapped_ts_col]
                    prep_res = prepare_interval_dataset(interval_df)
                    raw_dataframe = prep_res.dataframe
                    source_label = f"Uploaded Interval CSV: {uploaded_file.name}"

                is_synthetic = False
                export_filename = f"analyzed_{uploaded_file.name}"

                # Antenna Signal Animation Triggering for CSV (State-Aware)
                current_signal_id = f"csv_{uploaded_file.name}_{uploaded_file.size}_{agg_interval_sec}_{schema_choice.value}"
                prev_signal_id = st.session_state.get("last_analyzed_signal_id")
                is_new_signal = (prev_signal_id != current_signal_id)
                st.session_state["last_analyzed_signal_id"] = current_signal_id

                render_antenna_signal(
                    status_title=f"CSV Signal Locked: {uploaded_file.name}",
                    status_detail=f"{len(raw_dataframe):,} intervals processed · Schema verified · Model ready",
                    is_animating=is_new_signal,
                    signal_type="upload",
                    height=76
                )

                # Preprocessing Diagnostics & Pipeline Verification
                with st.expander("🔍 Preprocessing Diagnostics & Pipeline Verification", expanded=False):
                    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                    d_c1.metric("Input Schema", prep_res.schema_type.value)
                    d_c2.metric("Source Records", f"{prep_res.source_records:,}")
                    d_c3.metric("Output Intervals", f"{prep_res.total_intervals:,}")
                    d_c4.metric("Zero-Filled Gaps", f"{prep_res.zero_count_intervals:,}")
                    if prep_res.warnings:
                        for w in prep_res.warnings:
                            st.warning(f"⚠️ {w}")
                    else:
                        st.caption("✅ Temporal order and interval boundaries verified without warnings.")
            except Exception as proc_err:
                st.error(f"❌ Failed to preprocess uploaded data: {proc_err}")
                st.stop()

        else:
            st.info("👆 Please drop a CSV file above or click 'Browse files' to upload network traffic data.")
            st.markdown("""
            ### Supported CSV Schemas:
            1. **Interval-Level Counts:** Columns `interval` and `packet_count` (or any single numeric column).
            2. **Packet Timestamps:** A timestamp column where each row represents one packet event.
            3. **Network Flow Records:** Timestamped flow logs with forward and backward packet totals.
            """)
            st.caption("Tip: You can test the upload flow using `data/normal_network_traffic.csv` located in this repository.")
            st.stop()

    # ---------------------------------------------------------
    # E. IN-PAGE ANALYSIS CONFIGURATION
    # ---------------------------------------------------------
    st.subheader("⚙️ Analysis Configuration")

    # Packet versus Flow Selector (Phase 2 & 4)
    has_flow_col = "flow_count" in raw_dataframe.columns
    if has_flow_col:
        selected_metric = st.radio(
            "Analysis Count Metric",
            options=["Packet Count (packet_count)", "Flow Count (flow_count)"],
            index=0,
            horizontal=True,
            key="analyzer_metric_target",
            help="Select whether to evaluate arrival bursts using packet arrivals or flow session counts."
        )
        count_col = "flow_count" if "flow_count" in selected_metric else "packet_count"
        metric_unit_label = "flows" if count_col == "flow_count" else "packets"
    else:
        count_col = "packet_count"
        metric_unit_label = "packets"

    cfg_col1, cfg_col2 = st.columns([1.1, 1], gap="large")

    with cfg_col1:
        sub_c1, sub_c2 = st.columns(2)
        with sub_c1:
            alpha = st.select_slider(
                "Significance Threshold (α)",
                options=[0.001, 0.005, 0.01, 0.025, 0.05],
                value=0.001,
                format_func=lambda x: f"{x:g} ({x*100:g}%)",
                key="analyzer_alpha_slider",
                help="Flag intervals whose upper-tail survival probability P(X ≥ k) under the baseline rate is strictly less than α."
            )

        with sub_c2:
            default_base = getattr(active_dataset_entry, "clean_baseline_pct", 30) if active_dataset_entry else 30
            baseline_pct = st.slider(
                "Baseline Partition (%)",
                min_value=10,
                max_value=90,
                value=default_base,
                step=5,
                key="analyzer_baseline_slider",
                help="Percentage of initial observations strictly reserved for estimating baseline arrival rate λ̂ (default 30% for benchmark pack)."
            )

    total_raw_obs = len(raw_dataframe) if raw_dataframe is not None else 10000
    m_baseline = max(1, int(math.floor(total_raw_obs * (baseline_pct / 100.0))))
    eval_count = total_raw_obs - m_baseline

    with cfg_col2:
        st.markdown(f"""
        <div class="config-card">
            <div style="font-weight: 700; color: #f8fafc; margin-bottom: 6px; display: flex; justify-content: space-between; align-items: center;">
                <span>Partition Boundary: {baseline_pct}% Calibration</span>
                <span class="mode-badge" style="background: rgba(99, 102, 241, 0.2); color: #c4b5fd;">α = {alpha:g} ({alpha*100:g}%)</span>
            </div>
            <div style="color: #cbd5e1; font-size: 0.86rem; margin-bottom: 3px;">
                <b>Target Metric:</b> Evaluating <code>{count_col}</code> ({metric_unit_label})
            </div>
            <div style="color: #cbd5e1; font-size: 0.86rem; margin-bottom: 3px;">
                <b>Baseline Window:</b> First {m_baseline:,} intervals (1 to {m_baseline}) reserved for λ̂ calibration.
            </div>
            <div style="color: #cbd5e1; font-size: 0.86rem; margin-bottom: 5px;">
                <b>Evaluation Window:</b> Remaining {eval_count:,} intervals ({m_baseline+1} to {total_raw_obs}) evaluated for volume bursts.
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; border-top: 1px solid rgba(255, 255, 255, 0.07); padding-top: 5px; margin-top: 3px;">
                🛡️ <b>Zero Contamination:</b> Baseline arrival rate λ̂ is strictly isolated from evaluated observations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # F. COMPLETE LIVE RESULTS ON THE SAME SCREEN
    # ---------------------------------------------------------
    try:
        analysis = analyze_traffic(
            raw_dataframe,
            baseline_pct=baseline_pct,
            alpha=alpha,
            count_col=count_col,
            measurement_unit=metric_unit_label
        )
    except ValueError as val_err:
        st.error(f"❌ **Data Validation Error:** {val_err}")
        st.stop()
    except Exception as unexpected_err:
        st.error(f"❌ **Unexpected Error during Analysis:** {unexpected_err}")
        st.stop()

    # Calculate dispersion index (Var/Mean)
    full_counts = analysis.full_data["packet_count"]
    mean_count = float(np.mean(full_counts))
    var_count = float(np.var(full_counts, ddof=1)) if len(full_counts) > 1 else 0.0
    dispersion_index = (var_count / mean_count) if mean_count > 0 else 1.0

    st.markdown(
        f'<div class="status-card">'
        f'<b>Active Data Source:</b> {source_label} &nbsp;·&nbsp; '
        f'<b>Metric:</b> {count_col} ({metric_unit_label}) &nbsp;·&nbsp; '
        f'<b>Total Intervals:</b> {len(analysis.full_data):,} &nbsp;·&nbsp; '
        f'<b>Baseline Window:</b> First {analysis.baseline_count:,} intervals ({analysis.baseline_pct}%) &nbsp;·&nbsp; '
        f'<b>Evaluated Window:</b> {analysis.evaluated_count:,} intervals &nbsp;·&nbsp; '
        f'<b>Threshold:</b> α = {analysis.alpha:g}'
        f'</div>',
        unsafe_allow_html=True
    )

    # Model Suitability Warnings Banner (Phase 4)
    if analysis.suitability_warnings:
        for w in analysis.suitability_warnings:
            st.warning(f"⚠️ **Statistical Model Advisory:** {w}")

    # Baseline & Dispersion Diagnostic Badges
    st.markdown(f"""
    <div style="display: flex; gap: 12px; margin-bottom: 14px; flex-wrap: wrap;">
        <span class="mode-badge" style="background: rgba(99, 102, 241, 0.15); color: #c4b5fd;">
            Baseline Mean λ̂ = {analysis.lambda_hat:.4f} {analysis.measurement_unit}/int
        </span>
        <span class="mode-badge" style="background: rgba(99, 102, 241, 0.15); color: #c4b5fd;">
            Baseline Variance s² = {analysis.baseline_variance:.4f}
        </span>
        <span class="mode-badge" style="background: {'rgba(239, 68, 68, 0.2)' if analysis.baseline_dispersion > 1.5 else 'rgba(16, 185, 129, 0.15)'}; color: {'#fca5a5' if analysis.baseline_dispersion > 1.5 else '#6ee7b7'};">
            Baseline Dispersion D = {analysis.baseline_dispersion:.2f} ({'Overdispersed (Var > Mean)' if analysis.baseline_dispersion > 1.5 else ('Underdispersed' if analysis.baseline_dispersion < 0.5 else 'Equi-dispersed')})
        </span>
        <span class="mode-badge" style="background: rgba(148, 163, 184, 0.12); color: #cbd5e1;">
            Overall Dispersion D_all = {dispersion_index:.2f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # KPI Metrics Cards (6 metrics)
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric(
        "Total Observations",
        f"{len(analysis.full_data):,} int",
        help="Total number of discrete time intervals evaluated."
    )
    metric_kpi_label = "Total Flows" if metric_unit_label == "flows" else "Total Packets"
    kpi2.metric(
        metric_kpi_label,
        f"{analysis.total_packets:,}",
        help=f"Total {metric_unit_label} arrivals summed across all intervals."
    )
    kpi3.metric(
        "Baseline Mean (λ̂)",
        f"{analysis.lambda_hat:.2f} {metric_unit_label}/int",
        help=f"Exact baseline arrival rate: {analysis.lambda_hat:.4f} {metric_unit_label} per interval."
    )
    kpi4.metric(
        "Evaluated Intervals",
        f"{analysis.evaluated_count:,} int",
        help="Observations outside the baseline window evaluated against the Poisson model."
    )
    kpi5.metric(
        "Potential Bursts",
        f"{analysis.burst_count} intervals",
        help=f"Number of evaluated intervals with upper-tail probability p < α ({analysis.alpha:g})."
    )
    kpi6.metric(
        "Peak Evaluated Count",
        f"{analysis.peak_evaluated_count:,} {metric_unit_label}",
        help=f"Maximum {metric_unit_label} arrival count observed in the evaluation window."
    )

    # Interactive Plotly Chart
    st.subheader("📈 Traffic Time Series & Poisson Burst Classification")

    fig = go.Figure()

    # 1. Baseline trace
    base_df = analysis.baseline_data
    fig.add_trace(go.Scatter(
        x=base_df["interval"],
        y=base_df["packet_count"],
        mode="lines+markers",
        name="Baseline Partition (λ̂ Calibration)",
        line=dict(color="#4f46e5", width=2),
        marker=dict(size=4, color="#4f46e5"),
        hovertemplate="<b>Interval:</b> %{x}<br><b>Packets:</b> %{y}<br><b>Segment:</b> Baseline Partition<extra></extra>"
    ))

    # 2. Continuous Evaluated trace
    eval_df = analysis.evaluated_data
    fig.add_trace(go.Scatter(
        x=eval_df["interval"],
        y=eval_df["packet_count"],
        mode="lines+markers",
        name="Evaluated Observations",
        line=dict(color="#64748b", width=1.8),
        marker=dict(size=4, color="#64748b"),
        customdata=eval_df["upper_tail_probability"],
        hovertemplate="<b>Interval:</b> %{x}<br><b>Packets:</b> %{y}<br><b>P(X ≥ k):</b> %{customdata:.4e}<br><b>Segment:</b> Evaluated<extra></extra>"
    ))

    # 3. Flagged Bursts Overlay
    burst_points = eval_df[eval_df["is_burst"]]
    if not burst_points.empty:
        fig.add_trace(go.Scatter(
            x=burst_points["interval"],
            y=burst_points["packet_count"],
            mode="markers",
            name=f"Potential Burst (p < {alpha:g})",
            marker=dict(
                color="#ef4444",
                size=12,
                symbol="x",
                line=dict(width=2.5, color="#991b1b")
            ),
            customdata=burst_points["upper_tail_probability"],
            hovertemplate="<b>🚨 Potential Burst</b><br><b>Interval:</b> %{x}<br><b>Packets:</b> %{y}<br><b>P(X ≥ k):</b> %{customdata:.4e}<br><b>Threshold α:</b> " + str(alpha) + "<extra></extra>"
        ))

    # 4. Ground-truth / Injected attack overlay (when present)
    if analysis.has_labels and "has_attack_label" in eval_df.columns:
        attack_pts = eval_df[eval_df["has_attack_label"]]
        if not attack_pts.empty:
            attack_labels = attack_pts["attack_type"] if "attack_type" in attack_pts.columns else ["Anomalous Event"] * len(attack_pts)
            marker_name = "Simulation-Injected Label" if is_synthetic else "Ground-Truth Label"
            fig.add_trace(go.Scatter(
                x=attack_pts["interval"],
                y=attack_pts["packet_count"],
                mode="markers",
                name=marker_name,
                marker=dict(
                    color="#f59e0b",
                    size=10,
                    symbol="diamond-open",
                    line=dict(width=2.2, color="#d97706")
                ),
                customdata=attack_labels,
                hovertemplate=f"<b>⚠️ {marker_name}</b><br><b>Interval:</b> %{{x}}<br><b>Packets:</b> %{{y}}<br><b>Label:</b> %{{customdata}}<extra></extra>"
            ))

    # 5. Baseline Mean Reference Line
    fig.add_hline(
        y=analysis.lambda_hat,
        line_dash="dash",
        line_color="#059669",
        line_width=2,
        annotation_text=f"Baseline Mean λ̂ = {analysis.lambda_hat:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="#059669", size=11)
    )

    # 6. Shaded Baseline Rectangle
    last_base_interval = base_df["interval"].iloc[-1]
    first_base_interval = base_df["interval"].iloc[0]

    if isinstance(last_base_interval, (int, float, np.integer, np.floating)):
        fig.add_vrect(
            x0=first_base_interval - 0.5,
            x1=last_base_interval + 0.5,
            fillcolor="rgba(79, 70, 229, 0.06)",
            layer="below",
            line_width=0,
            annotation_text="Baseline Calibration Window",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#6366f1")
        )

        fig.add_vline(
            x=last_base_interval + 0.5,
            line_width=1.5,
            line_dash="dot",
            line_color="#818cf8",
            annotation_text="Partition Boundary",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color="#818cf8")
        )

    max_count = max(analysis.full_data["packet_count"].max(), int(analysis.lambda_hat * 1.5))
    y_limit = max_count * 1.15

    fig.update_layout(
        title=dict(
            text=f"<b>Interval {analysis.measurement_unit.capitalize()} Counts:</b> Baseline Calibration (λ̂) & Upper-Tail Burst Detection (p < α)",
            font=dict(size=15, color="#f8fafc"),
            y=0.96
        ),
        paper_bgcolor="#131826",
        plot_bgcolor="#0e131f",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#cbd5e1"),
        xaxis=dict(
            title=dict(text="Observation Interval (1-based index)", font=dict(color="#94a3b8", size=12)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            title=dict(text=f"{analysis.measurement_unit.capitalize()} Count", font=dict(color="#94a3b8", size=12)),
            range=[0, y_limit],
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color="#94a3b8")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
            font=dict(size=10.5, color="#cbd5e1")
        ),
        margin=dict(l=20, r=20, t=80, b=35),
        height=490,
        hovermode="closest"
    )

    st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------
    # UPPER-TAIL P-VALUE CHART WITH ALPHA THRESHOLD (Phase 4)
    # ---------------------------------------------------------
    st.subheader(f"🎯 Upper-Tail P-Value Evaluation & Significance Threshold (α = {alpha:g})")

    fig_pval = go.Figure()
    is_burst_mask = eval_df["is_burst"]

    # Non-burst evaluation points
    non_burst = eval_df[~is_burst_mask]
    if not non_burst.empty:
        fig_pval.add_trace(go.Scatter(
            x=non_burst["interval"],
            y=non_burst["upper_tail_probability"],
            mode="markers",
            name="Within Threshold (p ≥ α)",
            marker=dict(size=4.5, color="#64748b", opacity=0.75),
            customdata=non_burst["packet_count"],
            hovertemplate="<b>Interval:</b> %{x}<br><b>Count:</b> %{customdata}<br><b>P(X ≥ k):</b> %{y:.4e}<br><b>Status:</b> Within threshold<extra></extra>"
        ))

    # Flagged burst points
    burst_pts = eval_df[is_burst_mask]
    if not burst_pts.empty:
        fig_pval.add_trace(go.Scatter(
            x=burst_pts["interval"],
            y=burst_pts["upper_tail_probability"],
            mode="markers",
            name=f"Potential Burst (p < {alpha:g})",
            marker=dict(size=8, color="#ef4444", symbol="x", line=dict(width=1.8, color="#991b1b")),
            customdata=burst_pts["packet_count"],
            hovertemplate="<b>🚨 Potential Burst</b><br><b>Interval:</b> %{x}<br><b>Count:</b> %{customdata}<br><b>P(X ≥ k):</b> %{y:.4e}<br><b>Threshold α:</b> " + str(alpha) + "<extra></extra>"
        ))

    # Threshold horizontal reference line
    fig_pval.add_hline(
        y=alpha,
        line_dash="dash",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Significance Threshold α = {alpha:g}",
        annotation_position="top right",
        annotation_font=dict(color="#f87171", size=11)
    )

    fig_pval.update_layout(
        title=dict(
            text=f"<b>Upper-Tail Probability P(X ≥ k):</b> Evaluated Observations vs. Decision Threshold α = {alpha:g}",
            font=dict(size=14.5, color="#f8fafc")
        ),
        paper_bgcolor="#131826",
        plot_bgcolor="#0e131f",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#cbd5e1"),
        xaxis=dict(
            title=dict(text="Observation Interval (1-based index)", font=dict(color="#94a3b8", size=12)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color="#94a3b8")
        ),
        yaxis=dict(
            title=dict(text="Upper-Tail Probability P(X ≥ k) [Log Scale]", font=dict(color="#94a3b8", size=12)),
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zeroline=False,
            tickfont=dict(color="#94a3b8"),
            type="log"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
            font=dict(size=10.5, color="#cbd5e1")
        ),
        margin=dict(l=20, r=20, t=75, b=35),
        height=360,
        hovermode="closest"
    )

    st.plotly_chart(fig_pval, width="stretch")

    # ---------------------------------------------------------
    # CONTINGENCY MATRIX (ONLY WHEN LABELS EXIST)
    # ---------------------------------------------------------
    if analysis.has_labels and analysis.label_contingency:
        ctg = analysis.label_contingency
        label_title = "Simulation Evaluation vs Injected Labels" if is_synthetic else "Ground-Truth Evaluation"
        
        st.markdown(f"### 🎯 Empirical Label Comparison ({label_title})")
        st.caption("Recomputed dynamically from actual observations without hardcoding.")

        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("True Positives (Overlap)", f"{ctg['true_positives']} int", help="Intervals that contained attack/anomaly labels AND were flagged as bursts by the Poisson detector.")
        c_col2.metric("Statistical False Alarms", f"{ctg['false_positives']} int", help="Benign intervals flagged as bursts due to high arrival volume.")
        c_col3.metric("Missed Intervals", f"{ctg['false_negatives']} int", help="Labeled intervals whose volume did not exceed the statistical threshold.")
        c_col4.metric("Empirical Recall", f"{ctg['recall']:.1%}", help="Proportion of labeled intervals captured by the Poisson detector.")

        st.markdown(f"""
        <div class="contingency-card">
            <div style="font-size: 0.95rem; font-weight: 700; color: #f59e0b; margin-bottom: 6px;">Evaluation Metrics Summary</div>
            <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0;">
                <li><b>Precision:</b> <code>{ctg['precision']:.1%}</code> — {ctg['true_positives']} of {ctg['total_flagged']} flagged intervals coincided with simulated attack markers.</li>
                <li><b>Recall:</b> <code>{ctg['recall']:.1%}</code> — Captured {ctg['true_positives']} of {ctg['total_attacks']} labeled intervals.</li>
                <li><b>F1-Score:</b> <code>{ctg['f1_score']:.3f}</code> across {ctg['total_evaluated']} evaluated observations.</li>
                <li><b>Statistical Note:</b> Poisson upper-tail testing identifies <i>volume rarity</i>. In real networks, benign batch syncs produce false alarms, while stealthy low-rate probes blend into the baseline.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # RESULTS TABLE & STATISTICAL INTERPRETATION
    # ---------------------------------------------------------
    col_table, col_summary = st.columns([1.1, 0.9])

    with col_table:
        st.subheader("📋 Evaluation Results Table")
        filter_option = st.radio(
            "Display Filter",
            options=["All Evaluated Observations", "Flagged Bursts Only"],
            horizontal=True,
            key="analyzer_table_filter"
        )

        display_df = eval_df.copy()
        if filter_option == "Flagged Bursts Only":
            display_df = display_df[display_df["is_burst"]]

        if display_df.empty:
            st.info("No intervals match the selected filter (no bursts detected below threshold).")
        else:
            table_dict = {
                "Interval": display_df["interval"],
            }
            if "timestamp" in display_df.columns:
                table_dict["Timestamp"] = display_df["timestamp"]
            table_dict["Observed Count (k)"] = display_df["packet_count"]
            table_dict["Baseline (λ̂)"] = [f"{analysis.lambda_hat:.2f}"] * len(display_df)
            table_dict["Upper-Tail P(X ≥ k)"] = display_df["upper_tail_probability"].apply(
                lambda p: f"{p:.4e}" if p < 0.0001 else f"{p:.6f}"
            )
            table_dict["Detection Status"] = display_df["status"].apply(
                lambda s: "🚨 Potential burst" if s == "Potential burst" else "✅ Within threshold"
            )
            if "scenario_label" in display_df.columns:
                table_dict["Scenario Label"] = display_df["scenario_label"]
            elif "attack_type" in display_df.columns:
                table_dict["Scenario Label"] = display_df["attack_type"]
            elif "scenario" in display_df.columns:
                table_dict["Scenario Label"] = display_df["scenario"]

            table_dict["Statistical Evidence"] = [
                f"Surge k={row['packet_count']} ≥ λ̂={analysis.lambda_hat:.2f} (p={row['upper_tail_probability']:.2e} < α={alpha:g})"
                if row["is_burst"] else
                f"Normal k={row['packet_count']} ~ λ̂={analysis.lambda_hat:.2f} (p={row['upper_tail_probability']:.4f} ≥ α)"
                for _, row in display_df.iterrows()
            ]

            table_view = pd.DataFrame(table_dict)

            st.dataframe(
                table_view,
                width="stretch",
                hide_index=True,
                height=320
            )

    with col_summary:
        st.subheader("📊 Statistical Interpretation")
        if analysis.burst_count > 0:
            st.warning(
                f"**{analysis.burst_count} of {analysis.evaluated_count} evaluated intervals ({analysis.burst_percentage:.1f}%)** "
                f"yielded an upper-tail probability strictly below $\\alpha = {alpha:g}$.\n\n"
                "Under the null hypothesis of a stationary Poisson process with rate $\\hat{\\lambda}$, "
                "observing arrival volumes of this magnitude has a probability less than the chosen significance threshold."
            )
        else:
            st.success(
                f"**Zero evaluated intervals** fell below the significance threshold $\\alpha = {alpha:g}$.\n\n"
                "All observed packet counts in the evaluation window are consistent with ordinary random fluctuations around $\\hat{\\lambda}$."
            )

        # Overdispersion demonstration
        dispersion_text = (
            f"**Overdispersion Index ($D = \\sigma^2/\\mu$):** `{dispersion_index:.2f}` — "
            + ("Substantial overdispersion detected ($D > 1$), indicating heavy tails and variance clustering that challenge the Poisson equi-dispersion assumption."
               if dispersion_index > 1.5 else
               "Close to equi-dispersion ($D \\approx 1$), consistent with Poisson assumptions.")
        )
        st.markdown(dispersion_text)

        st.markdown(f"""
        **Fitted Model Parameters:**
        - **Distribution:** $\\text{{Poisson}}(\\hat{{\\lambda}})$
        - **Estimated Baseline Mean ($\\hat{{\\lambda}}$):** `{analysis.lambda_hat:.4f}` {analysis.measurement_unit} / interval
        - **Baseline Sample Variance ($s^2$):** `{analysis.baseline_variance:.4f}`
        - **Significance Level ($\\alpha$):** `{analysis.alpha}`
        - **Baseline Partition ($m$):** `{analysis.baseline_count}` intervals ({analysis.baseline_pct}%)
        - **Evaluated Partition:** `{analysis.evaluated_count}` intervals
        """)

        st.info(
            "**Key Modeling Insights:**\n"
            "1. **Volume Rarity vs Malicious Intent:** Statistical burst detection identifies unusual volume, not confirmed malicious intent.\n"
            "2. **Stationarity:** $\\lambda$ is assumed constant during the baseline window.\n"
            "3. **Independence:** Real TCP network traffic exhibits correlation and packet trains, causing heavier tails.\n"
            "4. **Multiple Testing:** Successive testing across many intervals inflates family-wise false alarms."
        )

        # Experimental Negative Binomial comparison expander
        if analysis.experimental_nb:
            with st.expander("🔬 Model Contrast: Negative Binomial vs. Poisson", expanded=False):
                st.markdown(r"""
                **Theoretical Motivation:**  
                The Poisson model assumes equi-dispersion ($\mathbb{E}[X] = \operatorname{Var}(X) = \lambda$). When network traffic exhibits variance clustering or burstiness ($s^2 > \hat{\lambda}$), standard Poisson upper-tail probabilities can flag natural variance fluctuations as anomalies.
                
                The **Negative Binomial** distribution introduces a dispersion parameter $r$ to accommodate heavy-tailed variance without confusing ordinary variance with surges:
                """)
                nb = analysis.experimental_nb
                nb_col1, nb_col2, nb_col3 = st.columns(3)
                nb_col1.metric("Poisson Bursts (p < α)", f"{analysis.burst_count}")
                nb_col2.metric("Negative Binomial Bursts", f"{nb['burst_count']}")
                nb_col3.metric("NB Dispersion (r)", f"{nb['dispersion_r']:.2f}")
                st.info(f"💡 **Comparative Finding:** {nb['interpretation']}")
                st.caption("Note: Results from the Negative Binomial model are provided strictly for statistical comparison and are kept distinct from the primary Poisson engine.")

    # ---------------------------------------------------------
    # CSV EXPORT
    # ---------------------------------------------------------
    st.divider()
    st.subheader("💾 Export Analysis Results")

    export_data = prepare_export_dataframe(analysis)
    export_data["data_source"] = source_label
    export_data["is_synthetic"] = is_synthetic

    csv_bytes = export_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Complete Analysis CSV",
        data=csv_bytes,
        file_name=export_filename,
        mime="text/csv",
        key="btn_download_analysis_csv",
        help="Download complete dataset with intervals, packet counts, baseline lambda, p-values, burst flags, and metadata."
    )

    # Mathematical Foundations expander
    with st.expander("📖 Mathematical Foundations & Poisson Derivation", expanded=False):
        st.markdown("#### 1. Poisson Distribution Formulation")
        st.write("Arrival counts over fixed equal observation intervals are modeled using the Poisson probability mass function (PMF):")
        st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k \in \{0, 1, 2, \dots\}")
        st.markdown(r"""
        - **$X$:** Discrete random variable representing arrival count in a fixed observation interval.
        - **$k$:** Observed nonnegative integer count ($k \ge 0$).
        - **$\lambda$:** Expected arrivals per interval ($\mathbb{E}[X] = \operatorname{Var}(X) = \lambda$).
        """)

        st.markdown("#### 2. Baseline Rate Estimation")
        st.write(r"Given an initial baseline partition containing $m$ observations $x_1, x_2, \dots, x_m$:")
        st.latex(r"\hat{\lambda} = \frac{1}{m} \sum_{i=1}^{m} x_i")
        st.caption(r"The baseline arrival rate $\hat{\lambda}$ is estimated strictly from baseline intervals and is completely isolated from the evaluation segment.")

        st.markdown("#### 3. Upper-Tail Cumulative Survival Probability")
        st.write("Probability of observing a count at least as large as measured count $k$ under the fitted baseline model:")
        st.latex(r"p = P(X \ge k) = 1 - P(X \le k - 1)")
        st.latex(r"p = \sum_{j=k}^{\infty} \frac{\lambda^j e^{-\lambda}}{j!}")
        st.markdown(r"""
        Computed stably via SciPy's survival function:
        ```python
        p = scipy.stats.poisson.sf(k - 1, lambda_hat)
        ```
        """)

        st.markdown("#### 4. Classification Decision Rule")
        st.latex(r"\text{Flag as potential burst if } p < \alpha")
        st.markdown(r"""
        - If $P(X \ge k) < \alpha \implies$ **Flag as Potential Burst** (volume is statistically improbable under baseline $\hat{\lambda}$)
        - If $P(X \ge k) \ge \alpha \implies$ **Within Normal Threshold** (consistent with ordinary Poisson fluctuations)
        """)


def main():
    """Main application routing controller."""
    query_page = st.query_params.get("page")
    query_ds = st.query_params.get("dataset")
    if query_ds and "selected_dataset_id" not in st.session_state:
        st.session_state["selected_dataset_id"] = query_ds

    if query_page in ["landing", "analyzer"]:
        st.session_state["current_page"] = query_page
    else:
        st.session_state["current_page"] = "landing"

    if st.session_state["current_page"] == "analyzer":
        render_analyzer_page()
    else:
        render_landing_page(on_navigate_to_analyzer=lambda ds=None: navigate_to("analyzer", dataset_id=ds))


if __name__ == "__main__":
    main()
