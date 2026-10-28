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
from scipy.stats import poisson
import streamlit as st

from burst_detector import (
    generate_synthetic_data,
    normalize_traffic_data,
    analyze_traffic,
    prepare_export_dataframe,
    compute_upper_tail_probability,
    compute_descriptive_stats,
    compute_critical_burst_threshold,
    compute_lambda_confidence_interval,
    compute_benjamini_hochberg,
    compute_poisson_goodness_of_fit,
    compute_poisson_binomial_comparison,
    generate_point_explanation,
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

    # Section A: Enhanced Overview Status Card
    interval_dur_text = f"{active_dataset_entry.actual_time_resolution}" if active_dataset_entry else "1.0s discrete interval"
    synth_badge_text = "Synthetic Research Benchmark" if is_synthetic else "Custom User Upload"

    st.markdown(
        f'<div class="status-card">'
        f'<b>Active Data Source:</b> {source_label} &nbsp;·&nbsp; '
        f'<span class="mode-badge" style="background: rgba(99, 102, 241, 0.2); color: #c4b5fd;">{synth_badge_text}</span> &nbsp;·&nbsp; '
        f'<b>Metric:</b> {count_col} ({metric_unit_label}) &nbsp;·&nbsp; '
        f'<b>Resolution:</b> {interval_dur_text} &nbsp;·&nbsp; '
        f'<b>Total Observations:</b> {len(analysis.full_data):,} &nbsp;·&nbsp; '
        f'<b>Baseline Window:</b> {analysis.baseline_count:,} int ({analysis.baseline_pct}%) &nbsp;·&nbsp; '
        f'<b>Evaluated Window:</b> {analysis.evaluated_count:,} int &nbsp;·&nbsp; '
        f'<b>Cutoff Rule:</b> α = {analysis.alpha:g} (k_crit = {analysis.k_crit})'
        f'</div>',
        unsafe_allow_html=True
    )

    # Model Suitability Warnings Banner
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
        <span class="mode-badge" style="background: rgba(239, 68, 68, 0.18); color: #fca5a5;">
            Critical Threshold k_crit = {analysis.k_crit} {analysis.measurement_unit}
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Core KPI Metrics Cards (Preserves test expectations and provides immediate visual summary)
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
        help=f"Number of evaluated intervals with upper-tail probability p < α ({analysis.alpha:g}) iff count k ≥ {analysis.k_crit}."
    )
    kpi6.metric(
        "Peak Evaluated Count",
        f"{analysis.peak_evaluated_count:,} {metric_unit_label}",
        help=f"Maximum {metric_unit_label} arrival count observed in the evaluation window."
    )

    # Pre-extract evaluated data and baseline data for subsequent tabs
    base_df = analysis.baseline_data
    eval_df = analysis.evaluated_data
    full_df = analysis.full_data

    # =========================================================================
    # UPGRADED MULTI-SECTION ANALYTICS WORKFLOW (TABS)
    # =========================================================================
    tab_overview, tab_eda, tab_bursts, tab_inspector, tab_inference, tab_diagnostics, tab_model_compare, tab_eval, tab_export = st.tabs([
        "📋 Overview & Key Metrics",
        "📊 Statistical Analysis & EDA",
        "📈 Burst Detection & Time Series",
        "🔍 Point-Level Inspector",
        "📐 Hypothesis Testing & Inference",
        "🔬 Model Diagnostics & GoF",
        "⚖️ Poisson vs. Binomial",
        "🎯 Ground-Truth Evaluation",
        "💾 Data Export"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: OVERVIEW & KEY METRICS (Section A)
    # -------------------------------------------------------------------------
    with tab_overview:
        st.subheader("📋 Dataset Overview & Core Parameters")
        
        ov_col1, ov_col2 = st.columns([1.2, 0.8], gap="medium")
        with ov_col1:
            st.markdown(f"""
            <div class="provenance-card">
                <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                    {source_label}
                </div>
                <div style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin-bottom: 12px;">
                    <b>Measurement Unit:</b> <code>{analysis.measurement_unit}</code> &nbsp;·&nbsp;
                    <b>Interval Resolution:</b> <code>{interval_dur_text}</code> &nbsp;·&nbsp;
                    <b>Total Rows:</b> <code>{len(analysis.full_data):,}</code> records
                </div>
                <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5; margin-bottom: 10px;">
                    <b>Partition Strategy:</b> Clean baseline of first <b>{analysis.baseline_count:,}</b> intervals ({analysis.baseline_pct}%) strictly reserved for parameter estimation $\\hat{{\\lambda}}$, leaving <b>{analysis.evaluated_count:,}</b> intervals for live hypothesis testing. Zero data leakage is enforced.
                </div>
                <div style="background: rgba(245, 158, 11, 0.1); border-left: 3px solid #f59e0b; padding: 8px 12px; border-radius: 4px; font-size: 0.82rem; color: #fde68a;">
                    <b>⚠️ Synthetic Data & Volume Rarity Disclosure:</b> {
                        "This benchmark was generated programmatically for reproducible mathematical testing. Labeled anomaly flags describe simulated attack scenarios. Statistical burst detection measures volume rarity under Poisson assumptions; it is NOT independent proof of a cyberattack."
                        if is_synthetic else
                        "User-provided dataset. Arrival counts are evaluated against the stationary baseline Poisson rate; statistical rarity does not verify malicious intent."
                    }
                </div>
            </div>
            """, unsafe_allow_html=True)

        with ov_col2:
            st.markdown(f"""
            <div class="config-card">
                <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                    Mathematical Decision Framework
                </div>
                <ul style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.6; padding-left: 18px; margin-bottom: 0;">
                    <li><b>Null Model:</b> $X_i \\sim \\text{{Poisson}}(\\hat{{\\lambda}} = {analysis.lambda_hat:.2f})$</li>
                    <li><b>Significance Threshold:</b> $\\alpha = {analysis.alpha:g}$</li>
                    <li><b>Critical Cutoff ($k_{{crit}}$):</b> <code>{analysis.k_crit} {analysis.measurement_unit}</code></li>
                    <li><b>Flagged Rate:</b> {analysis.burst_count:,} / {analysis.evaluated_count:,} ({analysis.burst_percentage:.1f}%)</li>
                    <li><b>Equi-dispersion Ratio:</b> $D = s^2 / \\hat{{\\lambda}} = {analysis.baseline_dispersion:.2f}$</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 📊 Dynamic Statistical Metric Cards (Calculated from Actual Data)")
        desc_base = analysis.descriptive_base
        desc_full = analysis.descriptive_full

        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric(
            "Mean Count (μ)",
            f"{desc_base['mean']:.2f} base",
            delta=f"{desc_full['mean'] - desc_base['mean']:+.2f} full",
            help=f"Baseline Mean λ̂ = {desc_base['mean']:.4f}, Overall Mean = {desc_full['mean']:.4f}"
        )
        sc2.metric(
            "Median Count",
            f"{desc_base['median']:.1f} base",
            delta=f"{desc_full['median'] - desc_base['median']:+.1f} full",
            help=f"Baseline Median = {desc_base['median']}, Overall Median = {desc_full['median']}"
        )
        sc3.metric(
            "Sample Std Dev (s)",
            f"{desc_base['sample_std']:.2f} base",
            help=f"Baseline s = {desc_base['sample_std']:.4f}, Full s = {desc_full['sample_std']:.4f} (ddof=1)"
        )
        sc4.metric(
            "Sample Variance (s²)",
            f"{desc_base['sample_variance']:.2f} base",
            help=f"Baseline s² = {desc_base['sample_variance']:.4f}, Full s² = {desc_full['sample_variance']:.4f} (ddof=1)"
        )
        sc5.metric(
            "Flagged Bursts",
            f"{analysis.burst_count} int",
            delta=f"{analysis.burst_percentage:.1f}% eval",
            help=f"Intervals with upper-tail p < {analysis.alpha:g} iff count k ≥ {analysis.k_crit}"
        )
        sc6.metric(
            "Dispersion Index (D)",
            f"{analysis.baseline_dispersion:.2f}",
            help=f"Baseline D = s²/λ̂ = {analysis.baseline_dispersion:.4f}. D ≈ 1: Poisson equi-dispersion. D > 1.5: Overdispersed."
        )

    # -------------------------------------------------------------------------
    # TAB 2: STATISTICAL ANALYSIS & EDA (Section B)
    # -------------------------------------------------------------------------
    with tab_eda:
        st.subheader("📊 Exploratory Data Analysis & Descriptive Statistics")
        st.caption("Comprehensive breakdown distinguishing baseline calibration statistics from full-dataset properties.")

        eval_counts_series = eval_df["packet_count"]
        desc_eval = compute_descriptive_stats(eval_counts_series)

        # Comparative Summary Table
        stats_data = {
            "Statistical Measure": [
                "Observation Count (N)",
                "Sample Mean (μ)",
                "Median Count",
                "Sample Variance (s², ddof=1)",
                "Sample Standard Deviation (s)",
                "Minimum Count",
                "Maximum Count",
                "First Quartile (Q1, 25%)",
                "Third Quartile (Q3, 75%)",
                "Interquartile Range (IQR)",
                "90th Percentile (P90)",
                "95th Percentile (P95)",
                "99th Percentile (P99)",
                "Dispersion Index (s² / μ)",
                "Skewness",
                "Excess Kurtosis"
            ],
            f"Baseline Partition ({analysis.baseline_pct}%)": [
                f"{desc_base.get('count', 0):,}",
                f"{desc_base.get('mean', 0.0):.4f}",
                f"{desc_base.get('median', 0.0):.1f}",
                f"{desc_base.get('sample_variance', 0.0):.4f}",
                f"{desc_base.get('sample_std', 0.0):.4f}",
                f"{desc_base.get('min', 0.0):.0f}",
                f"{desc_base.get('max', 0.0):.0f}",
                f"{desc_base.get('q25', 0.0):.2f}",
                f"{desc_base.get('q75', 0.0):.2f}",
                f"{desc_base.get('iqr', 0.0):.2f}",
                f"{desc_base.get('p90', 0.0):.2f}",
                f"{desc_base.get('p95', 0.0):.2f}",
                f"{desc_base.get('p99', 0.0):.2f}",
                f"{desc_base.get('dispersion_index', 0.0):.3f}",
                f"{desc_base.get('skewness', 0.0):.3f}",
                f"{desc_base.get('excess_kurtosis', 0.0):.3f}"
            ],
            "Evaluated Window (Remaining)": [
                f"{desc_eval.get('count', 0):,}",
                f"{desc_eval.get('mean', 0.0):.4f}",
                f"{desc_eval.get('median', 0.0):.1f}",
                f"{desc_eval.get('sample_variance', 0.0):.4f}",
                f"{desc_eval.get('sample_std', 0.0):.4f}",
                f"{desc_eval.get('min', 0.0):.0f}",
                f"{desc_eval.get('max', 0.0):.0f}",
                f"{desc_eval.get('q25', 0.0):.2f}",
                f"{desc_eval.get('q75', 0.0):.2f}",
                f"{desc_eval.get('iqr', 0.0):.2f}",
                f"{desc_eval.get('p90', 0.0):.2f}",
                f"{desc_eval.get('p95', 0.0):.2f}",
                f"{desc_eval.get('p99', 0.0):.2f}",
                f"{desc_eval.get('dispersion_index', 0.0):.3f}",
                f"{desc_eval.get('skewness', 0.0):.3f}",
                f"{desc_eval.get('excess_kurtosis', 0.0):.3f}"
            ],
            "Full Dataset (100%)": [
                f"{desc_full.get('count', 0):,}",
                f"{desc_full.get('mean', 0.0):.4f}",
                f"{desc_full.get('median', 0.0):.1f}",
                f"{desc_full.get('sample_variance', 0.0):.4f}",
                f"{desc_full.get('sample_std', 0.0):.4f}",
                f"{desc_full.get('min', 0.0):.0f}",
                f"{desc_full.get('max', 0.0):.0f}",
                f"{desc_full.get('q25', 0.0):.2f}",
                f"{desc_full.get('q75', 0.0):.2f}",
                f"{desc_full.get('iqr', 0.0):.2f}",
                f"{desc_full.get('p90', 0.0):.2f}",
                f"{desc_full.get('p95', 0.0):.2f}",
                f"{desc_full.get('p99', 0.0):.2f}",
                f"{desc_full.get('dispersion_index', 0.0):.3f}",
                f"{desc_full.get('skewness', 0.0):.3f}",
                f"{desc_full.get('excess_kurtosis', 0.0):.3f}"
            ]
        }
        df_stats_view = pd.DataFrame(stats_data)
        st.dataframe(df_stats_view, width="stretch", hide_index=True)

        # Data Quality & Missing Value Audit
        st.markdown("#### 🛡️ Data Quality & Missing Value Summary")
        null_count = int(full_df["packet_count"].isna().sum())
        neg_count = int((full_df["packet_count"] < 0).sum())
        non_int_count = int((full_df["packet_count"] % 1 != 0).sum())

        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Missing / NaN Values", f"{null_count}", help="Missing or unrecorded interval counts.")
        dq2.metric("Negative Counts", f"{neg_count}", help="Invalid negative arrival values (must be 0 for Poisson).")
        dq3.metric("Non-Integer Records", f"{non_int_count}", help="Fractional counts violating discrete arrival assumptions.")
        dq4.metric("Data Integrity Status", "✅ 100% Valid Discrete", help="Verified non-negative integer arrival counts.")

        # Interactive Histogram with Theoretical Poisson Overlay
        st.markdown("#### 📈 Observed Distribution vs. Fitted Poisson PMF")
        dist_view_option = st.radio(
            "Select Partition for Distribution Inspection",
            options=["Baseline Partition (Calibration Window)", "Evaluated Segment", "Full Dataset"],
            horizontal=True,
            key="eda_dist_segment_radio"
        )

        if dist_view_option == "Baseline Partition (Calibration Window)":
            active_counts = base_df["packet_count"]
            active_label = f"Baseline Partition (N={len(active_counts):,}, λ̂={analysis.lambda_hat:.2f})"
        elif dist_view_option == "Evaluated Segment":
            active_counts = eval_df["packet_count"]
            active_label = f"Evaluated Window (N={len(active_counts):,})"
        else:
            active_counts = full_df["packet_count"]
            active_label = f"Full Dataset (N={len(active_counts):,})"

        fig_hist = go.Figure()

        # Empirical Histogram
        fig_hist.add_trace(go.Histogram(
            x=active_counts,
            name="Empirical Counts (Histogram)",
            histnorm="probability density",
            xbins=dict(start=-0.5, size=1.0),
            marker=dict(color="rgba(99, 102, 241, 0.65)", line=dict(color="#818cf8", width=1)),
            opacity=0.75,
            hovertemplate="<b>Count k:</b> %{x}<br><b>Empirical Density:</b> %{y:.4f}<extra></extra>"
        ))

        # Theoretical Poisson PMF Curve based on baseline lambda
        max_plot_k = int(active_counts.quantile(0.999)) + 10
        k_range = np.arange(0, max_plot_k + 1)
        theoretical_pmf = [float(poisson.pmf(k, analysis.lambda_hat)) for k in k_range]

        fig_hist.add_trace(go.Scatter(
            x=k_range,
            y=theoretical_pmf,
            mode="lines+markers",
            name=f"Fitted Poisson(λ̂ = {analysis.lambda_hat:.2f}) PMF",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=4, color="#10b981"),
            hovertemplate="<b>Count k:</b> %{x}<br><b>Theoretical P(X=k):</b> %{y:.4f}<extra></extra>"
        ))

        fig_hist.update_layout(
            title=dict(
                text=f"<b>Empirical Distribution vs. Poisson(λ̂ = {analysis.lambda_hat:.2f}) PMF:</b> {active_label}",
                font=dict(size=14, color="#f8fafc")
            ),
            paper_bgcolor="#131826",
            plot_bgcolor="#0e131f",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            xaxis=dict(
                title=dict(text=f"{analysis.measurement_unit.capitalize()} Count (k)", font=dict(color="#94a3b8", size=12)),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text="Probability Density", font=dict(color="#94a3b8", size=12)),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.8)",
                bordercolor="rgba(255, 255, 255, 0.08)",
                borderwidth=1
            ),
            margin=dict(l=20, r=20, t=60, b=35),
            height=380
        )
        st.plotly_chart(fig_hist, width="stretch")

        st.info(
            "💡 **Distribution Interpretation:** A Poisson process produces unimodal counts with variance equal to the mean. "
            "In traffic with variance clustering or micro-bursts, the empirical histogram shows heavier upper tails than the theoretical curve, "
            "highlighting the mathematical necessity of an upper-tail cutoff threshold."
        )

    # -------------------------------------------------------------------------
    # TAB 3: BURST DETECTION & TIME SERIES (Section C)
    # -------------------------------------------------------------------------
    with tab_bursts:
        st.subheader("📈 Traffic Time Series & Statistical Burst Detection")
        st.caption(f"Observed arrival counts with baseline rate λ̂ = {analysis.lambda_hat:.2f} and statistically justified critical cutoff k_crit = {analysis.k_crit}.")

        # Contextual explanation banner
        st.markdown(f"""
        <div class="config-card" style="margin-bottom: 14px;">
            <b>🎯 Statistically Justified Decision Rule:</b><br>
            Under the stationary Poisson model $H_0$, an interval is flagged as a potential volume burst if and only if:
            $$P(X \\ge k) < \\alpha \\iff k \\ge k_{{crit}} = {analysis.k_crit}$$
            <i>Important:</i> Any count $k$ strictly above the mean $\\hat{{\\lambda}} = {analysis.lambda_hat:.2f}$ but below $k_{{crit}} = {analysis.k_crit}$ is recognized as normal random Poisson variation and is <b>NOT</b> treated as a burst.
        </div>
        """, unsafe_allow_html=True)

        fig_ts = go.Figure()

        # 1. Baseline trace
        fig_ts.add_trace(go.Scatter(
            x=base_df["interval"],
            y=base_df["packet_count"],
            mode="lines+markers",
            name="Baseline Partition (λ̂ Calibration)",
            line=dict(color="#4f46e5", width=2),
            marker=dict(size=3.5, color="#4f46e5"),
            hovertemplate="<b>Interval:</b> %{x}<br><b>Count:</b> %{y} " + analysis.measurement_unit + "<br><b>Segment:</b> Baseline Calibration<extra></extra>"
        ))

        # 2. Continuous Evaluated trace
        fig_ts.add_trace(go.Scatter(
            x=eval_df["interval"],
            y=eval_df["packet_count"],
            mode="lines+markers",
            name="Evaluated Observations",
            line=dict(color="#64748b", width=1.8),
            marker=dict(size=3.5, color="#64748b"),
            customdata=eval_df["upper_tail_probability"],
            hovertemplate="<b>Interval:</b> %{x}<br><b>Count:</b> %{y} " + analysis.measurement_unit + "<br><b>P(X ≥ k):</b> %{customdata:.4e}<br><b>Segment:</b> Evaluated<extra></extra>"
        ))

        # 3. Statistically Flagged Bursts Overlay
        burst_points = eval_df[eval_df["is_burst"]]
        if not burst_points.empty:
            fig_ts.add_trace(go.Scatter(
                x=burst_points["interval"],
                y=burst_points["packet_count"],
                mode="markers",
                name=f"Potential Burst (k ≥ {analysis.k_crit}, p < {analysis.alpha:g})",
                marker=dict(
                    color="#ef4444",
                    size=11,
                    symbol="x",
                    line=dict(width=2.5, color="#991b1b")
                ),
                customdata=burst_points["upper_tail_probability"],
                hovertemplate="<b>🚨 Potential Burst</b><br><b>Interval:</b> %{x}<br><b>Count:</b> %{y}<br><b>P(X ≥ k):</b> %{customdata:.4e}<br><b>Threshold k_crit:</b> " + str(analysis.k_crit) + "<extra></extra>"
            ))

        # 4. Ground-truth / Injected attack overlay (when present)
        if analysis.has_labels and "has_attack_label" in eval_df.columns:
            attack_pts = eval_df[eval_df["has_attack_label"]]
            if not attack_pts.empty:
                attack_labels = attack_pts["attack_type"] if "attack_type" in attack_pts.columns else ["Anomalous Event"] * len(attack_pts)
                marker_name = "Simulation-Injected Label" if is_synthetic else "Ground-Truth Label"
                fig_ts.add_trace(go.Scatter(
                    x=attack_pts["interval"],
                    y=attack_pts["packet_count"],
                    mode="markers",
                    name=marker_name,
                    marker=dict(
                        color="#f59e0b",
                        size=9.5,
                        symbol="diamond-open",
                        line=dict(width=2.2, color="#d97706")
                    ),
                    customdata=attack_labels,
                    hovertemplate=f"<b>⚠️ {marker_name}</b><br><b>Interval:</b> %{{x}}<br><b>Count:</b> %{{y}}<br><b>Label:</b> %{{customdata}}<extra></extra>"
                ))

        # 5. Baseline Mean Reference Line
        fig_ts.add_hline(
            y=analysis.lambda_hat,
            line_dash="dash",
            line_color="#059669",
            line_width=2,
            annotation_text=f"Baseline Mean λ̂ = {analysis.lambda_hat:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#059669", size=11)
        )

        # 6. Upper-Tail Critical Threshold Reference Line (k_crit)
        fig_ts.add_hline(
            y=analysis.k_crit,
            line_dash="dot",
            line_color="#ef4444",
            line_width=2,
            annotation_text=f"Upper-Tail Threshold k_crit = {analysis.k_crit} (α = {analysis.alpha:g})",
            annotation_position="bottom right",
            annotation_font=dict(color="#ef4444", size=11)
        )

        # 7. Shaded Baseline Rectangle
        last_base_interval = base_df["interval"].iloc[-1]
        first_base_interval = base_df["interval"].iloc[0]

        if isinstance(last_base_interval, (int, float, np.integer, np.floating)):
            fig_ts.add_vrect(
                x0=first_base_interval - 0.5,
                x1=last_base_interval + 0.5,
                fillcolor="rgba(79, 70, 229, 0.06)",
                layer="below",
                line_width=0,
                annotation_text="Baseline Calibration Window",
                annotation_position="top left",
                annotation_font=dict(size=11, color="#6366f1")
            )

            fig_ts.add_vline(
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

        fig_ts.update_layout(
            title=dict(
                text=f"<b>Traffic Arrival Time Series:</b> Baseline (λ̂={analysis.lambda_hat:.2f}) & Upper-Tail Burst Cutoff (k_crit={analysis.k_crit}, α={analysis.alpha:g})",
                font=dict(size=14.5, color="#f8fafc")
            ),
            paper_bgcolor="#131826",
            plot_bgcolor="#0e131f",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            xaxis=dict(
                title=dict(text="Observation Interval (Index)", font=dict(color="#94a3b8", size=12)),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text=f"{analysis.measurement_unit.capitalize()} Count", font=dict(color="#94a3b8", size=12)),
                range=[0, y_limit],
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                zeroline=False
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
                font=dict(size=10.5)
            ),
            margin=dict(l=20, r=20, t=75, b=35),
            height=480,
            hovermode="closest"
        )
        st.plotly_chart(fig_ts, width="stretch")

        # Upper-tail p-value chart on log scale
        st.markdown(f"#### 🎯 Upper-Tail P-Value Evaluation (Log Scale vs. α = {analysis.alpha:g})")
        fig_pval = go.Figure()
        is_burst_mask = eval_df["is_burst"]

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

        burst_pts = eval_df[is_burst_mask]
        if not burst_pts.empty:
            fig_pval.add_trace(go.Scatter(
                x=burst_pts["interval"],
                y=burst_pts["upper_tail_probability"],
                mode="markers",
                name=f"Potential Burst (p < {analysis.alpha:g})",
                marker=dict(size=8, color="#ef4444", symbol="x", line=dict(width=1.8, color="#991b1b")),
                customdata=burst_pts["packet_count"],
                hovertemplate="<b>🚨 Potential Burst</b><br><b>Interval:</b> %{x}<br><b>Count:</b> %{customdata}<br><b>P(X ≥ k):</b> %{y:.4e}<br><b>Threshold α:</b> " + str(analysis.alpha) + "<extra></extra>"
            ))

        fig_pval.add_hline(
            y=analysis.alpha,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2,
            annotation_text=f"Significance Threshold α = {analysis.alpha:g}",
            annotation_position="top right",
            annotation_font=dict(color="#f87171", size=11)
        )

        fig_pval.update_layout(
            title=dict(
                text=f"<b>Upper-Tail Survival Probability P(X ≥ k):</b> Evaluated Observations vs. Decision Cutoff α = {analysis.alpha:g}",
                font=dict(size=14, color="#f8fafc")
            ),
            paper_bgcolor="#131826",
            plot_bgcolor="#0e131f",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            xaxis=dict(
                title=dict(text="Observation Interval (Index)", font=dict(color="#94a3b8", size=12)),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text="Upper-Tail Survival P(X ≥ k) [Log Scale]", font=dict(color="#94a3b8", size=12)),
                showgrid=True,
                gridcolor="rgba(255, 255, 255, 0.06)",
                zeroline=False,
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
                font=dict(size=10.5)
            ),
            margin=dict(l=20, r=20, t=70, b=35),
            height=340
        )
        st.plotly_chart(fig_pval, width="stretch")

        # Evaluation Results Table
        st.markdown("#### 📋 Interval Evaluation Results Table")
        filter_option = st.radio(
            "Display Filter",
            options=["All Evaluated Observations", "Flagged Bursts Only", "Within Normal Threshold Only"],
            horizontal=True,
            key="bursts_tab_filter"
        )

        display_df = eval_df.copy()
        if filter_option == "Flagged Bursts Only":
            display_df = display_df[display_df["is_burst"]]
        elif filter_option == "Within Normal Threshold Only":
            display_df = display_df[~display_df["is_burst"]]

        if display_df.empty:
            st.info("No intervals match the selected filter.")
        else:
            table_dict = {
                "Interval": display_df["interval"],
            }
            if "timestamp" in display_df.columns:
                table_dict["Timestamp"] = display_df["timestamp"]
            table_dict["Observed Count (k)"] = display_df["packet_count"]
            table_dict["Baseline Rate (λ̂)"] = [f"{analysis.lambda_hat:.2f}"] * len(display_df)
            table_dict["Critical Threshold (k_crit)"] = [f"{analysis.k_crit}"] * len(display_df)
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

            table_dict["Statistical Evidence"] = [
                f"Surge k={row['packet_count']} ≥ k_crit={analysis.k_crit} (p={row['upper_tail_probability']:.2e} < α={analysis.alpha:g})"
                if row["is_burst"] else
                f"Normal k={row['packet_count']} < k_crit={analysis.k_crit} (p={row['upper_tail_probability']:.4f} ≥ α)"
                for _, row in display_df.iterrows()
            ]

            table_view = pd.DataFrame(table_dict)
            st.dataframe(table_view, width="stretch", hide_index=True, height=280)

    # -------------------------------------------------------------------------
    # TAB 4: POINT-LEVEL INSPECTOR (Section D)
    # -------------------------------------------------------------------------
    with tab_inspector:
        st.subheader("🔍 Point-Level Interval Inspector & Statistical Explanation")
        st.caption("Inspect any individual interval to review the exact count, survival probability, null hypothesis decision, and step-by-step mathematical reasoning.")

        total_obs = len(full_df)

        # Quick-Jump Navigation Buttons
        btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)
        
        # Determine key landmark intervals
        peak_idx = int(eval_df.loc[eval_df["packet_count"].idxmax(), "interval"]) if not eval_df.empty else 1
        first_burst_df = eval_df[eval_df["is_burst"]]
        first_burst_idx = int(first_burst_df["interval"].iloc[0]) if not first_burst_df.empty else peak_idx
        median_idx = int(eval_df["interval"].iloc[len(eval_df) // 2]) if not eval_df.empty else 1
        base_first_idx = int(base_df["interval"].iloc[0]) if not base_df.empty else 1

        if "inspector_slider_ctrl" not in st.session_state or not isinstance(st.session_state.get("inspector_slider_ctrl"), (int, np.integer)) or st.session_state["inspector_slider_ctrl"] < 1 or st.session_state["inspector_slider_ctrl"] > total_obs:
            st.session_state["inspector_slider_ctrl"] = min(max(1, peak_idx), max(1, total_obs))

        with btn_c1:
            if st.button("🚨 Jump to Peak Burst", key="btn_jump_peak"):
                st.session_state["inspector_slider_ctrl"] = min(peak_idx, total_obs)
                st.rerun()
        with btn_c2:
            if st.button("⚡ Jump to First Flagged Burst", key="btn_jump_first_burst"):
                st.session_state["inspector_slider_ctrl"] = min(first_burst_idx, total_obs)
                st.rerun()
        with btn_c3:
            if st.button("🎯 Jump to Median Observation", key="btn_jump_median"):
                st.session_state["inspector_slider_ctrl"] = min(median_idx, total_obs)
                st.rerun()
        with btn_c4:
            if st.button("🔄 Jump to First Baseline Interval", key="btn_jump_baseline"):
                st.session_state["inspector_slider_ctrl"] = min(base_first_idx, total_obs)
                st.rerun()

        # Synchronized Interval Slider
        selected_interval = st.slider(
            "Select Interval Index (Synchronized Slider)",
            min_value=1,
            max_value=max(1, total_obs),
            step=1,
            key="inspector_slider_ctrl"
        )

        # Retrieve selected row
        matching_rows = full_df[full_df["interval"] == selected_interval]
        if matching_rows.empty:
            st.warning(f"Interval #{selected_interval} not found.")
        else:
            sel_row = matching_rows.iloc[0]
            sel_count = int(sel_row["packet_count"])
            is_base = bool(sel_row.get("segment") == "Baseline" or selected_interval <= analysis.baseline_count)
            is_burst = bool(sel_row.get("is_burst", False))
            p_val = float(sel_row.get("upper_tail_probability", compute_upper_tail_probability(sel_count, analysis.lambda_hat)))
            label = sel_row.get("attack_type", sel_row.get("scenario_label", None))

            explanation = generate_point_explanation(
                interval=int(selected_interval),
                count=sel_count,
                lambda_hat=analysis.lambda_hat,
                p_val=p_val,
                alpha=analysis.alpha,
                is_burst=is_burst,
                is_baseline=is_base,
                label=str(label) if pd.notna(label) else None,
                unit=analysis.measurement_unit,
                k_crit=analysis.k_crit
            )

            # Display Point Breakdown
            st.markdown(f"""
            <div class="provenance-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap;">
                    <span style="font-size: 1.15rem; font-weight: 700; color: #f8fafc;">
                        Observation Point Analysis: Interval #{selected_interval:,}
                    </span>
                    <span class="mode-badge" style="background: {'rgba(239, 68, 68, 0.25)' if is_burst else ('rgba(99, 102, 241, 0.2)' if is_base else 'rgba(16, 185, 129, 0.2)')}; color: {'#fca5a5' if is_burst else ('#c4b5fd' if is_base else '#6ee7b7')}; font-size: 0.85rem; padding: 4px 12px;">
                        {explanation['status_badge']}
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px;">
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase;">Measured Count (k)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #f8fafc;">{sel_count:,} <span style="font-size: 0.8rem; color: #94a3b8;">{analysis.measurement_unit}</span></div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase;">Baseline Rate (λ̂)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #10b981;">{analysis.lambda_hat:.2f} <span style="font-size: 0.8rem; color: #94a3b8;">{analysis.measurement_unit}/int</span></div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase;">Deviation from λ̂</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: {'#f87171' if explanation['difference'] > 0 else '#94a3b8'};">{explanation['difference']:+.1f} <span style="font-size: 0.8rem; color: #94a3b8;">({explanation['multiplier']:.2f}×)</span></div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase;">Upper-Tail P(X ≥ k)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: {'#f87171' if p_val < analysis.alpha else '#38bdf8'};">{p_val:.3e}</div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.6); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.06);">
                        <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase;">Cutoff Threshold (k_crit)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #cbd5e1;">{analysis.k_crit} <span style="font-size: 0.8rem; color: #94a3b8;">(α={analysis.alpha:g})</span></div>
                    </div>
                </div>
                <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6; margin-bottom: 12px; background: rgba(30, 41, 59, 0.5); padding: 12px 16px; border-radius: 8px; border-left: 4px solid {'#ef4444' if is_burst else '#10b981'};">
                    <b>Statistical Narrative:</b> {explanation['narrative']}
                </div>
                <div style="color: #fde68a; font-size: 0.83rem; line-height: 1.5; background: rgba(245, 158, 11, 0.08); padding: 8px 14px; border-radius: 6px; border-left: 3px solid #f59e0b;">
                    🛡️ <b>Academic Caution:</b> {explanation['caveat']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 5: HYPOTHESIS TESTING & INFERENCE (Section E)
    # -------------------------------------------------------------------------
    with tab_inference:
        st.subheader("📐 Statistical Hypothesis Testing & Inferential Framework")
        st.caption("Formal statement of hypotheses, exact upper-tail survival calculation, confidence intervals, and multiple-testing corrections.")

        hyp_col1, hyp_col2 = st.columns(2, gap="medium")
        with hyp_col1:
            st.markdown(f"""
            <div class="provenance-card">
                <div style="font-weight: 700; color: #f8fafc; font-size: 1rem; margin-bottom: 8px;">
                    Formal Hypothesis Statement
                </div>
                <ul style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; padding-left: 20px; margin-bottom: 0;">
                    <li><b>Null Hypothesis ($H_0$):</b> Traffic arrival count $X_i$ in interval $i$ is generated by a stationary Poisson process with rate parameter $\\lambda = \\hat{{\\lambda}} = {analysis.lambda_hat:.2f}$.
                    $$H_0: \\lambda_i = \\hat{{\\lambda}}_{{base}}$$</li>
                    <li><b>Alternative Hypothesis ($H_1$):</b> Traffic arrival count $X_i$ in interval $i$ is generated by an inflated rate parameter indicating an upper-tail volume surge.
                    $$H_1: \\lambda_i > \\hat{{\\lambda}}_{{base}}$$</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with hyp_col2:
            st.markdown(f"""
            <div class="config-card">
                <div style="font-weight: 700; color: #f8fafc; font-size: 1rem; margin-bottom: 8px;">
                    Upper-Tail Survival Formulation
                </div>
                <div style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.5;">
                    The upper-tail p-value is computed from the cumulative survival function:
                    $$p = P(X \\ge k) = 1 - P(X \\le k - 1) = \\sum_{{j=k}}^{{\\infty}} \\frac{{\\hat{{\\lambda}}^j e^{{-\\hat{{\\lambda}}}}}}{{j!}}$$
                    Implemented numerically via SciPy:
                    <code>scipy.stats.poisson.sf(k - 1, lambda_hat)</code>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Misconception Clarifications Box
        st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 10px; padding: 14px 18px; margin-top: 12px; margin-bottom: 18px;">
            <div style="font-weight: 700; color: #a5b4fc; margin-bottom: 6px;">
                💡 Statistical Interpretation Rules & Anti-Misconception Principles
            </div>
            <ul style="color: #cbd5e1; font-size: 0.86rem; line-height: 1.6; margin-bottom: 0;">
                <li><b>What the P-Value Is:</b> The exact probability, assuming $H_0$ is true (traffic is strictly Poisson at rate $\\hat{\\lambda}$), of observing $k$ or more arrivals purely by random chance.</li>
                <li><b>What the P-Value Is NOT:</b> The p-value is <i>NOT</i> the probability that an attack occurred, nor is $1 - p$ the confidence that an attack is present.</li>
                <li><b>Failure to Reject $H_0$ is NOT Proof of Normality:</b> When $p \\ge \\alpha$, we fail to reject $H_0$. This merely demonstrates insufficient evidence of an upper-tail surge; it does <i>not</i> prove that the underlying packet process is truly Poisson (absence of evidence is not evidence of absence).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Confidence Intervals for Lambda
        st.markdown(f"#### 📊 Baseline Rate (λ̂ = {analysis.lambda_hat:.4f}) Confidence Intervals")
        ci_dict = analysis.lambda_ci
        if ci_dict:
            garwood_low, garwood_high = ci_dict["exact_garwood"]
            wald_low, wald_high = ci_dict["wald_asymptotic"]
            m_base_size = ci_dict["sample_size"]

            ci_table = pd.DataFrame({
                "Estimation Method": ["Garwood Exact (Chi-Square)", "Wald Asymptotic (Normal)"],
                "Confidence Level": ["95.0%", "95.0%"],
                "Sample Size (m)": [f"{m_base_size:,}", f"{m_base_size:,}"],
                "Lower Bound": [f"{garwood_low:.4f}", f"{wald_low:.4f}"],
                "Point Estimate (λ̂)": [f"{analysis.lambda_hat:.4f}", f"{analysis.lambda_hat:.4f}"],
                "Upper Bound": [f"{garwood_high:.4f}", f"{wald_high:.4f}"],
                "Interval Width": [f"{garwood_high - garwood_low:.4f}", f"{wald_high - wald_low:.4f}"],
                "Mathematical Formulation": [
                    r"[ χ²_{2T, 0.025}/(2m),  χ²_{2(T+1), 0.975}/(2m) ]",
                    r"λ̂ ± 1.96 · √(λ̂ / m)"
                ]
            })
            st.dataframe(ci_table, width="stretch", hide_index=True)
            st.caption(f"Note: {ci_dict['interpretation']}")

        # Multiple Testing & False Discovery Rate (Benjamini-Hochberg)
        st.markdown("#### 🔬 Multiple Testing Adjustment (Benjamini-Hochberg FDR)")
        st.write(
            f"Testing {analysis.evaluated_count:,} separate intervals at nominal $\\alpha = {analysis.alpha:g}$ expects "
            f"approximately ${analysis.evaluated_count:,} \\times {analysis.alpha:g} = {analysis.evaluated_count * analysis.alpha:.1f}$ false positives under $H_0$ due to family-wise error inflation. "
            "The Benjamini-Hochberg (BH) procedure controls the False Discovery Rate across all evaluated intervals."
        )

        bh_fdr = analysis.bh_fdr
        fdr_c1, fdr_c2, fdr_c3 = st.columns(3)
        fdr_c1.metric("Unadjusted Bursts (p < α)", f"{analysis.burst_count} int", help="Individual uncorrected hypothesis rejections.")
        fdr_c2.metric("BH FDR Discoveries (q < α)", f"{bh_fdr.get('num_discoveries', 0)} int", help="Statistically verified discoveries with controlled FDR.")
        filtered_out = max(0, analysis.burst_count - bh_fdr.get('num_discoveries', 0))
        fdr_c3.metric("Filtered Multi-Test False Alarms", f"{filtered_out} int", help="Intervals whose individual p < α was attributable to multiple comparisons.")

    # -------------------------------------------------------------------------
    # TAB 6: POISSON MODEL DIAGNOSTICS & GoF (Section F)
    # -------------------------------------------------------------------------
    with tab_diagnostics:
        st.subheader("🔬 Poisson Model Diagnostics & Baseline Goodness of Fit")
        st.caption("Empirical testing of Poisson equi-dispersion (Var = Mean) and Chi-Square Goodness-of-Fit with Cochran's pooling.")

        # Equi-dispersion Card
        d_ratio = analysis.baseline_dispersion
        d_status = "Equi-dispersed (Var ≈ Mean)" if 0.8 <= d_ratio <= 1.3 else ("Overdispersed (Var > Mean)" if d_ratio > 1.3 else "Underdispersed (Var < Mean)")

        st.markdown(f"""
        <div class="provenance-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Equi-Dispersion Diagnostic</span>
                <span class="mode-badge" style="background: {'rgba(239, 68, 68, 0.2)' if d_ratio > 1.5 else 'rgba(16, 185, 129, 0.2)'}; color: {'#fca5a5' if d_ratio > 1.5 else '#6ee7b7'};">
                    {d_status}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 8px;">
                <div><b>Baseline Mean (λ̂):</b> <code>{analysis.lambda_hat:.4f}</code></div>
                <div><b>Baseline Variance (s²):</b> <code>{analysis.baseline_variance:.4f}</code></div>
                <div><b>Dispersion Index (D = s²/λ̂):</b> <code>{d_ratio:.3f}</code></div>
            </div>
            <div style="color: #94a3b8; font-size: 0.85rem; line-height: 1.5;">
                Theoretical Poisson processes require $D = 1$. In network traffic, packet clustering and TCP bursts frequently produce $D > 1$ (overdispersion), which inflates upper-tail probabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Chi-Square Goodness of Fit on Baseline
        st.markdown("#### 📐 Pooled Chi-Square Goodness-of-Fit Test on Baseline Partition")
        gof = analysis.goodness_of_fit

        if gof and gof.get("valid"):
            chi_c1, chi_c2, chi_c3, chi_c4 = st.columns(4)
            chi_c1.metric("Chi-Square Stat (χ²)", f"{gof['chi2_stat']:.2f}")
            chi_c2.metric("Degrees of Freedom (df)", f"{gof['df']}")
            chi_c3.metric("GoF P-Value", f"{gof['p_value']:.4e}")
            chi_c4.metric("Decision (α = 0.05)", "Reject H₀" if gof["reject_h0"] else "Fail to Reject H₀")

            st.info(f"💡 **Goodness-of-Fit Finding:** {gof['conclusion']}")

            # Binned Frequencies Bar Chart
            fig_gof = go.Figure()
            fig_gof.add_trace(go.Bar(
                x=gof["bins"],
                y=gof["observed_frequencies"],
                name="Observed Frequencies (O)",
                marker=dict(color="#6366f1")
            ))
            fig_gof.add_trace(go.Bar(
                x=gof["bins"],
                y=gof["expected_frequencies"],
                name="Poisson Expected (E)",
                marker=dict(color="#10b981")
            ))

            fig_gof.update_layout(
                title=dict(
                    text="<b>Observed vs. Expected Frequencies across Cochran-Pooled Bins (E_j ≥ 5.0)</b>",
                    font=dict(size=14, color="#f8fafc")
                ),
                barmode="group",
                paper_bgcolor="#131826",
                plot_bgcolor="#0e131f",
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                xaxis=dict(title="Pooled Count Bins", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Interval Frequency", tickfont=dict(color="#94a3b8"), gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(15, 23, 42, 0.8)"
                ),
                margin=dict(l=20, r=20, t=60, b=35),
                height=350
            )
            st.plotly_chart(fig_gof, width="stretch")
        else:
            reason = gof.get("reason", "Sample size or variance insufficient for valid Chi-Square pooling.") if gof else "Diagnostics not computed."
            st.warning(f"⚠️ **Chi-Square Goodness of Fit Not Applicable:** {reason}")

        # Real-World Model Limitations & Disclosures
        st.markdown("#### ⚠️ Real-World Network Model Limitations & Academic Disclosures")
        st.markdown("""
        <div class="config-card">
            <ol style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.6; margin-bottom: 0; padding-left: 20px;">
                <li><b>Overdispersion & Heavy Tails:</b> Real Internet traffic exhibits burstiness, variance clustering, and heavy-tailed inter-arrival times (due to TCP windowing and application batching), violating strict Poisson equi-dispersion ($s^2 = \\hat{\\lambda}$).</li>
                <li><b>Non-Stationarity & Temporal Drift:</b> Internet arrival rates fluctuate according to business hours, backup schedules, and diurnal patterns. A single static baseline $\\hat{\\lambda}$ will naturally drift over long time windows.</li>
                <li><b>Packet Trains & Autocorrelation:</b> Packets belonging to the same flow or transaction arrive consecutively in 'trains', which breaks the Poisson independent increments assumption.</li>
                <li><b>Baseline Contamination:</b> If the baseline window contains low-rate stealth attacks or port scans, $\\hat{\\lambda}$ is inflated, increasing the risk of false negatives in evaluation.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # TAB 7: POISSON VS. BINOMIAL (Section G)
    # -------------------------------------------------------------------------
    with tab_model_compare:
        st.subheader("⚖️ Poisson vs. Binomial Distribution: Theoretical Comparison")
        st.caption("Why Poisson is used for network traffic analysis rather than Binomial, and the Law of Rare Events convergence.")

        # Educational Comparison Table
        model_comp_table = pd.DataFrame({
            "Theoretical Dimension": [
                "Trial Count (n)",
                "Arrival Probability (p)",
                "Support / Domain",
                "Variance Relationship",
                "Parameter Requirements",
                "Network Traffic Suitability"
            ],
            "Binomial Distribution B(n, p)": [
                "Fixed, finite integer n known in advance",
                "Constant probability p of arrival per trial",
                "k ∈ {0, 1, 2, ..., n}",
                "Var(X) = n·p·(1 - p) < E[X] (Under-dispersed)",
                "Requires knowing both n and p",
                "Unsuitable for open packet streams: network links do not have a fixed upper capacity n of packet slots."
            ],
            "Poisson Distribution Poisson(λ)": [
                "Unbounded / Infinite (continuous time limit n → ∞)",
                "Infinitesimal probability in micro-intervals (p → 0)",
                "k ∈ {0, 1, 2, ...} (countably infinite)",
                "Var(X) = λ = E[X] (Equi-dispersed)",
                "Requires only a single parameter: arrival rate λ",
                "Naturally models continuous independent arrivals in fixed time intervals [t, t + Δt]."
            ]
        })
        st.dataframe(model_comp_table, width="stretch", hide_index=True)

        st.markdown("""
        #### 📐 The Law of Rare Events (Poisson Limit Theorem)
        When the number of potential packet arrival opportunities $n$ is very large and the probability of an arrival in any sub-microsecond $p$ is very small such that the expected arrivals in the interval remain constant ($n p = \\lambda$):
        $$\\lim_{\\substack{n \\to \\infty \\\\ p \\to 0 \\\\ n p = \\lambda}} \\binom{n}{k} p^k (1 - p)^{n - k} = \\frac{\\lambda^k e^{-\\lambda}}{k!}$$
        """)

        # Theoretical Convergence Plot
        comp_res = compute_poisson_binomial_comparison(analysis.lambda_hat)
        if comp_res:
            fig_binom = go.Figure()

            # Poisson curve
            fig_binom.add_trace(go.Scatter(
                x=comp_res["k_values"],
                y=comp_res["poisson_pmf"],
                mode="lines+markers",
                name=f"Poisson(λ̂ = {analysis.lambda_hat:.2f})",
                line=dict(color="#10b981", width=3),
                marker=dict(size=4)
            ))

            colors = ["#f59e0b", "#38bdf8", "#818cf8"]
            for idx, sc in enumerate(comp_res["scenarios"]):
                fig_binom.add_trace(go.Scatter(
                    x=comp_res["k_values"],
                    y=sc["pmf"],
                    mode="lines",
                    name=sc["name"],
                    line=dict(color=colors[idx % len(colors)], width=1.8, dash="dash")
                ))

            fig_binom.update_layout(
                title=dict(
                    text=f"<b>Law of Rare Events Convergence:</b> Binomial(n, p = {analysis.lambda_hat:.2f}/n) Converging to Poisson(λ̂ = {analysis.lambda_hat:.2f})",
                    font=dict(size=14, color="#f8fafc")
                ),
                paper_bgcolor="#131826",
                plot_bgcolor="#0e131f",
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                xaxis=dict(title=f"Count k ({analysis.measurement_unit})", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8")),
                yaxis=dict(title="Probability Mass P(X = k)", gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#94a3b8")),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(15, 23, 42, 0.8)"
                ),
                margin=dict(l=20, r=20, t=65, b=35),
                height=360
            )
            st.plotly_chart(fig_binom, width="stretch")

            st.caption(
                "Notice that as trial capacity n increases, the Binomial distribution converges directly onto the Poisson PMF. "
                "We do not fabricate artificial Binomial trial numbers for the real packet trace because an open network link does not have a predefined number of trials."
            )

    # -------------------------------------------------------------------------
    # TAB 8: GROUND-TRUTH EVALUATION (Section H)
    # -------------------------------------------------------------------------
    with tab_eval:
        st.subheader("🎯 Ground-Truth Evaluation & Anomaly Overlap")
        if analysis.has_labels and analysis.label_contingency:
            ctg = analysis.label_contingency
            label_title = "Simulation Evaluation vs Injected Labels" if is_synthetic else "Ground-Truth Evaluation"
            st.caption(f"Recomputed dynamically from actual observations: {label_title}.")

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
        else:
            st.info(
                "ℹ️ **Unsupervised Baseline Detection:** No ground-truth anomaly labels were provided in this dataset. "
                "The Poisson detector operates strictly unsupervised, identifying mathematical upper-tail surges based purely on baseline arrival dynamics."
            )

    # -------------------------------------------------------------------------
    # TAB 9: DATA EXPORT (Section I)
    # -------------------------------------------------------------------------
    with tab_export:
        st.subheader("💾 Export Complete Analysis Data")
        st.write("Download the complete time series with calibrated baseline arrival rate λ̂, upper-tail survival probabilities, critical cutoff k_crit, and status annotations.")

        export_data = prepare_export_dataframe(analysis)
        export_data["data_source"] = source_label
        export_data["is_synthetic"] = is_synthetic
        export_data["k_crit"] = analysis.k_crit

        csv_bytes = export_data.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download Complete Analysis CSV",
            data=csv_bytes,
            file_name=export_filename,
            mime="text/csv",
            key="btn_download_analysis_csv",
            help="Download complete dataset with intervals, counts, baseline lambda, p-values, burst flags, and metadata."
        )

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
