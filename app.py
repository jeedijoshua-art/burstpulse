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
from investigation_workflow import (
    LikelyCause,
    AnalystAssessment,
    ASSESSMENT_MARKERS,
    AnalystReview,
    InvestigationGroup,
    get_dataset_reviews,
    save_analyst_review,
    reset_demo_reviews,
    ensure_groups_state,
    get_dataset_groups,
    save_investigation_group,
    delete_investigation_group,
    reset_demo_groups,
    calculate_group_analytics,
    run_automated_demo_workflow,
    generate_single_burst_report_markdown,
    generate_group_investigation_report_markdown,
    generate_summary_investigation_report_markdown,
    get_investigative_recommendations
)

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

    # Pre-extract evaluated data and baseline data
    base_df = analysis.baseline_data
    eval_df = analysis.evaluated_data
    full_df = analysis.full_data
    total_obs = len(full_df)

    # Resolve stable dataset key for review isolation
    if active_dataset_entry:
        dataset_key = active_dataset_entry.id
    elif uploaded_file is not None:
        dataset_key = f"csv_{uploaded_file.name}_{uploaded_file.size}"
    else:
        dataset_key = "default_traffic_session"

    # Retrieve reviews and investigation groups for this dataset
    reviews_map = get_dataset_reviews(st.session_state, dataset_key)
    groups_map = get_dataset_groups(st.session_state, dataset_key)

    # Calculate review counts
    flagged_df = eval_df[eval_df["is_burst"]].copy()
    total_flagged = len(flagged_df)
    flagged_intervals = flagged_df["interval"].tolist() if not flagged_df.empty else []

    count_benign = sum(1 for r in reviews_map.values() if r.assessment == AnalystAssessment.LIKELY_BENIGN.value)
    count_suspected = sum(1 for r in reviews_map.values() if r.assessment == AnalystAssessment.SUSPECTED_ATTACK.value)
    count_confirmed = sum(1 for r in reviews_map.values() if r.assessment == AnalystAssessment.CONFIRMED_ATTACK.value)
    count_reviewed = count_benign + count_suspected + count_confirmed
    count_unreviewed = max(0, total_flagged - count_reviewed)

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 1: ANALYST TRIAGE & AUTOMATED DEMO ACTION BAR
    # -------------------------------------------------------------------------
    st.markdown("### 🛡️ Analyst Triage & Demonstration Engine")

    triage_col1, triage_col2, triage_col3 = st.columns([1.8, 1.4, 4.8], gap="medium")

    with triage_col1:
        if st.button("▶️ Run Automated Demo", key="btn_run_automated_demo", help="Demonstrate the complete analyst investigation workflow on real detected bursts."):
            demo_res = run_automated_demo_workflow(st.session_state, dataset_key, analysis)
            if demo_res["success"]:
                st.session_state["demo_banner"] = ("success", demo_res["message"])
                if demo_res.get("active_group_id"):
                    st.session_state["selected_investigation_group_id"] = demo_res["active_group_id"]
                if demo_res.get("demo_intervals"):
                    st.session_state["selected_investigation_interval"] = demo_res["demo_intervals"][0]
            else:
                st.session_state["demo_banner"] = ("warning", demo_res["message"])
            st.rerun()

    with triage_col2:
        if st.button("🔄 Reset Demo Records", key="btn_reset_demo_records", help="Remove simulated demo records while preserving all genuine analyst reviews and groups."):
            num_reset_reviews = reset_demo_reviews(st.session_state, dataset_key)
            num_reset_groups = reset_demo_groups(st.session_state, dataset_key)
            st.session_state["demo_banner"] = ("info", f"Reset complete: Removed {num_reset_reviews} demo reviews and {num_reset_groups} demo groups. Genuine analyst investigations preserved.")
            st.rerun()

    with triage_col3:
        # Display review summary badges
        st.markdown(f"""
        <div style="display: flex; gap: 8px; flex-wrap: wrap; align-items: center; justify-content: flex-end; padding-top: 4px;">
            <span class="mode-badge" style="background: rgba(239, 68, 68, 0.2); color: #fca5a5;">
                🚨 Unreviewed: {count_unreviewed}
            </span>
            <span class="mode-badge" style="background: rgba(59, 130, 246, 0.2); color: #93c5fd;">
                ✅ Likely Benign: {count_benign}
            </span>
            <span class="mode-badge" style="background: rgba(245, 158, 11, 0.2); color: #fde68a;">
                ⚠️ Suspected: {count_suspected}
            </span>
            <span class="mode-badge" style="background: rgba(168, 85, 247, 0.2); color: #d8b4fe;">
                🛑 Confirmed: {count_confirmed}
            </span>
            <span class="mode-badge" style="background: rgba(148, 163, 184, 0.15); color: #cbd5e1;">
                Total Bursts: {total_flagged}
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Immediately visible demo notification & demo showcase container
    if "demo_banner" in st.session_state:
        b_type, b_msg = st.session_state["demo_banner"]
        if b_type == "success":
            st.success(f"**DEMO ACTIVE — SIMULATED ANALYST REVIEW:** {b_msg}")
        elif b_type == "warning":
            st.warning(b_msg)
        else:
            st.info(b_msg)

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 2: CLEAN & PROFESSIONAL TRAFFIC GRAPH (PHASE 6)
    # -------------------------------------------------------------------------
    st.subheader("📈 Traffic Arrival Time Series & Multi-State Classification")
    st.caption(
        f"Real-time arrival counts with stationary baseline rate λ̂ = {analysis.lambda_hat:.2f} {analysis.measurement_unit}/int "
        f"and upper-tail decision cutoff k_crit = {analysis.k_crit} {analysis.measurement_unit} (α = {analysis.alpha:g})."
    )

    fig_main = go.Figure()

    # 1. Baseline calibration trace (clean continuous line)
    fig_main.add_trace(go.Scatter(
        x=base_df["interval"],
        y=base_df["packet_count"],
        mode="lines",
        name="Baseline Calibration (λ̂)",
        line=dict(color="#4f46e5", width=2.0),
        hovertemplate="<b>Baseline Window</b><br><b>Interval:</b> %{x}<br><b>Count:</b> %{y} " + analysis.measurement_unit + "<extra></extra>"
    ))

    # 2. Continuous evaluated trace (Normal observations rendered cleanly without marker clumping)
    fig_main.add_trace(go.Scatter(
        x=eval_df["interval"],
        y=eval_df["packet_count"],
        mode="lines",
        name="Evaluated Arrival Rate",
        line=dict(color="#64748b", width=1.6),
        customdata=eval_df["upper_tail_probability"],
        hovertemplate="<b>Interval:</b> %{x}<br><b>Count:</b> %{y} " + analysis.measurement_unit + "<br><b>P(X ≥ k):</b> %{customdata:.4e}<br><b>Segment:</b> Evaluated<extra></extra>"
    ))

    # 3. Categorized Flagged Bursts by Analyst Marker States
    if not flagged_df.empty:
        flagged_copy = flagged_df.copy()
        flagged_copy["assessment"] = [
            reviews_map[int(i)].assessment if int(i) in reviews_map else AnalystAssessment.UNREVIEWED.value
            for i in flagged_copy["interval"]
        ]
        flagged_copy["likely_cause"] = [
            reviews_map[int(i)].likely_cause if int(i) in reviews_map else "Pending Analyst Review"
            for i in flagged_copy["interval"]
        ]
        flagged_copy["is_demo"] = [
            reviews_map[int(i)].is_demo if int(i) in reviews_map else False
            for i in flagged_copy["interval"]
        ]
        flagged_copy["group_name"] = [
            reviews_map[int(i)].group_name if int(i) in reviews_map and reviews_map[int(i)].group_name else "Ungrouped"
            for i in flagged_copy["interval"]
        ]

        # Iterate through the 4 distinct marker categories
        for state_key, marker_cfg in ASSESSMENT_MARKERS.items():
            state_pts = flagged_copy[flagged_copy["assessment"] == state_key]
            if not state_pts.empty:
                custom_hover = [
                    f"Count: {row['packet_count']}<br>P(X ≥ k): {row['upper_tail_probability']:.2e}<br>Cause: {row['likely_cause']}<br>Group: {row['group_name']}{' (Demo)' if row['is_demo'] else ''}"
                    for _, row in state_pts.iterrows()
                ]
                fig_main.add_trace(go.Scatter(
                    x=state_pts["interval"],
                    y=state_pts["packet_count"],
                    mode="markers",
                    name=marker_cfg["name"],
                    marker=dict(
                        color=marker_cfg["color"],
                        size=marker_cfg["size"],
                        symbol=marker_cfg["symbol"],
                        line=dict(width=marker_cfg["width"], color=marker_cfg["line_color"])
                    ),
                    customdata=custom_hover,
                    hovertemplate=f"<b>{marker_cfg['name']}</b><br><b>Interval:</b> %{{x}}<br>%{{customdata}}<extra></extra>"
                ))

    # 4. Simulation/Ground-truth attack labels (when available, distinctly rendered)
    if analysis.has_labels and "has_attack_label" in eval_df.columns:
        attack_pts = eval_df[eval_df["has_attack_label"]]
        if not attack_pts.empty:
            attack_labels = attack_pts["attack_type"] if "attack_type" in attack_pts.columns else ["Simulated Anomaly"] * len(attack_pts)
            marker_name = "Simulation Benchmark Marker" if is_synthetic else "Ground-Truth Label"
            fig_main.add_trace(go.Scatter(
                x=attack_pts["interval"],
                y=attack_pts["packet_count"],
                mode="markers",
                name=marker_name,
                marker=dict(
                    color="#f59e0b",
                    size=9,
                    symbol="diamond-open",
                    line=dict(width=1.8, color="#d97706")
                ),
                customdata=attack_labels,
                hovertemplate=f"<b>⚠️ {marker_name}</b><br><b>Interval:</b> %{{x}}<br><b>Count:</b> %{{y}}<br><b>Label:</b> %{{customdata}}<extra></extra>"
            ))

    # 5. Baseline arrival rate reference line
    fig_main.add_hline(
        y=analysis.lambda_hat,
        line_dash="dash",
        line_color="#059669",
        line_width=2,
        annotation_text=f"Baseline Rate λ̂ = {analysis.lambda_hat:.2f}",
        annotation_position="top left",
        annotation_font=dict(color="#059669", size=11)
    )

    # 6. Statistically derived upper-tail threshold line
    fig_main.add_hline(
        y=analysis.k_crit,
        line_dash="dot",
        line_color="#ef4444",
        line_width=2,
        annotation_text=f"Upper-Tail Threshold k_crit = {analysis.k_crit} (α = {analysis.alpha:g})",
        annotation_position="bottom right",
        annotation_font=dict(color="#ef4444", size=11)
    )

    # 7. Shaded baseline window
    last_base_interval = base_df["interval"].iloc[-1]
    first_base_interval = base_df["interval"].iloc[0]

    if isinstance(last_base_interval, (int, float, np.integer, np.floating)):
        fig_main.add_vrect(
            x0=first_base_interval - 0.5,
            x1=last_base_interval + 0.5,
            fillcolor="rgba(79, 70, 229, 0.06)",
            layer="below",
            line_width=0,
            annotation_text="Baseline Calibration Window",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#6366f1")
        )

        fig_main.add_vline(
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

    # Layout with decoupled title and horizontal legend bar
    fig_main.update_layout(
        title=dict(
            text="<b>Network Traffic Arrival Time Series:</b> Baseline Calibration & Multi-State Classification",
            font=dict(size=14.5, color="#f8fafc"),
            x=0.01,
            y=0.98,
            xanchor="left"
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
            y=1.04,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(15, 23, 42, 0.85)",
            bordercolor="rgba(255, 255, 255, 0.12)",
            borderwidth=1,
            font=dict(size=10.5)
        ),
        margin=dict(l=20, r=20, t=85, b=35),
        height=520,
        hovermode="closest"
    )

    st.plotly_chart(fig_main, width="stretch")

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 3: FOUR ANALYST CATEGORY VIEWS CONSOLE (PHASE 1)
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("🛡️ Analyst Investigation Console & Category Views")
    st.caption("Inspect and triage detected bursts across four dedicated review categories with full interval tables, group assignments, and metadata.")

    cat_options = [
        "UNREVIEWED",
        "SAFE / LIKELY BENIGN",
        "SUSPECTED ATTACK",
        "CONFIRMED ATTACK"
    ]

    selected_category_tab = st.radio(
        "Select Analyst Category View",
        options=cat_options,
        format_func=lambda x: {
            "UNREVIEWED": f"🚨 UNREVIEWED ({count_unreviewed:,})",
            "SAFE / LIKELY BENIGN": f"🛡️ SAFE / LIKELY BENIGN ({count_benign:,})",
            "SUSPECTED ATTACK": f"⚠️ SUSPECTED ATTACK ({count_suspected:,})",
            "CONFIRMED ATTACK": f"🛑 CONFIRMED ATTACK ({count_confirmed:,})"
        }[x],
        horizontal=True,
        key="analyst_category_view_radio"
    )

    # Filter bursts strictly by selected category
    category_target_assessment = {
        "UNREVIEWED": AnalystAssessment.UNREVIEWED.value,
        "SAFE / LIKELY BENIGN": AnalystAssessment.LIKELY_BENIGN.value,
        "SUSPECTED ATTACK": AnalystAssessment.SUSPECTED_ATTACK.value,
        "CONFIRMED ATTACK": AnalystAssessment.CONFIRMED_ATTACK.value
    }[selected_category_tab]

    cat_burst_rows = []
    for _, row in flagged_df.iterrows():
        i_num = int(row["interval"])
        cnt = int(row["packet_count"])
        p_val = float(row["upper_tail_probability"])
        rev = reviews_map.get(i_num)

        a_assessment = rev.assessment if rev else AnalystAssessment.UNREVIEWED.value
        a_cause = rev.likely_cause if rev else "Pending Review"
        a_group = rev.group_name if rev and rev.group_name else "Ungrouped"
        is_d = rev.is_demo if rev else False

        if a_assessment == category_target_assessment:
            diff = cnt - analysis.lambda_hat
            ratio = (cnt / analysis.lambda_hat) if analysis.lambda_hat > 0 else 1.0

            r_dict = {
                "Interval": i_num,
                "Count (k)": cnt,
                "Unit": analysis.measurement_unit,
                "Baseline (λ̂)": round(analysis.lambda_hat, 2),
                "Abs Dev": f"{diff:+.1f}",
                "Ratio": f"{ratio:.2f}×",
                "Upper-Tail P": f"{p_val:.2e}" if p_val < 0.0001 else f"{p_val:.4f}",
                "Threshold (k_crit)": analysis.k_crit,
                "Analyst Assessment": f"{a_assessment}{' (Demo)' if is_d else ''}",
                "Likely Cause": a_cause,
                "Group Assignment": a_group
            }
            if "timestamp" in row:
                r_dict["Timestamp"] = str(row["timestamp"])
            if "attack_type" in row and pd.notna(row["attack_type"]):
                r_dict["Ground-Truth"] = str(row["attack_type"])
            elif "scenario_label" in row and pd.notna(row["scenario_label"]):
                r_dict["Ground-Truth"] = str(row["scenario_label"])

            cat_burst_rows.append(r_dict)

    df_cat_bursts = pd.DataFrame(cat_burst_rows)

    if df_cat_bursts.empty:
        st.info(f"No detected bursts currently match category '{selected_category_tab}'.")
    else:
        st.dataframe(
            df_cat_bursts,
            width="stretch",
            hide_index=True,
            height=260
        )
        st.caption(f"Showing all {len(df_cat_bursts):,} burst intervals in category '{selected_category_tab}'.")

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 4: NAMED INVESTIGATION GROUPS & ANALYTICS (PHASES 2 & 3)
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("📁 Named Investigation Groups & Multi-Burst Analytics")
    st.caption("Organize nonconsecutive burst intervals into named investigations, evaluate aggregate arrival statistics, and generate group-level reports.")

    # Refresh current groups
    existing_groups = list(groups_map.values())
    group_options = ["+ Create New Investigation Group"] + [f"{g.name} [{g.group_id}]" for g in existing_groups]

    # Resolve active selected group
    default_grp_idx = 0
    if "selected_investigation_group_id" in st.session_state:
        for idx, g in enumerate(existing_groups):
            if g.group_id == st.session_state["selected_investigation_group_id"]:
                default_grp_idx = idx + 1
                break

    sel_group_label = st.selectbox(
        "Select Investigation Group to Manage or Inspect",
        options=group_options,
        index=default_grp_idx if default_grp_idx < len(group_options) else 0,
        key="investigation_group_selector"
    )

    active_group: Optional[InvestigationGroup] = None
    if sel_group_label != "+ Create New Investigation Group":
        selected_gid = sel_group_label.split("[")[-1].replace("]", "").strip()
        active_group = groups_map.get(selected_gid)
        if active_group:
            st.session_state["selected_investigation_group_id"] = active_group.group_id

    # Group Form Container
    with st.expander(f"⚙️ Configure Investigation Group: {active_group.name if active_group else 'New Group'}", expanded=(active_group is not None)):
        with st.form(key=f"form_manage_group_{active_group.group_id if active_group else 'new'}"):
            g_col1, g_col2 = st.columns(2, gap="medium")

            with g_col1:
                suggested_names = [
                    "Scheduled Backup Surge",
                    "Legitimate Traffic Spike",
                    "Possible DDoS Investigation",
                    "Network Scanning Investigation",
                    "Unexplained Traffic Anomaly"
                ]
                default_name = active_group.name if active_group else suggested_names[0]
                inp_group_name = st.text_input("Investigation Group Name (Required)", value=default_name)

                # Cause dropdown (9 required options)
                cause_choices = [c.value for c in LikelyCause]
                def_cause = active_group.likely_cause if active_group else LikelyCause.SCHEDULED_BACKUP.value
                inp_cause = st.selectbox(
                    "Likely Cause of Burst (Required)",
                    options=cause_choices,
                    index=cause_choices.index(def_cause) if def_cause in cause_choices else 0,
                    help="Underlying operational or threat hypothesis for this group of arrivals."
                )

                inp_desc = st.text_area(
                    "Investigation Description",
                    value=active_group.description if active_group else "",
                    placeholder="Describe scope, targeted endpoints, and operational background...",
                    height=80
                )

            with g_col2:
                # Assessment dropdown (4 required options)
                assess_choices = [a.value for a in AnalystAssessment]
                def_assess = active_group.assessment if active_group else AnalystAssessment.UNREVIEWED.value
                inp_assessment = st.selectbox(
                    "Analyst Assessment (Required)",
                    options=assess_choices,
                    index=assess_choices.index(def_assess) if def_assess in assess_choices else 0,
                    help="Formal security triage decision. Decoupled from likely cause."
                )

                conf_levels = [0.3, 0.6, 0.8, 0.95]
                def_conf = active_group.confidence if active_group else 0.8
                inp_conf = st.select_slider(
                    "Confidence Level",
                    options=conf_levels,
                    value=def_conf if def_conf in conf_levels else 0.8,
                    format_func=lambda x: f"{int(x*100)}% ({'High' if x>=0.8 else ('Medium' if x>=0.5 else 'Low')})"
                )

                inp_evidence = st.text_input(
                    "Supporting Evidence Reference / Log Traces",
                    value=active_group.supporting_evidence if active_group else "",
                    placeholder="e.g. Host-Auth-Trace #1029, NetFlow-Sample-08"
                )

                inp_notes = st.text_area(
                    "Analyst Notes & Context",
                    value=active_group.notes if active_group else "",
                    placeholder="Corroborating investigative details...",
                    height=80
                )

            # Member interval selection (Supports nonconsecutive intervals)
            st.markdown("##### 🎯 Member Burst Intervals Selection")
            current_members = active_group.member_intervals if active_group else []
            unrev_intervals = [i for i in flagged_intervals if i not in reviews_map or reviews_map[i].assessment == AnalystAssessment.UNREVIEWED.value]

            bulk_col1, bulk_col2 = st.columns([2.5, 1.5])
            with bulk_col1:
                chk_bulk_all = st.checkbox(
                    f"Select all currently unreviewed bursts ({len(unrev_intervals)} intervals)",
                    value=False,
                    help="Pre-fills member intervals with all unreviewed bursts in this dataset."
                )
            with bulk_col2:
                confirm_bulk = st.checkbox(
                    "⚠️ Confirm bulk review classification",
                    value=False,
                    help="Explicit confirmation required before applying bulk classification across unreviewed bursts."
                )

            if chk_bulk_all and confirm_bulk and unrev_intervals:
                selectable_defaults = sorted(list(set(current_members + unrev_intervals)))
            else:
                selectable_defaults = [i for i in current_members if i in flagged_intervals]

            inp_members = st.multiselect(
                "Assign Nonconsecutive Member Intervals (IDs)",
                options=flagged_intervals,
                default=selectable_defaults,
                format_func=lambda i: f"Interval #{i:,} (Count: {eval_df.loc[eval_df['interval']==i, 'packet_count'].values[0]:,} {analysis.measurement_unit})"
            )

            # Actions
            btn_col1, btn_col2, _ = st.columns([2.0, 1.5, 3.5])
            with btn_col1:
                btn_save_group = st.form_submit_button("💾 Save / Update Investigation Group", type="primary")
            with btn_col2:
                btn_delete_group = st.form_submit_button("🗑️ Delete / Ungroup")

            if btn_save_group:
                if not inp_group_name.strip():
                    st.error("❌ Investigation group name cannot be empty.")
                elif not inp_members:
                    st.error("❌ Please select at least one member burst interval for this group.")
                else:
                    gid = active_group.group_id if active_group else f"grp_{dataset_key}_{int(datetime.now(timezone.utc).timestamp())}"
                    new_group = InvestigationGroup(
                        group_id=gid,
                        dataset_key=dataset_key,
                        name=inp_group_name.strip(),
                        likely_cause=inp_cause,
                        assessment=inp_assessment,
                        description=inp_desc.strip(),
                        notes=inp_notes.strip(),
                        supporting_evidence=inp_evidence.strip(),
                        confidence=inp_conf,
                        member_intervals=inp_members,
                        is_demo=False
                    )
                    save_investigation_group(
                        st.session_state, new_group, eval_df=eval_df,
                        lambda_hat=analysis.lambda_hat, alpha=analysis.alpha,
                        k_crit=analysis.k_crit, unit=analysis.measurement_unit
                    )
                    st.session_state["selected_investigation_group_id"] = gid
                    st.success(f"✅ Investigation group '{new_group.name}' saved with {len(inp_members)} member intervals. Member reviews updated.")
                    st.rerun()

            if btn_delete_group and active_group:
                delete_investigation_group(st.session_state, dataset_key, active_group.group_id)
                st.session_state.pop("selected_investigation_group_id", None)
                st.info(f"Group '{active_group.name}' removed and member intervals unbound.")
                st.rerun()

    # If an investigation group is selected, render comprehensive group analytics & reports
    if active_group and active_group.member_intervals:
        grp_analytics = calculate_group_analytics(
            active_group, eval_df, analysis.lambda_hat, analysis.alpha, analysis.measurement_unit
        )

        st.markdown(f"#### 📊 Investigation Summary: {active_group.name}")
        st.caption(f"Status: **{active_group.assessment}** &nbsp;|&nbsp; Cause: **{active_group.likely_cause}** &nbsp;|&nbsp; ID: `{active_group.group_id}`")

        # Group KPI Metrics Row
        g_kpi1, g_kpi2, g_kpi3, g_kpi4, g_kpi5 = st.columns(5)
        with g_kpi1:
            st.metric("Member Bursts", f"{grp_analytics['num_members']:,} int")
        with g_kpi2:
            st.metric("Interval Span", f"#{grp_analytics['earliest_interval']} – #{grp_analytics['latest_interval']}")
        with g_kpi3:
            st.metric("Count Range", f"{grp_analytics['min_count']:,} – {grp_analytics['max_count']:,}")
        with g_kpi4:
            st.metric("Mean Deviation", f"{grp_analytics['mean_deviation']:+.1f}", f"{grp_analytics['relative_ratio']:.2f}× λ̂")
        with g_kpi5:
            st.metric("P < α Improbability", f"{grp_analytics['prop_below_alpha']:.0%}", f"Min P: {grp_analytics['min_p_value']:.1e}")

        # Methodological Span Note
        st.info(f"ℹ️ **Span Telemetry Notice:** {grp_analytics['span_note']}")

        # Member Intervals Detail Table
        m_rows = []
        for _, m_row in grp_analytics["member_df"].iterrows():
            i_val = int(m_row["interval"])
            cnt = int(m_row["packet_count"])
            p_val = float(m_row["upper_tail_probability"])
            diff = cnt - analysis.lambda_hat
            ratio = (cnt / analysis.lambda_hat) if analysis.lambda_hat > 0 else 1.0
            gt = str(m_row.get("attack_type", m_row.get("scenario_label", "Unlabeled")))
            m_rows.append({
                "Interval": i_val,
                "Count (k)": cnt,
                "Abs Dev": f"{diff:+.1f}",
                "Ratio": f"{ratio:.2f}×",
                "Upper-Tail P": f"{p_val:.2e}",
                "Detector Flag": "🚨 Outlier" if m_row.get("is_burst", True) else "Normal",
                "Ground-Truth Label": gt
            })
        st.dataframe(pd.DataFrame(m_rows), width="stretch", hide_index=True, height=200)

        # -------------------------------------------------------------------------
        # WORKFLOW STEP 5: GROUP INVESTIGATION REPORT & EXPORT (PHASE 4)
        # -------------------------------------------------------------------------
        st.markdown("#### 📑 Group Investigation Report")
        grp_report_md = generate_group_investigation_report_markdown(
            active_group, grp_analytics, source_label, "BurstPulse Synthetic/Dataset Engine", analysis.measurement_unit
        )

        with st.expander(f"👁️ Preview Formal Report: {active_group.name}", expanded=False):
            st.markdown(grp_report_md)

        rep_btn_c1, rep_btn_c2 = st.columns(2)
        with rep_btn_c1:
            st.download_button(
                label=f"⬇️ Download Group Report (.md)",
                data=grp_report_md.encode("utf-8"),
                file_name=f"investigation_report_{active_group.group_id}.md",
                mime="text/markdown",
                key="btn_download_group_report_md"
            )
        with rep_btn_c2:
            export_grp_df = grp_analytics["member_df"].copy()
            st.download_button(
                label="⬇️ Export Member Telemetry (CSV)",
                data=export_grp_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{active_group.group_id}_member_telemetry.csv",
                mime="text/csv",
                key="btn_download_group_members_csv"
            )

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 5B: SINGLE BURST & FULL SUMMARY REPORTS
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("🔍 Single Burst Inspection & Global Reports")
    st.caption("Inspect individual outlier intervals, fine-tune individual reviews, or generate a comprehensive triage summary report.")

    selectable_intervals = flagged_intervals if flagged_intervals else eval_df["interval"].tolist()
    if selectable_intervals:
        default_target = selectable_intervals[0]
        if "selected_investigation_interval" not in st.session_state or st.session_state["selected_investigation_interval"] not in selectable_intervals:
            if not flagged_df.empty:
                default_target = int(flagged_df.loc[flagged_df["packet_count"].idxmax(), "interval"])
            st.session_state["selected_investigation_interval"] = default_target

        target_idx = st.selectbox(
            "Select Discrete Interval for Single-Burst Telemetry & Reporting",
            options=selectable_intervals,
            index=selectable_intervals.index(st.session_state["selected_investigation_interval"]) if st.session_state["selected_investigation_interval"] in selectable_intervals else 0,
            format_func=lambda i: f"Interval #{i:,} — Count: {eval_df.loc[eval_df['interval']==i, 'packet_count'].values[0]:,} {analysis.measurement_unit} [{reviews_map[i].assessment if i in reviews_map else 'Unreviewed'}]",
            key="single_interval_report_target_selector"
        )
        st.session_state["selected_investigation_interval"] = target_idx

        t_row = eval_df[eval_df["interval"] == target_idx].iloc[0]
        t_cnt = int(t_row["packet_count"])
        t_pval = float(t_row["upper_tail_probability"])
        t_gt = str(t_row.get("attack_type", t_row.get("scenario_label", ""))) or None

        act_rev = reviews_map.get(target_idx) or AnalystReview(
            dataset_key=dataset_key,
            interval=target_idx,
            observed_count=t_cnt,
            measurement_unit=analysis.measurement_unit,
            lambda_hat=analysis.lambda_hat,
            p_value=t_pval,
            alpha=analysis.alpha,
            k_crit=analysis.k_crit,
            likely_cause=LikelyCause.INSUFFICIENT_EVIDENCE.value,
            assessment=AnalystAssessment.UNREVIEWED.value,
            notes="",
            is_demo=False,
            ground_truth_label=t_gt
        )

        single_rep_md = generate_single_burst_report_markdown(act_rev, source_label)
        all_dataset_reviews = list(reviews_map.values())
        summary_rep_md = generate_summary_investigation_report_markdown(all_dataset_reviews, source_label, analysis)

        r_c1, r_c2 = st.columns(2, gap="medium")
        with r_c1:
            st.markdown(f"**Single Burst Report: Interval #{target_idx:,}**")
            with st.expander("👁️ Preview Single Burst Report", expanded=False):
                st.markdown(single_rep_md)
            st.download_button(
                label=f"⬇️ Download Interval #{target_idx} Report (.md)",
                data=single_rep_md.encode("utf-8"),
                file_name=f"investigation_report_interval_{target_idx}.md",
                mime="text/markdown",
                key="btn_download_single_report_md"
            )

        with r_c2:
            st.markdown(f"**Global Triage Summary ({len(all_dataset_reviews)} Reviewed)**")
            with st.expander("👁️ Preview Global Summary Report", expanded=False):
                st.markdown(summary_rep_md)
            st.download_button(
                label="⬇️ Download Full Summary Report (.md)",
                data=summary_rep_md.encode("utf-8"),
                file_name=f"{dataset_key}_investigation_summary.md",
                mime="text/markdown",
                key="btn_download_summary_report_md"
            )

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 5C: FULL-WIDTH EVALUATION RESULTS EXPLORER TABLE
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("📋 Evaluation Results Explorer")
    st.caption("Comprehensive log of every evaluated interval with baseline metrics, p-values, detector status, and analyst classification states.")

    tbl_filter = st.radio(
        "Filter Observations by Status",
        options=[
            "All Evaluated Observations",
            "Flagged Bursts (All)",
            "Unreviewed Bursts Only",
            "Reviewed Likely-Benign",
            "Suspected Attacks",
            "Confirmed Attacks"
        ],
        horizontal=True,
        key="main_evaluation_table_filter"
    )

    explorer_rows = []
    for _, row in eval_df.iterrows():
        i_num = int(row["interval"])
        cnt = int(row["packet_count"])
        p_val = float(row["upper_tail_probability"])
        is_b = bool(row["is_burst"])
        rev = reviews_map.get(i_num)

        if rev:
            a_assessment = rev.assessment
            a_cause = rev.likely_cause
            is_d = rev.is_demo
            a_grp = rev.group_name or "Ungrouped"
        elif is_b:
            a_assessment = AnalystAssessment.UNREVIEWED.value
            a_cause = "Pending Analyst Review"
            is_d = False
            a_grp = "Ungrouped"
        else:
            a_assessment = "N/A (Within Threshold)"
            a_cause = "Within Expected Variation"
            is_d = False
            a_grp = "Ungrouped"

        diff = cnt - analysis.lambda_hat
        ratio = (cnt / analysis.lambda_hat) if analysis.lambda_hat > 0 else 1.0

        r_dict = {
            "Interval": i_num,
            "Count (k)": cnt,
            "Unit": analysis.measurement_unit,
            "Baseline (λ̂)": round(analysis.lambda_hat, 2),
            "Abs Dev": f"{diff:+.1f}",
            "Ratio": f"{ratio:.2f}×",
            "Upper-Tail P": f"{p_val:.2e}" if p_val < 0.0001 else f"{p_val:.4f}",
            "Threshold α": analysis.alpha,
            "Detector Status": "🚨 Potential burst" if is_b else "✅ Within threshold",
            "Analyst Assessment": f"{a_assessment}{' (Demo)' if is_d else ''}",
            "Likely Cause": a_cause,
            "Investigation Group": a_grp
        }
        if "timestamp" in row:
            r_dict["Timestamp"] = str(row["timestamp"])
        if "attack_type" in row and pd.notna(row["attack_type"]):
            r_dict["Ground-Truth"] = str(row["attack_type"])
        elif "scenario_label" in row and pd.notna(row["scenario_label"]):
            r_dict["Ground-Truth"] = str(row["scenario_label"])

        include = False
        if tbl_filter == "All Evaluated Observations":
            include = True
        elif tbl_filter == "Flagged Bursts (All)" and is_b:
            include = True
        elif tbl_filter == "Unreviewed Bursts Only" and is_b and a_assessment == AnalystAssessment.UNREVIEWED.value:
            include = True
        elif tbl_filter == "Reviewed Likely-Benign" and a_assessment == AnalystAssessment.LIKELY_BENIGN.value:
            include = True
        elif tbl_filter == "Suspected Attacks" and a_assessment == AnalystAssessment.SUSPECTED_ATTACK.value:
            include = True
        elif tbl_filter == "Confirmed Attacks" and a_assessment == AnalystAssessment.CONFIRMED_ATTACK.value:
            include = True

        if include:
            explorer_rows.append(r_dict)

    df_explorer = pd.DataFrame(explorer_rows)
    if df_explorer.empty:
        st.info(f"No evaluated observations match filter '{tbl_filter}'.")
    else:
        st.dataframe(df_explorer, width="stretch", hide_index=True, height=320)

    # Full CSV Export
    exp_c1, exp_c2 = st.columns([2.5, 1.5])
    with exp_c1:
        st.caption(f"Displaying {len(df_explorer):,} of {len(eval_df):,} evaluated intervals. All detector parameters and review records preserved.")
    with exp_c2:
        export_full_df = df_explorer.copy()
        export_full_df["dataset"] = source_label
        csv_export_bytes = export_full_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Table Data (CSV)",
            data=csv_export_bytes,
            file_name=f"{dataset_key}_evaluation_results.csv",
            mime="text/csv",
            key="btn_download_evaluation_explorer_csv"
        )

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 6: DEEP STATISTICAL ANALYTICS & MATHEMATICAL FOUNDATIONS
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("🔬 Statistical Foundations & Mathematical Diagnostics")
    st.caption("Detailed statistical distributions, hypothesis testing formulas, model diagnostics, and educational comparisons.")

    tab_eda, tab_inference, tab_diagnostics, tab_model_compare, tab_eval = st.tabs([
        "📊 Exploratory Data Analysis & Distribution",
        "📐 Hypothesis Testing & Confidence Intervals",
        "🔬 Poisson Model Diagnostics & GoF",
        "⚖️ Poisson vs. Binomial Analysis",
        "🎯 Ground-Truth Contingency Evaluation"
    ])

    desc_base = analysis.descriptive_base
    desc_full = analysis.descriptive_full
    desc_eval = compute_descriptive_stats(eval_df["packet_count"])

    # TAB 1: EDA & DISTRIBUTION
    with tab_eda:
        st.markdown("#### 📊 Comparative Descriptive Statistics")
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
        st.dataframe(pd.DataFrame(stats_data), width="stretch", hide_index=True)

        # Data quality checks
        null_count = int(full_df["packet_count"].isna().sum())
        neg_count = int((full_df["packet_count"] < 0).sum())
        non_int_count = int((full_df["packet_count"] % 1 != 0).sum())

        dq1, dq2, dq3, dq4 = st.columns(4)
        dq1.metric("Missing Values", f"{null_count}")
        dq2.metric("Negative Values", f"{neg_count}")
        dq3.metric("Non-Integer Counts", f"{non_int_count}")
        dq4.metric("Discrete Integrity", "✅ 100% Valid")

        # Distribution Plot
        st.markdown("#### 📈 Observed Counts vs. Theoretical Poisson PMF")
        active_counts = base_df["packet_count"]
        fig_hist = go.Figure()

        fig_hist.add_trace(go.Histogram(
            x=active_counts,
            name="Empirical Counts",
            histnorm="probability density",
            xbins=dict(start=-0.5, size=1.0),
            marker=dict(color="rgba(99, 102, 241, 0.65)", line=dict(color="#818cf8", width=1)),
            opacity=0.75
        ))

        max_plot_k = int(active_counts.quantile(0.999)) + 10
        k_range = np.arange(0, max_plot_k + 1)
        theoretical_pmf = [float(poisson.pmf(k, analysis.lambda_hat)) for k in k_range]

        fig_hist.add_trace(go.Scatter(
            x=k_range,
            y=theoretical_pmf,
            mode="lines+markers",
            name=f"Fitted Poisson(λ̂ = {analysis.lambda_hat:.2f}) PMF",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=4, color="#10b981")
        ))

        fig_hist.update_layout(
            title=dict(text=f"<b>Empirical Histogram vs. Poisson(λ̂ = {analysis.lambda_hat:.2f}) PMF</b>", font=dict(size=14, color="#f8fafc")),
            paper_bgcolor="#131826",
            plot_bgcolor="#0e131f",
            font=dict(family="Inter, sans-serif", color="#cbd5e1"),
            xaxis=dict(title=f"{analysis.measurement_unit.capitalize()} Count (k)", gridcolor="rgba(255, 255, 255, 0.06)"),
            yaxis=dict(title="Probability Density", gridcolor="rgba(255, 255, 255, 0.06)"),
            margin=dict(l=20, r=20, t=60, b=35),
            height=340
        )
        st.plotly_chart(fig_hist, width="stretch")

    # TAB 2: HYPOTHESIS TESTING & CIS
    with tab_inference:
        st.markdown("#### 📐 Formal Hypothesis Formulation & Confidence Intervals")
        st.markdown(f"""
        - **Null Hypothesis ($H_0$):** Arrival count $X_i \\sim \\text{{Poisson}}(\\hat{{\\lambda}} = {analysis.lambda_hat:.2f})$ follows stationary baseline rate.
        - **Alternative Hypothesis ($H_1$):** Arrival count $X_i$ is generated by rate exceeding baseline (upper-tail burst).
        - **Test Statistic & Upper-Tail Survival:**
        $$p = P(X \\ge k) = 1 - P(X \\le k - 1) = \\sum_{{j=k}}^{{\\infty}} \\frac{{\\hat{{\\lambda}}^j e^{{-\\hat{{\\lambda}}}}}}{{j!}} = \\texttt{{poisson.sf}}(k - 1, \\hat{{\\lambda}})$$
        """)

        # Lambda Confidence Intervals
        ci_dict = analysis.lambda_ci
        if ci_dict:
            garwood_low, garwood_high = ci_dict["exact_garwood"]
            wald_low, wald_high = ci_dict["wald_asymptotic"]
            ci_table = pd.DataFrame({
                "Method": ["Garwood Exact (Chi-Square)", "Wald Asymptotic (Normal)"],
                "Confidence": ["95.0%", "95.0%"],
                "Lower Bound": [f"{garwood_low:.4f}", f"{wald_low:.4f}"],
                "Point Estimate (λ̂)": [f"{analysis.lambda_hat:.4f}", f"{analysis.lambda_hat:.4f}"],
                "Upper Bound": [f"{garwood_high:.4f}", f"{wald_high:.4f}"],
                "Width": [f"{garwood_high - garwood_low:.4f}", f"{wald_high - wald_low:.4f}"]
            })
            st.dataframe(ci_table, width="stretch", hide_index=True)

        # FDR Control
        bh_fdr = analysis.bh_fdr
        f_c1, f_c2 = st.columns(2)
        f_c1.metric("Unadjusted Flagged Bursts (p < α)", f"{analysis.burst_count} int")
        f_c2.metric("Benjamini-Hochberg FDR Discoveries", f"{bh_fdr.get('num_discoveries', 0)} int")

    # TAB 3: MODEL DIAGNOSTICS & GOF
    with tab_diagnostics:
        st.markdown("#### 🔬 Poisson Equi-Dispersion & Goodness of Fit")
        d_val = analysis.baseline_dispersion
        d_diag = "Equi-dispersed" if 0.8 <= d_val <= 1.3 else ("Overdispersed (Var > Mean)" if d_val > 1.3 else "Underdispersed")
        st.info(f"💡 **Dispersion Diagnostic:** D = s² / λ̂ = {d_val:.3f} ({d_diag}).")

        gof = analysis.goodness_of_fit
        if gof and gof.get("valid"):
            st.write(f"**Chi-Square Goodness of Fit (Pooled Bins $E_j \\ge 5.0$):** $\\chi^2 = {gof['chi2_stat']:.2f}$, $df = {gof['df']}$, $p = {gof['p_value']:.4e}$.")
            fig_gof = go.Figure()
            fig_gof.add_trace(go.Bar(x=gof["bins"], y=gof["observed_frequencies"], name="Observed", marker=dict(color="#6366f1")))
            fig_gof.add_trace(go.Bar(x=gof["bins"], y=gof["expected_frequencies"], name="Poisson Expected", marker=dict(color="#10b981")))
            fig_gof.update_layout(
                barmode="group",
                paper_bgcolor="#131826",
                plot_bgcolor="#0e131f",
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                margin=dict(l=20, r=20, t=30, b=30),
                height=300
            )
            st.plotly_chart(fig_gof, width="stretch")
        else:
            st.warning("Chi-Square Goodness of Fit requires sufficient sample size and bin coverage.")

    # TAB 4: POISSON VS. BINOMIAL
    with tab_model_compare:
        st.markdown("#### ⚖️ Poisson vs. Binomial Distribution: Law of Rare Events")
        model_comp_table = pd.DataFrame({
            "Theoretical Dimension": [
                "Trial Count (n)",
                "Arrival Probability (p)",
                "Support / Domain",
                "Variance Relationship",
                "Network Traffic Suitability"
            ],
            "Binomial Distribution B(n, p)": [
                "Fixed, finite integer n known in advance",
                "Constant probability p per trial",
                "k ∈ {0, 1, 2, ..., n}",
                "Var(X) = n·p·(1 - p) < E[X] (Under-dispersed)",
                "Unsuitable for open packet streams: network interfaces have no predefined finite trial capacity."
            ],
            "Poisson Distribution Poisson(λ)": [
                "Unbounded / Infinite (continuous limit n → ∞)",
                "Infinitesimal probability in micro-intervals (p → 0)",
                "k ∈ {0, 1, 2, ...} (countably infinite)",
                "Var(X) = λ = E[X] (Equi-dispersed)",
                "Naturally models continuous independent packet/flow arrivals in fixed time intervals [t, t + Δt]."
            ]
        })
        st.dataframe(model_comp_table, width="stretch", hide_index=True)

        comp_res = compute_poisson_binomial_comparison(analysis.lambda_hat)
        if comp_res:
            fig_binom = go.Figure()
            fig_binom.add_trace(go.Scatter(x=comp_res["k_values"], y=comp_res["poisson_pmf"], mode="lines+markers", name=f"Poisson(λ̂={analysis.lambda_hat:.2f})", line=dict(color="#10b981", width=3)))
            for sc in comp_res["scenarios"]:
                fig_binom.add_trace(go.Scatter(x=comp_res["k_values"], y=sc["pmf"], mode="lines", name=sc["name"], line=dict(dash="dash")))
            fig_binom.update_layout(
                title=dict(text="<b>Law of Rare Events Convergence: Binomial Converging to Poisson</b>", font=dict(size=14, color="#f8fafc")),
                paper_bgcolor="#131826",
                plot_bgcolor="#0e131f",
                font=dict(family="Inter, sans-serif", color="#cbd5e1"),
                margin=dict(l=20, r=20, t=50, b=30),
                height=320
            )
            st.plotly_chart(fig_binom, width="stretch")

    # TAB 5: GROUND-TRUTH CONTINGENCY (WHEN LABELS EXIST)
    with tab_eval:
        st.markdown("#### 🎯 Empirical Label Comparison (Ground-Truth / Simulation)")
        if analysis.has_labels and analysis.label_contingency:
            ctg = analysis.label_contingency
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            c_col1.metric("True Positives (Overlap)", f"{ctg['true_positives']} int", help="Intervals that contained attack/anomaly labels AND were flagged as bursts by the Poisson detector.")
            c_col2.metric("Statistical False Alarms", f"{ctg['false_positives']} int", help="Benign intervals flagged as bursts due to high arrival volume.")
            c_col3.metric("Missed Intervals", f"{ctg['false_negatives']} int", help="Labeled intervals whose volume did not exceed the statistical threshold.")
            c_col4.metric("Empirical Recall", f"{ctg['recall']:.1%}", help="Proportion of labeled intervals captured by the Poisson detector.")

            st.markdown(f"""
            <div class="contingency-card">
                <b>Summary:</b> Precision = <code>{ctg['precision']:.1%}</code>, Recall = <code>{ctg['recall']:.1%}</code>, F1-Score = <code>{ctg['f1_score']:.3f}</code> across {ctg['total_evaluated']} evaluated observations.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ Unsupervised detection: No ground-truth labels provided. Triage relies on baseline Poisson volume deviation.")

    # -------------------------------------------------------------------------
    # WORKFLOW STEP 7: MATHEMATICAL FOUNDATIONS EXPANDER
    # -------------------------------------------------------------------------
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
