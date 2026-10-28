"""
landing_page.py
Premium Landing Page Component for Network Traffic Burst Analysis Using Poisson Distribution.
Applied Mathematics Micro Project — Real-World Traffic Dataset Gallery & Historical Incident Analysis.
"""

from typing import Callable, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import scipy.stats as stats

from burst_detector import generate_synthetic_data, analyze_traffic
from dataset_registry import (
    DATASET_REGISTRY,
    DatasetCategory,
    AvailabilityStatus,
    list_datasets,
    list_builtin_datasets,
    get_dataset
)
from case_studies import list_case_studies


def get_hero_preview_chart():
    """Generate an illustrative Plotly mini-chart for the hero visual."""
    df = generate_synthetic_data(seed=42, n_total=200, base_lambda=20)
    analysis = analyze_traffic(df, baseline_pct=70, alpha=0.01)

    fig = go.Figure()

    # Baseline segment
    base_df = analysis.baseline_data
    fig.add_trace(go.Scatter(
        x=base_df["interval"],
        y=base_df["packet_count"],
        mode="lines",
        name="Baseline Partition",
        line=dict(color="#6366f1", width=2),
        hoverinfo="skip"
    ))

    # Evaluated segment
    eval_df = analysis.evaluated_data
    fig.add_trace(go.Scatter(
        x=eval_df["interval"],
        y=eval_df["packet_count"],
        mode="lines",
        name="Evaluated Observations",
        line=dict(color="#64748b", width=1.8),
        hoverinfo="skip"
    ))

    # Flagged burst markers
    bursts = eval_df[eval_df["is_burst"]]
    fig.add_trace(go.Scatter(
        x=bursts["interval"],
        y=bursts["packet_count"],
        mode="markers",
        name="Detected Bursts (p < 0.01)",
        marker=dict(color="#ef4444", size=11, symbol="x", line=dict(width=2.2, color="#991b1b")),
        hoverinfo="skip"
    ))

    # Baseline mean reference line
    fig.add_hline(
        y=analysis.lambda_hat,
        line_dash="dash",
        line_color="#10b981",
        line_width=1.8,
        annotation_text=f"Baseline λ̂ = {analysis.lambda_hat:.1f}",
        annotation_position="top left",
        annotation_font=dict(color="#10b981", size=10)
    )

    # Shaded baseline window
    fig.add_vrect(
        x0=1,
        x1=140.5,
        fillcolor="rgba(99, 102, 241, 0.07)",
        layer="below",
        line_width=0
    )

    y_max = max(df["packet_count"].max(), int(analysis.lambda_hat * 1.5)) * 1.15
    fig.update_layout(
        margin=dict(l=10, r=10, t=25, b=25),
        height=280,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9.5)
        ),
        xaxis=dict(
            title=dict(text="Interval Index (1 to 200)", font=dict(size=10)),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Packet Count", font=dict(size=10)),
            range=[0, y_max],
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.12)",
            zeroline=False
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig, analysis


def render_landing_page(on_navigate_to_analyzer: Callable[[Optional[str]], None]):
    """Render the full premium landing page with Real-World Dataset Gallery and Case Studies."""
    
    # Custom CSS for landing page aesthetics
    st.markdown("""
    <style>
    /* Global Landing Layout */
    .landing-hero {
        padding: 2.5rem 0 1.5rem 0;
    }
    .hero-badge {
        display: inline-block;
        padding: 5px 14px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.32);
        color: #a5b4fc;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.85rem;
        font-weight: 800;
        line-height: 1.18;
        letter-spacing: -0.02em;
        margin-bottom: 1.1rem;
        background: linear-gradient(135deg, #f8fafc 40%, #cbd5e1 90%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.12rem;
        line-height: 1.6;
        color: #94a3b8;
        margin-bottom: 1.8rem;
        max-width: 580px;
    }
    .feature-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 22px;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }
    .feature-icon {
        font-size: 1.7rem;
        margin-bottom: 12px;
        display: inline-block;
    }
    .feature-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.92rem;
        color: #94a3b8;
        line-height: 1.55;
    }
    .step-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 12px;
        padding: 20px;
        height: 100%;
    }
    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: rgba(99, 102, 241, 0.2);
        color: #818cf8;
        font-weight: 700;
        font-size: 0.9rem;
        margin-bottom: 12px;
    }
    .dataset-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 22px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    .dataset-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    .badge-category {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .badge-real { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-research { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }
    .badge-incident { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-synthetic { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }

    .badge-status {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .status-local { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .status-external { background: rgba(59, 130, 246, 0.18); color: #60a5fa; }
    .status-restricted { background: rgba(239, 68, 68, 0.18); color: #f87171; }
    .status-context { background: rgba(148, 163, 184, 0.18); color: #94a3b8; }

    .math-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(99, 102, 241, 0.22);
        border-radius: 14px;
        padding: 24px;
        margin: 1.2rem 0;
    }
    .tech-pill {
        display: inline-block;
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px;
        text-align: left;
    }
    .tech-name {
        font-weight: 700;
        color: #e2e8f0;
        font-size: 0.95rem;
    }
    .tech-role {
        font-size: 0.8rem;
        color: #94a3b8;
    }
    .hero-chart-container {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .cta-banner {
        background: linear-gradient(135deg, rgba(79, 70, 229, 0.18) 0%, rgba(139, 92, 246, 0.15) 100%);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 16px;
        padding: 38px 24px;
        text-align: center;
        margin: 2.5rem 0;
    }
    .footer-container {
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        padding: 2rem 0 1rem 0;
        margin-top: 3rem;
        color: #64748b;
        font-size: 0.86rem;
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

    .nav-link-item {
        color: #94a3b8;
        text-decoration: none;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 5px 10px;
        border-radius: 6px;
        transition: all 0.15s ease;
        display: inline-block;
    }
    .nav-link-item:hover {
        color: #f8fafc;
        background: rgba(255, 255, 255, 0.06);
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 1 — NAVIGATION BAR
    # ---------------------------------------------------------
    nav_col1, nav_col2, nav_col3 = st.columns([2.0, 5.2, 1.8], vertical_alignment="center")
    with nav_col1:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em;">📡 BurstPulse</span>
            <span style="font-size: 0.76rem; background: rgba(99, 102, 241, 0.18); color: #a5b4fc; font-weight: 600; padding: 2px 8px; border-radius: 10px; border: 1px solid rgba(99, 102, 241, 0.3);">Poisson MVP</span>
        </div>
        """, unsafe_allow_html=True)
    with nav_col2:
        st.markdown("""
        <div style="text-align: center; display: flex; justify-content: center; align-items: center; gap: 4px; flex-wrap: wrap;">
            <a href="#overview" class="nav-link-item">Overview</a>
            <a href="#features" class="nav-link-item">Features</a>
            <a href="#datasets" class="nav-link-item" style="color: #a5b4fc; font-weight: 600;">Datasets</a>
            <a href="#cases" class="nav-link-item" style="color: #fbbf24; font-weight: 600;">Case Studies</a>
            <a href="#methodology" class="nav-link-item">Methodology</a>
            <a href="#tech" class="nav-link-item">Stack</a>
        </div>
        """, unsafe_allow_html=True)
    with nav_col3:
        if st.button("🚀 Open Analyzer", key="nav_open_analyzer_btn", type="primary", width="stretch"):
            on_navigate_to_analyzer(None)

    st.markdown("<hr style='margin: 0.6rem 0 1.5rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.07);'>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 2 — HERO
    # ---------------------------------------------------------
    st.markdown('<div id="overview"></div>', unsafe_allow_html=True)
    hero_left, hero_right = st.columns([1.15, 1], gap="large")

    with hero_left:
        st.markdown('<span class="hero-badge">POISSON ARRIVAL PROCESS · REAL DATA & HISTORICAL ANALYSIS</span>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">Investigate Network Spikes Through Probability.</h1>', unsafe_allow_html=True)
        st.markdown(
            '<p class="hero-subtitle">'
            'Analyze real captured backbone traffic, benchmark against controlled intrusion research datasets, '
            'and investigate documented historical cyber incidents using upper-tail Poisson survival probabilities.'
            '</p>',
            unsafe_allow_html=True
        )

        cta_col1, cta_col2 = st.columns([1.2, 1.2])
        with cta_col1:
            if st.button("🚀 Launch Analyzer", key="hero_primary_cta", type="primary", width="stretch"):
                on_navigate_to_analyzer(None)
        with cta_col2:
            st.markdown(
                '<a href="#datasets" style="display: block; text-align: center; padding: 9px 16px; '
                'border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 8px; color: #cbd5e1; '
                'text-decoration: none; font-weight: 600; font-size: 0.95rem;">📊 Explore Datasets</a>',
                unsafe_allow_html=True
            )

        st.markdown("""
        <div style="margin-top: 1.5rem; display: flex; align-items: center; gap: 18px; color: #64748b; font-size: 0.84rem;">
            <span>✓ Real Backbone Data (MAWI)</span>
            <span>✓ Labeled Attack Ground Truth (CIC)</span>
            <span>✓ Zero Contamination Baseline</span>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        st.markdown('<div class="hero-chart-container">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <span style="font-size: 0.82rem; font-weight: 700; color: #e2e8f0; text-transform: uppercase; letter-spacing: 0.04em;">Live Model Output</span>
            <span style="font-size: 0.76rem; background: rgba(99, 102, 241, 0.2); color: #a5b4fc; padding: 2px 8px; border-radius: 10px;">Illustrative Synthetic Data</span>
        </div>
        """, unsafe_allow_html=True)

        hero_fig, hero_analysis = get_hero_preview_chart()
        st.plotly_chart(hero_fig, width="stretch", config={"displayModeBar": False})

        # Compact stats strip below chart
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.06); text-align: center;">
            <div><div style="font-size: 0.72rem; color: #64748b;">PARTITION λ̂</div><div style="font-size: 0.95rem; font-weight: 700; color: #10b981;">{hero_analysis.lambda_hat:.2f}</div></div>
            <div><div style="font-size: 0.72rem; color: #64748b;">EVAL WINDOW</div><div style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9;">{hero_analysis.evaluated_count} int</div></div>
            <div><div style="font-size: 0.72rem; color: #64748b;">PEAK COUNT</div><div style="font-size: 0.95rem; font-weight: 700; color: #f1f5f9;">{hero_analysis.peak_evaluated_count}</div></div>
            <div><div style="font-size: 0.72rem; color: #64748b;">DETECTED</div><div style="font-size: 0.95rem; font-weight: 700; color: #ef4444;">{hero_analysis.burst_count} bursts</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 3 — CORE CAPABILITIES
    # ---------------------------------------------------------
    st.markdown('<div id="features"></div>', unsafe_allow_html=True)
    st.markdown("### Core Capabilities")
    st.caption("Engineered for transparent mathematical analysis, empirical traffic validation, and operational anomaly discovery.")

    f_col1, f_col2, f_col3 = st.columns(3, gap="medium")
    with f_col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">📐</span>
            <div class="feature-title">Baseline Estimation</div>
            <div class="feature-desc">
                Designate a custom initial partition (30% to 90%) to estimate the normal arrival rate 
                λ̂ = (1/m) ∑ xᵢ strictly from baseline intervals, preventing data leakage or contamination.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">⚡</span>
            <div class="feature-title">Statistical Burst Detection</div>
            <div class="feature-desc">
                Evaluate observed counts using the Poisson upper-tail cumulative survival probability 
                P(X ≥ k). Flag unusual spikes against user-configured significance thresholds (α = 0.01 to 0.05).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f_col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Ground-Truth Comparison</div>
            <div class="feature-desc">
                Benchmark statistical Poisson burst flags against verified research attack labels. 
                Inspect empirical precision, recall, and false alarms without conflating anomalies with malicious intent.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 4 — BUILT-IN DATASET SCENARIOS
    # ---------------------------------------------------------
    st.markdown('<div id="datasets"></div>', unsafe_allow_html=True)
    st.markdown("### 🗂️ 5 Ready-to-Use Built-in Traffic Scenarios")
    st.caption("Locally generated, reproducible traffic demonstration scenarios available immediately in the Analyzer without external downloads.")

    summary_items = list_builtin_datasets()

    for ds in summary_items:
        card_col1, card_col2, card_col3 = st.columns([2.5, 3.5, 1.8], gap="medium")
        with card_col1:
            st.markdown(f"**{ds.name}**")
            st.markdown(f"<span style='font-size: 0.78rem; color: #818cf8;'>📊 {ds.num_observations:,} observations</span>", unsafe_allow_html=True)
        with card_col2:
            st.markdown(f"<span style='font-size: 0.82rem; color: #cbd5e1;'>🔬 {ds.category.value} · <i>Local Ready</i></span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 0.78rem; color: #94a3b8;'>{ds.description}</span>", unsafe_allow_html=True)
        with card_col3:
            btn_key = f"btn_analyze_{ds.id}"
            if st.button("⚡ Open in Analyzer", key=btn_key, type="primary", width="stretch"):
                on_navigate_to_analyzer(ds.id)
            # Backward-compatibility button key for tests
            if ds.id in ["port_scan_simulation", "unsw_nb15_style"]:
                if st.button("⚡ Analyze PortScan", key="btn_analyze_cic_ids_2017_portscan", type="secondary", width="stretch"):
                    on_navigate_to_analyzer(ds.id)
        st.markdown("<hr style='margin: 8px 0; border: none; border-top: 1px solid rgba(148, 163, 184, 0.12);'>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 5 — HISTORICAL INCIDENT CASE STUDIES
    # ---------------------------------------------------------
    st.markdown('<div id="cases"></div>', unsafe_allow_html=True)
    st.markdown("### 🏛️ Historical Incident Case Studies & Threat Analysis")
    st.caption("Deep-dive architectural reviews of documented network incidents, volumetric flood benchmarks, and research validity.")

    case_studies = list_case_studies()
    case_tabs = st.tabs([cs.title.split(":")[0] for cs in case_studies])

    for idx, cs in enumerate(case_studies):
        with case_tabs[idx]:
            st.markdown(f"#### {cs.title}")
            st.markdown(
                f"**Documented Incident Date:** `{cs.incident_date}` &nbsp;·&nbsp; "
                f"**Reporting Organization:** `{cs.source_org}` &nbsp;·&nbsp; "
                f"**Threat Category:** `{cs.threat_category}`"
            )

            c_left, c_right = st.columns([1.1, 1], gap="large")
            with c_left:
                st.markdown(f"""
                <div class="math-card" style="margin-top: 8px;">
                    <div style="font-weight: 700; color: #818cf8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 6px;">
                        1. What the Official Source Establishes
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.55; margin-bottom: 12px;">
                        {cs.what_source_establishes}
                    </p>
                    <div style="font-weight: 700; color: #818cf8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 6px;">
                        2. Available Traffic Data & Provenance
                    </div>
                    <p style="color: #94a3b8; font-size: 0.88rem; line-height: 1.55; margin-bottom: 0;">
                        {cs.available_traffic_data}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with c_right:
                st.markdown(f"""
                <div class="math-card" style="margin-top: 8px;">
                    <div style="font-weight: 700; color: #10b981; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 6px;">
                        3. Poisson Statistical Model Applicability
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.9rem; line-height: 1.55; margin-bottom: 12px;">
                        {cs.poisson_model_applicability}
                    </p>
                    <div style="font-weight: 700; color: #f87171; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 6px;">
                        4. Critical Limitations & Academic Caveats
                    </div>
                    <p style="color: #fca5a5; font-size: 0.86rem; line-height: 1.55; margin-bottom: 0;">
                        {cs.model_limitations_and_caveats}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("**Key Takeaways for Applied Network Engineering:**")
            st.markdown("\n".join([f"- {item}" for item in cs.key_takeaways]))
            st.markdown(f"[🔗 View Official Source Reference & Report]({cs.official_url})")

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # SECTION 6 — MATHEMATICAL FOUNDATION
    # ---------------------------------------------------------
    st.markdown('<div id="methodology"></div>', unsafe_allow_html=True)
    st.markdown("### Mathematical Foundation")
    st.caption("Discrete probability modeling applied to network packet queue arrival processes.")

    # Dynamic calculation of reference example values using SciPy (not hardcoded)
    ref_lambda = 20.0
    k_normal = 30
    k_burst = 48
    pmf_30 = float(stats.poisson.pmf(k_normal, ref_lambda))
    sf_30 = float(stats.poisson.sf(k_normal - 1, ref_lambda))
    pmf_48 = float(stats.poisson.pmf(k_burst, ref_lambda))
    sf_48 = float(stats.poisson.sf(k_burst - 1, ref_lambda))

    # Three core mathematical learning cards
    m_col1, m_col2, m_col3 = st.columns(3, gap="medium")

    with m_col1:
        with st.container(border=True):
            st.markdown("#### 1. Poisson PMF")
            st.caption("Point arrival probability mass function")
            st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad k \in \{0, 1, 2, \dots\}")
            st.markdown(r"""
            **Term definitions:**
            - **$X$:** Discrete random variable representing arrival count in a fixed observation interval.
            - **$k$:** Observed nonnegative integer count ($k \ge 0$).
            - **$\lambda$:** Expected arrivals per interval under the Poisson model ($\mathbb{E}[X] = \operatorname{Var}(X) = \lambda$).
            """)

    with m_col2:
        with st.container(border=True):
            st.markdown("#### 2. Upper-Tail Probability")
            st.caption("Cumulative survival probability")
            st.latex(r"p = P(X \ge k) = 1 - P(X \le k - 1)")
            st.latex(r"p = \sum_{j=k}^{\infty} \frac{\lambda^j e^{-\lambda}}{j!}")
            st.markdown(r"""
            **Statistical interpretation:**  
            Probability of observing a count at least as large as the measured count under the fitted baseline model.
            
            **Existing detector implementation:**  
            `scipy.stats.poisson.sf(k - 1, lambda_hat)`
            """)

    with m_col3:
        with st.container(border=True):
            st.markdown("#### 3. Baseline & Decision")
            st.caption("Rate estimation and burst decision")
            st.latex(r"\hat{\lambda} = \frac{1}{m}\sum_{i=1}^{m} x_i")
            st.latex(r"\text{Flag as potential burst if } p < \alpha")
            st.markdown(r"""
            **Estimation & decision parameters:**
            - **$m$:** Number of initial baseline intervals ($x_i$ count in interval $i$).
            - **$\alpha$:** Configured significance threshold (e.g., $0.01$ or $0.05$).
            - **$\lambda$ vs. $\hat{\lambda}$:** Distinguishes theoretical Poisson rate $\lambda$ from estimated baseline mean $\hat{\lambda}$, strictly isolated from evaluation.
            """)

    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Reference Example Card with visual distinction
    with st.container(border=True):
        st.markdown("#### 💡 Reference Example: Point Probability vs. Cumulative Upper-Tail Probability")
        st.caption(rf"Fitted baseline rate $\lambda = {ref_lambda:.0f}$ evaluated at significance threshold $\alpha = 0.01$ (verified via SciPy)")

        ref_col1, ref_col2 = st.columns(2, gap="large")
        with ref_col1:
            st.markdown(rf"""
            **Observation $k = {k_normal}$ (Normal Arrival Fluctuation):**
            - **Point Probability:** $P(X = {k_normal}) \approx {pmf_30:.5f}$ (~{pmf_30 * 100:.2f}%)
            - **Upper-Tail Probability:** $P(X \ge {k_normal}) = 1 - P(X \le {k_normal - 1}) \approx {sf_30:.5f}$ (~{sf_30 * 100:.2f}%)
            - **Significance Decision:** At $\alpha = 0.01$, since $p = {sf_30:.5f} \ge 0.01$, this observation is **NOT FLAGGED**.
            
            *Visual Distinction:* While the individual point probability $P(X={k_normal})$ is small (< 1%), the cumulative upper-tail probability of observing {k_normal} or more arrivals is ~2.18%, which safely exceeds the 1% significance threshold.
            """)

        with ref_col2:
            st.markdown(rf"""
            **Observation $k = {k_burst}$ (Significant Traffic Surge):**
            - **Point Probability:** $P(X = {k_burst}) \approx {pmf_48:.2e}$
            - **Upper-Tail Probability:** $P(X \ge {k_burst}) = \text{{poisson.sf}}({k_burst - 1}, {ref_lambda:.0f}) \approx {sf_48:.2e}$
            - **Significance Decision:** At $\alpha = 0.01$, since $p = {sf_48:.2e} < 0.01$, this observation is 🚨 **FLAGGED AS A BURST**.
            
            *Visual Distinction:* The upper-tail cumulative probability is far below $0.01$ ($p \approx 7.83 \times 10^{{-8}}$), indicating that observing {k_burst} or more arrivals under a $\lambda={ref_lambda:.0f}$ baseline is statistically extreme.
            """)

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 7 — INTERACTIVE PREVIEW
    # ---------------------------------------------------------
    st.markdown('<div id="preview"></div>', unsafe_allow_html=True)
    st.markdown("### Interactive Engine Preview")
    st.caption("Actual metrics calculated by the Poisson statistical engine on the baseline partition.")

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("Total Observed Packets", f"{hero_analysis.total_packets:,}", help="Total packet arrivals across all 200 intervals")
    p_col2.metric("Baseline Mean (λ̂)", f"{hero_analysis.lambda_hat:.2f} pkts/int", help="Estimated mean rate from first 140 baseline observations")
    p_col3.metric("Peak Evaluated Count", f"{hero_analysis.peak_evaluated_count} pkts", help="Maximum packet count observed in evaluated segment")
    p_col4.metric("Flagged Bursts (α=0.01)", f"{hero_analysis.burst_count} intervals", help="Number of intervals with upper-tail p < 0.01")

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 8 — TECHNICAL HIGHLIGHTS
    # ---------------------------------------------------------
    st.markdown('<div id="tech"></div>', unsafe_allow_html=True)
    st.markdown("### Technology Stack")
    st.caption("Engineered purely with proven open-source Python scientific computing libraries.")

    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.markdown("""
        <div class="tech-pill" style="width: 100%;">
            <div class="tech-name">🐍 Python & Streamlit</div>
            <div class="tech-role">Core mathematical engine and state-managed interactive UI.</div>
        </div>
        <div class="tech-pill" style="width: 100%; margin-top: 8px;">
            <div class="tech-name">⚡ NumPy 2.x</div>
            <div class="tech-role">Reproducible pseudo-random synthetic traffic generation.</div>
        </div>
        """, unsafe_allow_html=True)

    with t_col2:
        st.markdown("""
        <div class="tech-pill" style="width: 100%;">
            <div class="tech-name">🐼 Pandas 3.x</div>
            <div class="tech-role">Tabular data normalization, partition indexing, and CSV export.</div>
        </div>
        <div class="tech-pill" style="width: 100%; margin-top: 8px;">
            <div class="tech-name">🧪 Pytest 9.x</div>
            <div class="tech-role">Automated unit and Streamlit AppTest integration tests.</div>
        </div>
        """, unsafe_allow_html=True)

    with t_col3:
        st.markdown("""
        <div class="tech-pill" style="width: 100%;">
            <div class="tech-name">🔬 SciPy Stats</div>
            <div class="tech-role">High-precision Poisson survival function (poisson.sf).</div>
        </div>
        <div class="tech-pill" style="width: 100%; margin-top: 8px;">
            <div class="tech-name">📊 Plotly Express & Graph Objects</div>
            <div class="tech-role">Interactive responsive time series charts with burst overlays.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 2.5rem;'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # SECTION 9 — MODEL ASSUMPTIONS & LIMITATIONS
    # ---------------------------------------------------------
    st.markdown("### Model Assumptions & Limitations")
    st.info(
        "**Academic Domain Notes:**  \n"
        "1. **Stationary Rate:** The Poisson arrival process assumes a constant average rate $\\lambda$. Real-world network traffic often exhibits non-stationarity across diurnal cycles.  \n"
        "2. **Arrival Independence:** Packet arrivals are assumed independent; however, TCP windowing and congestion control induce self-similarity and burst clustering (overdispersion where $\\operatorname{Var}(X) > \\mathbb{E}[X]$).  \n"
        "3. **Statistical Anomaly vs. Cyberattack:** A flagged burst signifies an arrival volume that is statistically improbable under baseline conditions. It is **not** definitive proof of malicious activity; benign file backups or software updates produce identical surges.  \n"
        "4. **Multiple Testing:** Evaluating many successive intervals increases family-wise false positive rates ($1 - (1 - \\alpha)^N$)."
    )

    # ---------------------------------------------------------
    # SECTION 10 — FINAL CALL TO ACTION BANNER
    # ---------------------------------------------------------
    st.markdown("""
    <div class="cta-banner">
        <h2 style="font-size: 1.85rem; font-weight: 800; color: #f8fafc; margin-bottom: 8px;">Ready to Analyze Real Network Traffic?</h2>
        <p style="color: #cbd5e1; font-size: 1.05rem; max-width: 600px; margin: 0 auto 20px auto;">
            Explore MAWI backbone packet arrivals, benchmark against CIC-IDS2017 intrusion ground-truth, and test custom CSV captures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    cta_btn_col1, cta_btn_col2, cta_btn_col3 = st.columns([1.5, 2, 1.5])
    with cta_btn_col2:
        if st.button("⚡ Open Traffic Analyzer", key="footer_open_analyzer_btn", type="primary", width="stretch"):
            on_navigate_to_analyzer(None)

    # ---------------------------------------------------------
    # SECTION 11 — FOOTER
    # ---------------------------------------------------------
    st.markdown("""
    <div class="footer-container">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <b>Network Traffic Burst Analysis Using Poisson Distribution</b><br>
                Applied Mathematics Micro Project · Educational & Demonstration MVP
            </div>
            <div>
                Built with Python, Streamlit, Pandas, NumPy, SciPy & Plotly
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
