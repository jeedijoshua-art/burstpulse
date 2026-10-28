"""Tests for BurstPulse Analyst Investigation Workflow, Classification Dropdowns,
Investigation Reports, and Automated Demonstration Mode.
"""

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from investigation_workflow import (
    ASSESSMENT_MARKERS,
    AnalystAssessment,
    AnalystReview,
    LikelyCause,
    ensure_reviews_state,
    generate_single_burst_report_markdown,
    generate_summary_investigation_report_markdown,
    get_dataset_reviews,
    get_investigative_recommendations,
    reset_demo_reviews,
    run_automated_demo_workflow,
    save_analyst_review,
)

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


# -----------------------------------------------------------------------------
# Unit Tests: Enums and Transparent Recommendations
# -----------------------------------------------------------------------------

def test_enums_completeness():
    """Verify all 9 required likely causes and 4 analyst assessments exist."""
    assert len(LikelyCause) == 9
    assert LikelyCause.LEGITIMATE_SURGE.value == "Legitimate traffic surge / flash crowd"
    assert LikelyCause.SCHEDULED_BACKUP.value == "Scheduled backup or bulk transfer"
    assert LikelyCause.SOFTWARE_DEPLOYMENT.value == "Software update or deployment"
    assert LikelyCause.SCHEDULED_JOB.value == "Expected scheduled job"
    assert LikelyCause.MISCONFIGURATION.value == "Misconfiguration or malfunction"
    assert LikelyCause.POSSIBLE_DDOS.value == "Possible DDoS activity"
    assert LikelyCause.POSSIBLE_SCANNING.value == "Possible scanning or probing"
    assert LikelyCause.SUSPICIOUS_PATTERN.value == "Suspicious traffic pattern"
    assert LikelyCause.INSUFFICIENT_EVIDENCE.value == "Other / insufficient evidence"

    assert len(AnalystAssessment) == 4
    assert AnalystAssessment.UNREVIEWED.value == "Unreviewed"
    assert AnalystAssessment.LIKELY_BENIGN.value == "Reviewed — likely benign"
    assert AnalystAssessment.SUSPECTED_ATTACK.value == "Suspected attack"
    assert AnalystAssessment.CONFIRMED_ATTACK.value == "Confirmed attack — evidence required"


def test_assessment_marker_styles():
    """Verify distinct colors and symbols for all 4 assessment states."""
    for assessment in AnalystAssessment:
        cfg = ASSESSMENT_MARKERS[assessment.value]
        assert "color" in cfg
        assert "symbol" in cfg
        assert "name" in cfg

    # Phase 4 exact requirements:
    # Red cross: unreviewed
    # Blue cross: reviewed benign
    # Yellow: suspected attack
    # Purple: confirmed attack
    assert ASSESSMENT_MARKERS[AnalystAssessment.UNREVIEWED.value]["color"] == "#ef4444"
    assert ASSESSMENT_MARKERS[AnalystAssessment.LIKELY_BENIGN.value]["color"] == "#3b82f6"
    assert ASSESSMENT_MARKERS[AnalystAssessment.SUSPECTED_ATTACK.value]["color"] == "#f59e0b"
    assert ASSESSMENT_MARKERS[AnalystAssessment.CONFIRMED_ATTACK.value]["color"] == "#a855f7"


def test_transparent_recommendations_non_diagnostic():
    """Verify transparent investigative recommendation templates for each cause."""
    for cause in LikelyCause:
        recs = get_investigative_recommendations(cause.value)
        assert isinstance(recs, list)
        assert len(recs) >= 3


# -----------------------------------------------------------------------------
# Unit Tests: Review Storage and Isolation
# -----------------------------------------------------------------------------

def test_session_state_review_isolation():
    """Verify reviews for dataset A do not leak into dataset B."""
    state = {}
    ensure_reviews_state(state)

    # Save review for dataset 'builtin:syn_dc'
    rev1 = AnalystReview(
        dataset_key="builtin:syn_dc",
        interval=42,
        observed_count=95,
        measurement_unit="packets",
        lambda_hat=20.5,
        p_value=1e-8,
        alpha=0.001,
        k_crit=35,
        assessment=AnalystAssessment.LIKELY_BENIGN.value,
        likely_cause=LikelyCause.SCHEDULED_BACKUP.value,
        notes="Verified scheduled backup job",
        confidence=0.9,
        evidence="Server backup log #9981",
        is_demo=False
    )
    save_analyst_review(state, rev1)

    # Save review for dataset 'builtin:syn_ddos'
    rev2 = AnalystReview(
        dataset_key="builtin:syn_ddos",
        interval=42,
        observed_count=350,
        measurement_unit="packets",
        lambda_hat=20.5,
        p_value=1e-15,
        alpha=0.001,
        k_crit=35,
        assessment=AnalystAssessment.SUSPECTED_ATTACK.value,
        likely_cause=LikelyCause.POSSIBLE_DDOS.value,
        notes="High volume spoofed flood",
        confidence=0.8,
        evidence="Flow count 100k",
        is_demo=False
    )
    save_analyst_review(state, rev2)

    dc_reviews = get_dataset_reviews(state, "builtin:syn_dc")
    ddos_reviews = get_dataset_reviews(state, "builtin:syn_ddos")

    assert len(dc_reviews) == 1
    assert len(ddos_reviews) == 1
    assert dc_reviews[42].assessment == AnalystAssessment.LIKELY_BENIGN.value
    assert ddos_reviews[42].assessment == AnalystAssessment.SUSPECTED_ATTACK.value


# -----------------------------------------------------------------------------
# Unit Tests: Automated Demo Workflow & Integrity Rules
# -----------------------------------------------------------------------------

def test_automated_demo_integrity_and_reset():
    """Verify demo populates actual flagged intervals, marks as demo, and resets safely."""
    # Synthetic evaluation DataFrame with 4 flagged bursts
    eval_df = pd.DataFrame({
        "interval": [1, 2, 3, 4, 5, 6, 7, 8],
        "packet_count": [20, 22, 85, 21, 95, 20, 110, 22],
        "is_burst": [False, False, True, False, True, False, True, False],
        "upper_tail_probability": [0.5, 0.4, 1e-8, 0.5, 1e-10, 0.5, 1e-12, 0.4],
        "attack_type": ["Normal", "Normal", "DDoS", "Normal", "Backup", "Normal", "PortScan", "Normal"]
    })

    analysis_mock = SimpleNamespace(
        evaluated_data=eval_df,
        alpha=0.001,
        k_crit=35,
        lambda_hat=20.5,
        measurement_unit="packets"
    )

    state = {}
    ensure_reviews_state(state)

    # 1. Add a genuine analyst review first at interval 3
    genuine = AnalystReview(
        dataset_key="test_ds",
        interval=3,
        observed_count=85,
        measurement_unit="packets",
        lambda_hat=20.5,
        p_value=1e-8,
        alpha=0.001,
        k_crit=35,
        assessment=AnalystAssessment.CONFIRMED_ATTACK.value,
        likely_cause=LikelyCause.POSSIBLE_DDOS.value,
        notes="Genuine human finding",
        confidence=0.95,
        evidence="PCAP trace #1",
        is_demo=False
    )
    save_analyst_review(state, genuine)

    # 2. Run automated demo
    res = run_automated_demo_workflow(state, "test_ds", analysis_mock)
    assert res["success"] is True
    assert res["num_demo_records"] > 0

    all_reviews = get_dataset_reviews(state, "test_ds")
    # Genuine review at index 3 must NOT have been overwritten by demo
    assert all_reviews[3].is_demo is False
    assert all_reviews[3].notes == "Genuine human finding"

    # Demo reviews must be marked is_demo = True
    demo_count = sum(1 for r in all_reviews.values() if r.is_demo)
    assert demo_count > 0

    # 3. Running demo again must NOT duplicate records
    res2 = run_automated_demo_workflow(state, "test_ds", analysis_mock)
    all_reviews2 = get_dataset_reviews(state, "test_ds")
    assert len(all_reviews2) == len(all_reviews)

    # 4. Reset demo reviews: demo records removed, genuine review preserved
    num_cleared = reset_demo_reviews(state, "test_ds")
    assert num_cleared == demo_count

    remaining = get_dataset_reviews(state, "test_ds")
    assert len(remaining) == 1
    assert 3 in remaining
    assert remaining[3].is_demo is False


# -----------------------------------------------------------------------------
# Unit Tests: Report Generation
# -----------------------------------------------------------------------------

def test_single_burst_and_summary_reports():
    """Verify report markdown structure and required sections."""
    review = AnalystReview(
        dataset_key="syn_dc",
        interval=5,
        observed_count=95,
        measurement_unit="packets/interval",
        lambda_hat=20.5,
        p_value=1.23e-10,
        alpha=0.001,
        k_crit=35,
        assessment=AnalystAssessment.LIKELY_BENIGN.value,
        likely_cause=LikelyCause.SCHEDULED_BACKUP.value,
        notes="High transfer rate matching daily database snapshot.",
        confidence=0.9,
        evidence="Cronjob job_db_backup completed at 02:05",
        is_demo=False
    )

    report_md = generate_single_burst_report_markdown(
        review=review,
        dataset_name="Synthetic Datacenter Spine",
        timestamp_str="2026-10-28 02:05:00"
    )

    assert "BurstPulse Network Traffic Investigation Report" in report_md
    assert "Observed Count" in report_md
    assert "95 packets/interval" in report_md
    assert "Scheduled backup or bulk transfer" in report_md
    assert "Recommended Next Investigative Steps" in report_md
    assert "Methodological Limitations & Cyberattack Distinction" in report_md

    # Test summary report
    analysis_mock = SimpleNamespace(
        evaluated_count=1000,
        burst_count=12,
        measurement_unit="packets/interval",
        lambda_hat=20.5,
        alpha=0.001
    )

    summary_md = generate_summary_investigation_report_markdown(
        reviews=[review],
        dataset_name="Synthetic Datacenter Spine",
        analysis_result=analysis_mock
    )

    assert "BurstPulse Comprehensive Investigation Summary Report" in summary_md
    assert "Reviewed — likely benign" in summary_md
    assert "#5" in summary_md


# -----------------------------------------------------------------------------
# Integration Tests: Streamlit App UI Integration
# -----------------------------------------------------------------------------

def test_main_page_renders_graph_and_table_directly():
    """Verify that both the traffic graph and the evaluation table appear on the main page."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.query_params["page"] = "analyzer"
    at.run()

    assert not at.exception, f"App raised exception: {at.exception}"

    # Verify restored subheadings for traffic graph and evaluation table exist on main page
    subheadings = [sh.value for sh in at.subheader]
    assert any("Traffic Arrival Time Series" in sh for sh in subheadings), (
        "Restored Traffic graph section must appear directly on the main page"
    )
    assert any("Evaluation Results Explorer" in sh for sh in subheadings), (
        "Evaluation explorer table must appear directly below graph on the main page"
    )

    # Verify DataFrames exist (Full-width evaluation results table rendered directly below)
    assert len(at.dataframe) >= 1, "Evaluation explorer table must be rendered directly on the main page"

    # Verify Investigation and Classification panel components exist
    selectboxes = [sb.label for sb in at.selectbox]
    assert any("Likely Cause" in label for label in selectboxes), "Likely Cause dropdown must be present"
    assert any("Analyst Assessment" in label for label in selectboxes), "Analyst Assessment dropdown must be present"

    # Verify Automated Demo button exists
    buttons = [b.label for b in at.button]
    assert any("Run Automated Demo" in b for b in buttons), "Run Automated Demo button must be present"
    assert any("Reset Demo Records" in b for b in buttons), "Reset Demo Records button must be present"


def test_automated_demo_execution_via_apptest():
    """Run the automated demo in AppTest and verify state updates without error."""
    at = AppTest.from_file(APP_PATH, default_timeout=25)
    at.query_params["page"] = "analyzer"
    at.run()

    # Find the 'Run Automated Demo' button and click it
    demo_btn = None
    for b in at.button:
        if "Run Automated Demo" in b.label:
            demo_btn = b
            break

    assert demo_btn is not None, "Automated Demo button found"
    demo_btn.click().run()

    assert not at.exception, f"App raised exception after demo click: {at.exception}"

    # Verify success banner or session state update
    success_banners = [s.value for s in at.success]
    assert any("DEMO ACTIVE" in s or "demonstration" in s.lower() for s in success_banners), (
        f"Expected demo notification, got: {success_banners}"
    )

    # Verify reviews were added in session state
    db = at.session_state.get("analyst_reviews_db", {})
    assert len(db) > 0, "Demo reviews must be present in session state"
    total_reviews = sum(len(rev_map) for rev_map in db.values())
    assert total_reviews > 0, "Demo reviews must contain review records"
