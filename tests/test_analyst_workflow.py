"""Tests for BurstPulse Analyst Investigation Workflow, Classification Dropdowns,
Investigation Reports, Grouping, Analytics, and Automated Demonstration Mode.
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
    InvestigationGroup,
    LikelyCause,
    calculate_group_analytics,
    delete_investigation_group,
    ensure_groups_state,
    ensure_reviews_state,
    generate_group_investigation_report_markdown,
    generate_single_burst_report_markdown,
    generate_summary_investigation_report_markdown,
    get_dataset_groups,
    get_dataset_reviews,
    get_investigative_recommendations,
    reset_demo_groups,
    reset_demo_reviews,
    run_automated_demo_workflow,
    save_analyst_review,
    save_investigation_group,
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
# Unit Tests: Named Investigation Groups & Analytics (Phases 2 & 3)
# -----------------------------------------------------------------------------

def test_named_investigation_group_lifecycle_and_nonconsecutive_membership():
    """Verify group creation with nonconsecutive members, sync with reviews, editing, and ungrouping."""
    state = {}
    ensure_groups_state(state)
    ensure_reviews_state(state)

    eval_df = pd.DataFrame({
        "interval": [10, 20, 30, 40],
        "packet_count": [80, 95, 110, 120],
        "upper_tail_probability": [1e-6, 1e-8, 1e-10, 1e-12],
        "is_burst": [True, True, True, True]
    })

    # Create Group 1 with nonconsecutive intervals [10, 30]
    grp = InvestigationGroup(
        group_id="grp_backup_1",
        dataset_key="test_dataset",
        name="Scheduled Backup Surge",
        likely_cause=LikelyCause.SCHEDULED_BACKUP.value,
        assessment=AnalystAssessment.LIKELY_BENIGN.value,
        description="Daily incremental database snapshot transfer window.",
        notes="Matches authorized storage gateway IP.",
        supporting_evidence="Cronjob job_db_backup completed at 02:00",
        confidence=0.9,
        member_intervals=[10, 30],
        is_demo=False
    )
    save_investigation_group(state, grp, eval_df=eval_df, lambda_hat=20.0, alpha=0.001, k_crit=35, unit="packets")

    # Verify group was saved
    ds_groups = get_dataset_groups(state, "test_dataset")
    assert "grp_backup_1" in ds_groups
    assert ds_groups["grp_backup_1"].member_intervals == [10, 30]

    # Verify individual reviews were synchronized with group classification
    reviews = get_dataset_reviews(state, "test_dataset")
    assert 10 in reviews and 30 in reviews
    assert reviews[10].assessment == AnalystAssessment.LIKELY_BENIGN.value
    assert reviews[10].group_id == "grp_backup_1"
    assert reviews[10].group_name == "Scheduled Backup Surge"
    assert reviews[30].assessment == AnalystAssessment.LIKELY_BENIGN.value

    # Verify group analytics derived correctly
    analytics = calculate_group_analytics(grp, eval_df, lambda_hat=20.0, alpha=0.001, measurement_unit="packets")
    assert analytics["num_members"] == 2
    assert analytics["unique_intervals"] == [10, 30]
    assert analytics["earliest_interval"] == 10
    assert analytics["latest_interval"] == 30
    assert analytics["interval_span"] == 21  # 30 - 10 + 1 = 21
    assert analytics["is_continuous_burst"] is False
    assert "nonconsecutive" in analytics["span_note"]
    assert analytics["min_count"] == 80
    assert analytics["max_count"] == 110
    assert analytics["mean_count"] == 95.0
    assert analytics["median_count"] == 95.0
    assert analytics["mean_deviation"] == 75.0  # 95 - 20 = 75
    assert analytics["relative_ratio"] == 4.75   # 95 / 20 = 4.75
    assert analytics["num_below_alpha"] == 2
    assert analytics["prop_below_alpha"] == 1.0

    # Test editing the group name and members
    grp.name = "Updated Storage Window"
    grp.member_intervals = [10, 20, 30]
    save_investigation_group(state, grp, eval_df=eval_df, lambda_hat=20.0, alpha=0.001, k_crit=35, unit="packets")

    updated_grp = get_dataset_groups(state, "test_dataset")["grp_backup_1"]
    assert updated_grp.name == "Updated Storage Window"
    assert updated_grp.member_intervals == [10, 20, 30]

    # Test deleting/ungrouping
    res = delete_investigation_group(state, "test_dataset", "grp_backup_1")
    assert res is True
    assert "grp_backup_1" not in get_dataset_groups(state, "test_dataset")
    # Interval reviews remain intact but group link is unbound
    assert reviews[10].group_id is None
    assert reviews[10].group_name is None


def test_prevent_conflicting_group_memberships():
    """Ensure a burst interval cannot belong to conflicting active investigations."""
    state = {}
    eval_df = pd.DataFrame({
        "interval": [1, 2, 3],
        "packet_count": [50, 60, 70],
        "upper_tail_probability": [1e-5, 1e-6, 1e-7],
        "is_burst": [True, True, True]
    })

    grp1 = InvestigationGroup(
        group_id="grp_1", dataset_key="ds_test", name="Group 1",
        member_intervals=[1, 2]
    )
    save_investigation_group(state, grp1, eval_df=eval_df)

    # Save Group 2 which claims interval 2
    grp2 = InvestigationGroup(
        group_id="grp_2", dataset_key="ds_test", name="Group 2",
        member_intervals=[2, 3]
    )
    save_investigation_group(state, grp2, eval_df=eval_df)

    groups = get_dataset_groups(state, "ds_test")
    # Interval 2 must now be in Group 2, and removed from Group 1
    assert groups["grp_1"].member_intervals == [1]
    assert groups["grp_2"].member_intervals == [2, 3]


def test_group_investigation_report_generation():
    """Verify group report markdown includes all required sections and transparent recommendations."""
    eval_df = pd.DataFrame({
        "interval": [100, 105],
        "packet_count": [250, 310],
        "upper_tail_probability": [1e-12, 1e-15],
        "is_burst": [True, True],
        "attack_type": ["DDoS-SYN", "DDoS-SYN"]
    })

    grp = InvestigationGroup(
        group_id="grp_ddos_eval",
        dataset_key="cse_cic_ids2018",
        name="Possible DDoS Investigation",
        likely_cause=LikelyCause.POSSIBLE_DDOS.value,
        assessment=AnalystAssessment.SUSPECTED_ATTACK.value,
        description="High volume SYN arrival surge.",
        notes="Observed significant server latency spike.",
        supporting_evidence="NetFlow trace #998",
        confidence=0.85,
        member_intervals=[100, 105]
    )

    analytics = calculate_group_analytics(grp, eval_df, lambda_hat=25.0, alpha=0.001, measurement_unit="packets")
    report_md = generate_group_investigation_report_markdown(
        grp, analytics, "CSE-CIC-IDS2018 Style", "Synthetic Benchmark", "packets"
    )

    assert "BurstPulse Group Investigation Report: Possible DDoS Investigation" in report_md
    assert "Executive Summary & Span Telemetry" in report_md
    assert "Member Telemetry Breakdown" in report_md
    assert "Recommended Next Investigative Steps" in report_md
    assert "Methodological Limitations & Cyberattack Distinction" in report_md
    assert "DDoS-SYN" in report_md
    assert "250" in report_md and "310" in report_md


# -----------------------------------------------------------------------------
# Unit Tests: Automated Demo Workflow & Integrity Rules
# -----------------------------------------------------------------------------

def test_automated_demo_integrity_and_reset():
    """Verify demo populates actual flagged intervals, marks as demo, and resets safely."""
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
    ensure_groups_state(state)

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
    assert len(res["created_groups"]) > 0

    all_reviews = get_dataset_reviews(state, "test_ds")
    all_groups = get_dataset_groups(state, "test_ds")

    # Genuine review at index 3 must NOT have been overwritten by demo
    assert all_reviews[3].is_demo is False
    assert all_reviews[3].notes == "Genuine human finding"

    # Demo reviews and groups must be marked is_demo = True
    for grp in all_groups.values():
        assert grp.is_demo is True

    # 3. Running demo again must NOT duplicate records
    res2 = run_automated_demo_workflow(state, "test_ds", analysis_mock)
    all_groups2 = get_dataset_groups(state, "test_ds")
    assert len(all_groups2) == len(all_groups)

    # 4. Reset demo reviews & groups: demo records removed, genuine review preserved
    num_cleared_reviews = reset_demo_reviews(state, "test_ds")
    num_cleared_groups = reset_demo_groups(state, "test_ds")
    assert num_cleared_reviews > 0
    assert num_cleared_groups > 0

    remaining_reviews = get_dataset_reviews(state, "test_ds")
    remaining_groups = get_dataset_groups(state, "test_ds")

    assert len(remaining_reviews) == 1
    assert 3 in remaining_reviews
    assert remaining_reviews[3].is_demo is False
    assert len(remaining_groups) == 0


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
    assert any("Category Views" in sh for sh in subheadings), (
        "Four Category Views section must appear on the main page"
    )
    assert any("Investigation Groups" in sh for sh in subheadings), (
        "Named Investigation Groups section must appear on the main page"
    )

    # Verify DataFrames exist
    assert len(at.dataframe) >= 1, "Evaluation explorer table must be rendered directly on the main page"

    # Verify Automated Demo and Reset buttons exist
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

    # Verify success banner rendered immediately
    success_banners = [s.value for s in at.success]
    assert any("DEMO ACTIVE" in s or "demonstration" in s.lower() for s in success_banners), (
        f"Expected demo notification, got: {success_banners}"
    )

    # Verify reviews and groups were added in session state
    db = at.session_state.get("analyst_reviews_db", {})
    assert len(db) > 0, "Demo reviews must be present in session state"

    groups_db = at.session_state.get("investigation_groups_db", {})
    assert len(groups_db) > 0, "Demo groups must be present in session state"
