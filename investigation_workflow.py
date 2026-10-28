"""
investigation_workflow.py
BurstPulse Analyst Investigation, Grouping, & Automated Demonstration Engine.

Provides:
- Data models for analyst classifications (likely cause, assessment, confidence).
- Named multi-interval Investigation Group models with nonconsecutive interval support.
- Review and group persistence with strict dataset isolation and demo/genuine segregation.
- Group-level analytical statistics and transparent Markdown report generators.
- Automated demonstration generator operating deterministically on actual detector results.
- Graph marker styling and classification mapping.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd


class LikelyCause(str, Enum):
    LEGITIMATE_SURGE = "Legitimate traffic surge / flash crowd"
    SCHEDULED_BACKUP = "Scheduled backup or bulk transfer"
    SOFTWARE_DEPLOYMENT = "Software update or deployment"
    SCHEDULED_JOB = "Expected scheduled job"
    MISCONFIGURATION = "Misconfiguration or malfunction"
    POSSIBLE_DDOS = "Possible DDoS activity"
    POSSIBLE_SCANNING = "Possible scanning or probing"
    SUSPICIOUS_PATTERN = "Suspicious traffic pattern"
    INSUFFICIENT_EVIDENCE = "Other / insufficient evidence"


class AnalystAssessment(str, Enum):
    UNREVIEWED = "Unreviewed"
    LIKELY_BENIGN = "Reviewed — likely benign"
    SUSPECTED_ATTACK = "Suspected attack"
    CONFIRMED_ATTACK = "Confirmed attack — evidence required"


# Marker configuration mapping for Plotly
ASSESSMENT_MARKERS: Dict[str, Dict[str, Any]] = {
    AnalystAssessment.UNREVIEWED.value: {
        "color": "#ef4444",      # Red
        "symbol": "x",
        "name": "Detected Burst (Unreviewed)",
        "line_color": "#991b1b",
        "size": 11,
        "width": 2.5
    },
    AnalystAssessment.LIKELY_BENIGN.value: {
        "color": "#3b82f6",      # Blue
        "symbol": "cross",
        "name": "Reviewed — Likely Benign",
        "line_color": "#1d4ed8",
        "size": 11,
        "width": 2.5
    },
    AnalystAssessment.SUSPECTED_ATTACK.value: {
        "color": "#f59e0b",      # Amber/Yellow
        "symbol": "triangle-up",
        "name": "Suspected Attack",
        "line_color": "#b45309",
        "size": 12,
        "width": 2.5
    },
    AnalystAssessment.CONFIRMED_ATTACK.value: {
        "color": "#a855f7",      # Purple
        "symbol": "square",
        "name": "Confirmed Attack",
        "line_color": "#7e22ce",
        "size": 11,
        "width": 2.5
    }
}


@dataclass
class AnalystReview:
    """Represents an analyst review for a single observation interval."""
    dataset_key: str
    interval: int
    observed_count: int
    measurement_unit: str
    lambda_hat: float
    p_value: float
    alpha: float
    k_crit: int
    likely_cause: str = LikelyCause.INSUFFICIENT_EVIDENCE.value
    assessment: str = AnalystAssessment.UNREVIEWED.value
    notes: str = ""
    evidence: str = ""
    confidence: float = 0.6  # 0.3 Low, 0.6 Medium, 0.9 High
    is_demo: bool = False
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    ground_truth_label: Optional[str] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalystReview":
        return cls(**data)


@dataclass
class InvestigationGroup:
    """Represents a named multi-burst investigation grouping related temporal traffic anomalies."""
    group_id: str
    dataset_key: str
    name: str
    likely_cause: str = LikelyCause.INSUFFICIENT_EVIDENCE.value
    assessment: str = AnalystAssessment.UNREVIEWED.value
    description: str = ""
    notes: str = ""
    supporting_evidence: str = ""
    confidence: float = 0.8
    member_intervals: List[int] = field(default_factory=list)
    is_demo: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationGroup":
        return cls(**data)


def get_investigative_recommendations(likely_cause: str) -> List[str]:
    """
    Return transparent, non-diagnostic next investigative steps based on likely cause.
    These recommendations are investigative guidance, not automatic diagnoses or proof.
    """
    if likely_cause == LikelyCause.LEGITIMATE_SURGE.value:
        return [
            "Verify expected usage patterns against promotional event schedules or customer activity logs.",
            "Cross-reference upstream web analytics and content distribution network (CDN) cache-hit metrics.",
            "Confirm service health and ensure load balancers successfully distributed surge volume."
        ]
    elif likely_cause == LikelyCause.SCHEDULED_BACKUP.value:
        return [
            "Verify scheduled enterprise backup windows and database replication cron jobs.",
            "Confirm source IP address matches designated, authorized backup/storage controllers.",
            "Inspect storage transfer completion logs and integrity checksums."
        ]
    elif likely_cause == LikelyCause.SOFTWARE_DEPLOYMENT.value:
        return [
            "Inspect deployment pipeline records and CI/CD artifact distribution timestamps.",
            "Verify container registry or package repository pull activity for corresponding version releases.",
            "Monitor post-release service telemetry and error rates to distinguish valid rollout from retry storms."
        ]
    elif likely_cause == LikelyCause.SCHEDULED_JOB.value:
        return [
            "Check server cron tables and enterprise batch schedulers for tasks aligned with this interval.",
            "Verify job execution runtimes, CPU spikes, and expected output file transfers.",
            "Ensure batch transfer throttles are configured to avoid saturating interface capacity."
        ]
    elif likely_cause == LikelyCause.MISCONFIGURATION.value:
        return [
            "Inspect recent network infrastructure and application configuration change tickets.",
            "Review application logs for unhandled retry loops, DNS re-resolution storms, or broken connection pools.",
            "Check interface error counters, packet drop rates, and buffer overruns."
        ]
    elif likely_cause == LikelyCause.POSSIBLE_DDOS.value:
        return [
            "Investigate source IP address diversity and geographic entropy across arrival flows.",
            "Analyze protocol composition (TCP SYN, UDP amplification, ICMP, HTTP request rates).",
            "Review firewall / IDS / scrubbing center alerts and evaluate mitigation rule activation."
        ]
    elif likely_cause == LikelyCause.POSSIBLE_SCANNING.value:
        return [
            "Inspect destination port and IP distributions to identify horizontal or vertical scan patterns.",
            "Check TCP connection state ratios (e.g. elevated SYN-ACK without final ACK completion).",
            "Correlate source IP addresses with threat intelligence feeds and honeypot interaction logs."
        ]
    elif likely_cause == LikelyCause.SUSPICIOUS_PATTERN.value:
        return [
            "Collect packet captures (PCAP) around the anomaly window for deep protocol payload inspection.",
            "Correlate with endpoint detection (EDR) telemetry and authentication event logs.",
            "Escalate findings to the Security Operations Center (SOC) according to standard triage procedures."
        ]
    else:
        return [
            "Cause remains undetermined; gather additional network telemetry and endpoint context.",
            "Extend baseline observation window to determine if this count reflects recurring periodic variance.",
            "Consult network operations and application owners before taking remediation action."
        ]


# -----------------------------------------------------------------------------
# Review State Management
# -----------------------------------------------------------------------------

def ensure_reviews_state(session_state: Any) -> Dict[str, Dict[int, AnalystReview]]:
    """Initialize or retrieve the multi-dataset review storage in session_state."""
    if "analyst_reviews_db" not in session_state:
        session_state["analyst_reviews_db"] = {}
    return session_state["analyst_reviews_db"]


def get_dataset_reviews(session_state: Any, dataset_key: str) -> Dict[int, AnalystReview]:
    """Retrieve all reviews for a specific dataset key, guaranteeing strict isolation."""
    db = ensure_reviews_state(session_state)
    if dataset_key not in db:
        db[dataset_key] = {}
    return db[dataset_key]


def save_analyst_review(session_state: Any, review: AnalystReview) -> None:
    """Save or update an analyst review record for a specific dataset and interval."""
    db = ensure_reviews_state(session_state)
    if review.dataset_key not in db:
        db[review.dataset_key] = {}
    db[review.dataset_key][review.interval] = review


def reset_demo_reviews(session_state: Any, dataset_key: str) -> int:
    """
    Remove ONLY demo-generated review records for the dataset.
    Preserves all genuine analyst reviews untouched.
    Returns the count of removed demo records.
    """
    db = ensure_reviews_state(session_state)
    if dataset_key not in db:
        return 0
    demo_keys = [interval for interval, rev in db[dataset_key].items() if rev.is_demo]
    for k in demo_keys:
        del db[dataset_key][k]
    return len(demo_keys)


# -----------------------------------------------------------------------------
# Investigation Group State Management
# -----------------------------------------------------------------------------

def ensure_groups_state(session_state: Any) -> Dict[str, Dict[str, InvestigationGroup]]:
    """Initialize or retrieve the multi-dataset investigation groups storage in session_state."""
    if "investigation_groups_db" not in session_state:
        session_state["investigation_groups_db"] = {}
    return session_state["investigation_groups_db"]


def get_dataset_groups(session_state: Any, dataset_key: str) -> Dict[str, InvestigationGroup]:
    """Retrieve all investigation groups for a specific dataset key, guaranteeing strict isolation."""
    db = ensure_groups_state(session_state)
    if dataset_key not in db:
        db[dataset_key] = {}
    return db[dataset_key]


def save_investigation_group(
    session_state: Any,
    group: InvestigationGroup,
    eval_df: Optional[pd.DataFrame] = None,
    lambda_hat: float = 0.0,
    alpha: float = 0.001,
    k_crit: int = 0,
    unit: str = "packets"
) -> None:
    """
    Save or update an investigation group and synchronize member interval reviews.
    
    Guarantees:
    - Nonconsecutive intervals are stored as an exact list.
    - Prevents conflicting group memberships: removes member intervals from any other group in this dataset.
    - Updates individual AnalystReview records for each member interval to match group classification.
    """
    groups_db = ensure_groups_state(session_state)
    if group.dataset_key not in groups_db:
        groups_db[group.dataset_key] = {}

    # Disallow silent double-membership: remove members from other groups in this dataset
    for other_gid, other_grp in list(groups_db[group.dataset_key].items()):
        if other_gid != group.group_id:
            other_grp.member_intervals = [i for i in other_grp.member_intervals if i not in group.member_intervals]

    group.member_intervals = sorted(list(set(group.member_intervals)))
    group.updated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    groups_db[group.dataset_key][group.group_id] = group

    # Synchronize individual AnalystReview objects for members
    reviews_db = ensure_reviews_state(session_state)
    if group.dataset_key not in reviews_db:
        reviews_db[group.dataset_key] = {}

    for interval in group.member_intervals:
        existing = reviews_db[group.dataset_key].get(interval)
        if existing:
            existing.assessment = group.assessment
            existing.likely_cause = group.likely_cause
            if group.notes:
                existing.notes = group.notes
            if group.supporting_evidence:
                existing.evidence = group.supporting_evidence
            existing.confidence = group.confidence
            existing.group_id = group.group_id
            existing.group_name = group.name
            existing.is_demo = group.is_demo
            existing.updated_at = group.updated_at
        elif eval_df is not None:
            match = eval_df[eval_df["interval"] == interval]
            if not match.empty:
                r_row = match.iloc[0]
                gt = str(r_row.get("attack_type", r_row.get("scenario_label", ""))) or None
                rev = AnalystReview(
                    dataset_key=group.dataset_key,
                    interval=interval,
                    observed_count=int(r_row["packet_count"]),
                    measurement_unit=unit,
                    lambda_hat=lambda_hat,
                    p_value=float(r_row["upper_tail_probability"]),
                    alpha=alpha,
                    k_crit=k_crit,
                    likely_cause=group.likely_cause,
                    assessment=group.assessment,
                    notes=group.notes,
                    evidence=group.supporting_evidence,
                    confidence=group.confidence,
                    is_demo=group.is_demo,
                    ground_truth_label=gt,
                    group_id=group.group_id,
                    group_name=group.name
                )
                reviews_db[group.dataset_key][interval] = rev


def delete_investigation_group(session_state: Any, dataset_key: str, group_id: str) -> bool:
    """Delete an investigation group and unbind its member intervals."""
    groups_db = ensure_groups_state(session_state)
    if dataset_key in groups_db and group_id in groups_db[dataset_key]:
        grp = groups_db[dataset_key][group_id]
        reviews = get_dataset_reviews(session_state, dataset_key)
        for interval in grp.member_intervals:
            if interval in reviews:
                reviews[interval].group_id = None
                reviews[interval].group_name = None
        del groups_db[dataset_key][group_id]
        return True
    return False


def reset_demo_groups(session_state: Any, dataset_key: str) -> int:
    """Delete only demo-generated investigation groups, preserving genuine analyst groups."""
    groups_db = ensure_groups_state(session_state)
    if dataset_key not in groups_db:
        return 0
    demo_gids = [gid for gid, grp in groups_db[dataset_key].items() if grp.is_demo]
    for gid in demo_gids:
        delete_investigation_group(session_state, dataset_key, gid)
    return len(demo_gids)


# -----------------------------------------------------------------------------
# Group-Level Analytics
# -----------------------------------------------------------------------------

def calculate_group_analytics(
    group: InvestigationGroup,
    eval_df: pd.DataFrame,
    lambda_hat: float,
    alpha: float,
    measurement_unit: str = "packets"
) -> Dict[str, Any]:
    """
    Calculate comprehensive group-level statistics derived from actual member intervals.
    
    Adheres strictly to statistical integrity:
    - Never multiplies individual p-values to invent a pseudo-combined p-value.
    - Clearly distinguishes the overall interval span (t_max - t_min + 1) from discrete member bursts.
    - Accurately tracks member count deviations, p-value range, and significance.
    """
    members = sorted(list(set(group.member_intervals)))
    if not members:
        return {
            "num_members": 0,
            "unique_intervals": [],
            "earliest_interval": None,
            "latest_interval": None,
            "interval_span": 0,
            "is_continuous_burst": False,
            "earliest_timestamp": None,
            "latest_timestamp": None,
            "observed_counts": [],
            "min_count": 0,
            "max_count": 0,
            "mean_count": 0.0,
            "median_count": 0.0,
            "lambda_hat": lambda_hat,
            "mean_deviation": 0.0,
            "max_deviation": 0.0,
            "relative_ratio": 1.0,
            "min_p_value": 1.0,
            "max_p_value": 1.0,
            "p_value_range_str": "N/A",
            "alpha": alpha,
            "num_below_alpha": 0,
            "prop_below_alpha": 0.0,
            "member_df": pd.DataFrame(),
            "ground_truth_summary": {},
            "span_note": "No intervals currently assigned to this group."
        }

    # Extract member rows from eval_df
    m_df = eval_df[eval_df["interval"].isin(members)].copy()
    m_df = m_df.sort_values("interval")

    earliest_int = int(members[0])
    latest_int = int(members[-1])
    span = latest_int - earliest_int + 1
    is_continuous = (len(members) == span)

    counts = m_df["packet_count"].values if not m_df.empty else np.array([])
    p_vals = m_df["upper_tail_probability"].values if not m_df.empty else np.array([])

    min_cnt = int(np.min(counts)) if len(counts) > 0 else 0
    max_cnt = int(np.max(counts)) if len(counts) > 0 else 0
    mean_cnt = float(np.mean(counts)) if len(counts) > 0 else 0.0
    med_cnt = float(np.median(counts)) if len(counts) > 0 else 0.0

    mean_dev = mean_cnt - lambda_hat
    max_dev = max_cnt - lambda_hat
    rel_ratio = (mean_cnt / lambda_hat) if lambda_hat > 0 else 1.0

    min_p = float(np.min(p_vals)) if len(p_vals) > 0 else 1.0
    max_p = float(np.max(p_vals)) if len(p_vals) > 0 else 1.0
    p_range_str = f"[{min_p:.2e}, {max_p:.2e}]" if min_p != max_p else f"{min_p:.2e}"

    num_below = int(np.sum(p_vals < alpha)) if len(p_vals) > 0 else 0
    prop_below = float(num_below / len(members)) if len(members) > 0 else 0.0

    earliest_ts = str(m_df["timestamp"].iloc[0]) if "timestamp" in m_df.columns and not m_df.empty else None
    latest_ts = str(m_df["timestamp"].iloc[-1]) if "timestamp" in m_df.columns and not m_df.empty else None

    # Ground truth summary if available
    gt_summary = {}
    if "attack_type" in m_df.columns:
        gt_summary = dict(m_df["attack_type"].value_counts())
    elif "scenario_label" in m_df.columns:
        gt_summary = dict(m_df["scenario_label"].value_counts())

    span_note = (
        f"Investigation window spans {span:,} consecutive intervals (#{earliest_int:,} to #{latest_int:,}), "
        f"containing {len(members):,} discrete member burst intervals ({len(members)/span:.1%}). "
        + ("Observations within this span represent a continuous burst window." if is_continuous else "Member intervals are nonconsecutive; intermediate intervals are not assumed to be bursts.")
    )

    return {
        "num_members": len(members),
        "unique_intervals": members,
        "earliest_interval": earliest_int,
        "latest_interval": latest_int,
        "interval_span": span,
        "is_continuous_burst": is_continuous,
        "earliest_timestamp": earliest_ts,
        "latest_timestamp": latest_ts,
        "observed_counts": list(counts),
        "min_count": min_cnt,
        "max_count": max_cnt,
        "mean_count": mean_cnt,
        "median_count": med_cnt,
        "lambda_hat": lambda_hat,
        "mean_deviation": mean_dev,
        "max_deviation": max_dev,
        "relative_ratio": rel_ratio,
        "min_p_value": min_p,
        "max_p_value": max_p,
        "p_value_range_str": p_range_str,
        "alpha": alpha,
        "num_below_alpha": num_below,
        "prop_below_alpha": prop_below,
        "member_df": m_df,
        "ground_truth_summary": gt_summary,
        "span_note": span_note
    }


# -----------------------------------------------------------------------------
# Automated Demo Workflow
# -----------------------------------------------------------------------------

def run_automated_demo_workflow(
    session_state: Any,
    dataset_key: str,
    analysis_result: Any
) -> Dict[str, Any]:
    """
    Execute a one-click automated demonstration using ACTUAL detector results.
    
    Integrity Rules:
    - Never fabricates counts, lambdas, p-values, or decisions.
    - References actual flagged intervals from analysis_result.evaluated_data.
    - Groups actual flagged bursts into distinct named investigation groups.
    - Labels every record clearly with is_demo=True and DEMO markers.
    - Overwrites previous demo records without duplicating or touching genuine reviews.
    """
    eval_df = analysis_result.evaluated_data
    flagged = eval_df[eval_df["is_burst"]].copy()

    if flagged.empty:
        return {
            "success": False,
            "message": (
                f"No intervals were flagged as potential bursts at α = {analysis_result.alpha:g} "
                f"(k_crit = {analysis_result.k_crit}). Cannot attach demonstration reviews to nonexistent bursts."
            ),
            "num_demo_records": 0,
            "created_groups": []
        }

    # Reset any prior demo records first to avoid duplication
    reset_demo_reviews(session_state, dataset_key)
    reset_demo_groups(session_state, dataset_key)
    existing_reviews = get_dataset_reviews(session_state, dataset_key)

    # Exclude intervals that already have genuine human reviews
    candidate_positions = [
        pos for pos in range(len(flagged))
        if not (
            int(flagged.iloc[pos]["interval"]) in existing_reviews
            and not existing_reviews[int(flagged.iloc[pos]["interval"])].is_demo
        )
    ]

    if not candidate_positions:
        return {
            "success": False,
            "message": "All flagged intervals already possess authentic human analyst reviews. Demo will not overwrite authentic reviews.",
            "num_demo_records": 0,
            "created_groups": []
        }

    total_candidates = len(candidate_positions)
    created_groups: List[InvestigationGroup] = []

    if total_candidates >= 3:
        # Scenario 1: Group 1 - Benign scheduled surge (first burst + median burst if distinct)
        g1_pos = [candidate_positions[0]]
        if total_candidates > 3 and candidate_positions[1] not in g1_pos:
            g1_pos.append(candidate_positions[1])

        g1_intervals = [int(flagged.iloc[p]["interval"]) for p in g1_pos]
        grp1 = InvestigationGroup(
            group_id=f"demo_grp_{dataset_key}_benign",
            dataset_key=dataset_key,
            name="Scheduled Backup Surge",
            likely_cause=LikelyCause.SCHEDULED_BACKUP.value,
            assessment=AnalystAssessment.LIKELY_BENIGN.value,
            description="DEMO — Recurring incremental VM and database backup snapshot cluster.",
            notes="DEMO — Transfer volume matches authorized enterprise storage gateway window.",
            supporting_evidence="Automated Demo: Correlated with synthetic storage controller backup schedule.",
            confidence=0.90,
            member_intervals=g1_intervals,
            is_demo=True
        )
        save_investigation_group(
            session_state, grp1, eval_df=eval_df,
            lambda_hat=analysis_result.lambda_hat, alpha=analysis_result.alpha,
            k_crit=analysis_result.k_crit, unit=analysis_result.measurement_unit
        )
        created_groups.append(grp1)

        # Scenario 2: Group 2 - Suspected Attack (peak burst + final burst)
        cand_counts = [flagged.iloc[p]["packet_count"] for p in candidate_positions if p not in g1_pos]
        if cand_counts:
            peak_sub_idx = int(np.argmax(cand_counts))
            remaining_cands = [p for p in candidate_positions if p not in g1_pos]
            peak_pos = remaining_cands[peak_sub_idx]
            g2_pos = [peak_pos]
            if len(remaining_cands) > 1 and remaining_cands[-1] != peak_pos:
                g2_pos.append(remaining_cands[-1])

            g2_intervals = [int(flagged.iloc[p]["interval"]) for p in g2_pos]
            grp2 = InvestigationGroup(
                group_id=f"demo_grp_{dataset_key}_attack",
                dataset_key=dataset_key,
                name="Possible DDoS Investigation",
                likely_cause=LikelyCause.POSSIBLE_DDOS.value,
                assessment=AnalystAssessment.SUSPECTED_ATTACK.value,
                description="DEMO — Unusually concentrated arrival surge exceeding 3.5× baseline rate.",
                notes="DEMO — Peak arrival volume surge observed; destination application latency elevated.",
                supporting_evidence="Automated Demo: Corroborated with synthetic high-rate UDP/SYN packet distribution.",
                confidence=0.85,
                member_intervals=g2_intervals,
                is_demo=True
            )
            save_investigation_group(
                session_state, grp2, eval_df=eval_df,
                lambda_hat=analysis_result.lambda_hat, alpha=analysis_result.alpha,
                k_crit=analysis_result.k_crit, unit=analysis_result.measurement_unit
            )
            created_groups.append(grp2)

    else:
        # Scenario for 1 or 2 candidates
        g_intervals = [int(flagged.iloc[p]["interval"]) for p in candidate_positions]
        grp = InvestigationGroup(
            group_id=f"demo_grp_{dataset_key}_anomaly",
            dataset_key=dataset_key,
            name="Suspicious Traffic Anomaly",
            likely_cause=LikelyCause.SUSPICIOUS_PATTERN.value,
            assessment=AnalystAssessment.SUSPECTED_ATTACK.value,
            description="DEMO — Statistically significant volume surge flagged by Poisson upper-tail test.",
            notes="DEMO — High volume surge requiring protocol and source entropy inspection.",
            supporting_evidence="Automated Demo: Real flagged detector outliers populated with demonstration review.",
            confidence=0.80,
            member_intervals=g_intervals,
            is_demo=True
        )
        save_investigation_group(
            session_state, grp, eval_df=eval_df,
            lambda_hat=analysis_result.lambda_hat, alpha=analysis_result.alpha,
            k_crit=analysis_result.k_crit, unit=analysis_result.measurement_unit
        )
        created_groups.append(grp)

    total_demo_members = sum(len(g.member_intervals) for g in created_groups)
    active_gid = created_groups[0].group_id if created_groups else None

    return {
        "success": True,
        "message": f"Successfully initialized {len(created_groups)} demonstration investigation groups covering {total_demo_members} real detected bursts.",
        "num_demo_records": total_demo_members,
        "created_groups": created_groups,
        "active_group_id": active_gid,
        "demo_intervals": [i for g in created_groups for i in g.member_intervals]
    }


# -----------------------------------------------------------------------------
# Report Generators
# -----------------------------------------------------------------------------

def generate_single_burst_report_markdown(
    review: AnalystReview,
    dataset_name: str,
    timestamp_str: Optional[str] = None
) -> str:
    """Generate an authoritative, standalone investigation report in Markdown for a single interval."""
    diff = review.observed_count - review.lambda_hat
    mult = (review.observed_count / review.lambda_hat) if review.lambda_hat > 0 else 1.0
    status_note = (
        "⚠️ **DEMO — SIMULATED ANALYST REVIEW:** Generated programmatically for demonstration purposes."
        if review.is_demo else
        "This report represents an authentic analyst review recorded during live traffic evaluation."
    )
    ground_truth_text = f"Dataset Ground-Truth: `{review.ground_truth_label}`" if review.ground_truth_label else "Dataset Ground-Truth: `Unlabeled / Benchmark`"
    audit_note = (
        "⚠️ DEMO NOTICE: This classification was populated automatically as part of a demonstration."
        if review.is_demo else
        "AUTHENTIC FINDING: Human analyst judgment."
    )
    recs = get_investigative_recommendations(review.likely_cause)
    recs_md = "\n".join([f"- {r}" for r in recs])

    md = f"""# BurstPulse Network Traffic Investigation Report

**Generated At:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Dataset Reference:** {dataset_name}  
**Investigation Target:** Interval #{review.interval} ({timestamp_str or 'Discrete Index'})  
**Analyst Assessment Status:** **{review.assessment}** (Confidence: {int(review.confidence*100)}%)  

---

## 1. Executive Summary
During the evaluated observation window, interval **#{review.interval}** recorded **{review.observed_count:,} {review.measurement_unit}**, representing an absolute deviation of **{diff:+.1f} {review.measurement_unit}** (**{mult:.2f}×** above the stationary baseline arrival rate λ̂ = {review.lambda_hat:.2f} {review.measurement_unit}/interval).

Under the null hypothesis (H₀: stationary Poisson arrival process), the upper-tail survival probability of observing {review.observed_count} or more arrivals purely by chance is:
P(X ≥ {review.observed_count} | λ̂ = {review.lambda_hat:.2f}) = {review.p_value:.3e}

Because p = {review.p_value:.3e} < α = {review.alpha:g} (satisfying critical cutoff k ≥ k_crit = {review.k_crit}), the statistical detector rejected H₀ and flagged this interval as an improbable volume surge.

---

## 2. Statistical & Detector Telemetry
| Metric | Recorded Value | Theoretical / Baseline Context |
| :--- | :--- | :--- |
| **Observation Index** | `#{review.interval}` | Discrete temporal evaluation interval |
| **Observed Count (k)** | **{review.observed_count:,} {review.measurement_unit}** | Actual recorded arrivals in interval |
| **Baseline Rate (λ̂)** | `{review.lambda_hat:.2f}` {review.measurement_unit}/int | Stationary baseline MLE rate |
| **Absolute Deviation** | `{diff:+.2f}` | Observed count minus baseline expectation |
| **Relative Arrival Ratio** | `{mult:.2f}×` | Multiplicative factor above baseline rate |
| **Upper-Tail P(X ≥ k)** | `{review.p_value:.3e}` | Poisson cumulative survival probability |
| **Significance Level (α)** | `{review.alpha:g}` | Configured false-positive probability budget |
| **Critical Cutoff (k_crit)**| `{review.k_crit}` | Minimum arrivals required to reject H₀ |
| **Benchmark Ground Truth** | {ground_truth_text} | Recorded dataset label |

---

## 3. Analyst Review & Triage Classification
- **Audit Classification:** {audit_note}
- **Assessed Likely Cause:** `{review.likely_cause}`
- **Formal Assessment:** **{review.assessment}**
- **Assessment Confidence:** `{int(review.confidence*100)}%`
- **Analyst Notes:**
> {review.notes or 'No additional notes provided.'}
- **Supporting Telemetry / Evidence:**
> {review.evidence or 'No external log traces attached.'}

---

## 4. Recommended Next Investigative Steps
{recs_md}

---

## 5. Methodological Limitations & Cyberattack Distinction
1. **Volume Rarity vs. Malicious Intent:** Statistical volume surge detection establishes mathematical rarity under the calibrated Poisson baseline; it does **not** provide cryptographic or legal verification of malicious intent. Benign automated backups, batch synchronization jobs, and video streaming cause legitimate statistical bursts.
2. **Stealth Attack Blind Spot:** Low-rate, slow-and-low port scans or command-and-control beacons whose arrival volume blends within the baseline rate are not detected by pure volume testing.
3. **Report Status:** {status_note}
"""
    return md


def generate_group_investigation_report_markdown(
    group: InvestigationGroup,
    analytics: Dict[str, Any],
    dataset_name: str,
    provenance: str,
    measurement_unit: str = "packets"
) -> str:
    """Generate a formal, thorough group investigation report in Markdown."""
    recs = get_investigative_recommendations(group.likely_cause)
    recs_md = "\n".join([f"- {r}" for r in recs])

    audit_badge = (
        "⚠️ **DEMO — SIMULATED ANALYST REVIEW:** This investigation group was generated programmatically for demonstration purposes."
        if group.is_demo else
        "🛡️ **GENUINE ANALYST INVESTIGATION:** Authentic human review records recorded during traffic evaluation."
    )

    members_str = ", ".join([f"#{i}" for i in group.member_intervals])
    m_df = analytics.get("member_df", pd.DataFrame())
    rows = []
    if not m_df.empty:
        for _, row in m_df.iterrows():
            i_val = int(row["interval"])
            cnt = int(row["packet_count"])
            pval = float(row["upper_tail_probability"])
            dev = cnt - analytics["lambda_hat"]
            ratio = (cnt / analytics["lambda_hat"]) if analytics["lambda_hat"] > 0 else 1.0
            gt = str(row.get("attack_type", row.get("scenario_label", "N/A")))
            ts = str(row.get("timestamp", "-"))
            rows.append(
                f"| #{i_val:,} | {ts} | {cnt:,} {measurement_unit} | {dev:+.1f} | {ratio:.2f}× | {pval:.2e} | "
                f"{'🚨 Burst' if row.get('is_burst', True) else 'Normal'} | {gt} |"
            )
    member_table_md = "\n".join(rows) if rows else "| None | - | - | - | - | - | - | - |"

    md = f"""# BurstPulse Group Investigation Report: {group.name}

**Report Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Investigation ID:** `{group.group_id}`  
**Dataset Reference:** {dataset_name} ({provenance})  
**Analyst Assessment:** **{group.assessment}** (Confidence: {int(group.confidence * 100)}%)  
**Selected Likely Cause:** {group.likely_cause}  
**Audit Context:** {audit_badge}

---

## 1. Executive Summary & Span Telemetry
{group.description if group.description else 'Named multi-burst investigation grouping related temporal traffic anomalies.'}

- **Member Burst Intervals:** `{analytics['num_members']:,}` discrete intervals ({members_str})
- **Temporal Span:** Interval **#{analytics['earliest_interval']}** to **#{analytics['latest_interval']}** ({analytics['interval_span']:,} intervals total)
- **Span Characteristics:** {analytics['span_note']}
- **Baseline Calibration Expectation:** λ̂ = {analytics['lambda_hat']:.2f} {measurement_unit}/int (Stationary Poisson Null Hypothesis H₀)
- **Observed Count Range:** `{analytics['min_count']:,}` to `{analytics['max_count']:,}` {measurement_unit} (Mean: `{analytics['mean_count']:.2f}`, Median: `{analytics['median_count']:.2f}`)
- **Deviation from Baseline:** Mean deviation of **{analytics['mean_deviation']:+.2f} {measurement_unit}** (**{analytics['relative_ratio']:.2f}×** baseline arrival rate)
- **Statistical Significance:** Upper-tail p-value range `{analytics['p_value_range_str']}` at significance threshold α = {analytics['alpha']:g}
- **Improbability Criterion:** `{analytics['num_below_alpha']}/{analytics['num_members']}` member intervals ({analytics['prop_below_alpha']:.0%}) meet the rejection criterion P(X ≥ k) < α.

---

## 2. Member Telemetry Breakdown
| Interval | Timestamp | Observed Count | Abs Dev | Ratio | P(X ≥ k) | Detector Flag | Dataset Ground-Truth |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{member_table_md}

*Note: Group statistics reflect the exact member intervals above. Statistical p-values are reported per-interval under Poisson assumptions; they are not artificially multiplied across intervals.*

---

## 3. Analyst Findings & Corroborating Evidence
- **Investigation Notes:**  
> {group.notes if group.notes else 'No additional analyst notes recorded.'}

- **Supporting Telemetry / Evidence Logs:**  
> {group.supporting_evidence if group.supporting_evidence else 'No corroborating log traces attached.'}

---

## 4. Recommended Next Investigative Steps
Investigative guidance aligned with **{group.likely_cause}**:
{recs_md}

---

## 5. Methodological Limitations & Cyberattack Distinction
1. **Mathematical Anomaly vs. Adversarial Intent:** A cluster of Poisson volume bursts establishes empirical departure from the calibrated arrival baseline. However, non-malicious operational workloads (e.g., enterprise storage snapshots, CDN origin synchronization, software updates) frequently generate statistically improbable volume spikes.
2. **Span Continuity:** Intervals between nonconsecutive members did not violate threshold α and must not be assumed to be an ongoing attack.
3. **Evidence Requirement:** Formal attribution to an attack vector requires correlating source IP entropy, destination port uniformity, flow duration, and host authentication logs.
"""
    return md


def generate_summary_investigation_report_markdown(
    reviews: List[AnalystReview],
    dataset_name: str,
    analysis_result: Any
) -> str:
    """Generate a comprehensive multi-interval summary report covering all reviews."""
    total_reviews = len(reviews)
    benign_count = sum(1 for r in reviews if r.assessment == AnalystAssessment.LIKELY_BENIGN.value)
    suspected_count = sum(1 for r in reviews if r.assessment == AnalystAssessment.SUSPECTED_ATTACK.value)
    confirmed_count = sum(1 for r in reviews if r.assessment == AnalystAssessment.CONFIRMED_ATTACK.value)
    unreviewed_count = sum(1 for r in reviews if r.assessment == AnalystAssessment.UNREVIEWED.value)
    has_demo = any(r.is_demo for r in reviews)

    audit_note = (
        "⚠️ Report includes simulated demonstration records clearly indicated."
        if has_demo else
        "Report contains authentic analyst evaluation records."
    )

    rows = []
    for r in reviews:
        demo_str = " (Demo)" if r.is_demo else ""
        grp_str = f" [{r.group_name}]" if r.group_name else ""
        rows.append(
            f"| #{r.interval:,} | {r.observed_count:,} {r.measurement_unit} | {r.p_value:.2e} | "
            f"{r.likely_cause} | **{r.assessment}**{demo_str}{grp_str} | {int(r.confidence*100)}% |"
        )
    table_md = "\n".join(rows) if rows else "| None | - | - | - | - | - |"

    k_crit_val = getattr(analysis_result, "k_crit", "N/A")
    md = f"""# BurstPulse Comprehensive Investigation Summary Report

**Generated At:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Dataset Reference:** {dataset_name}  
**Baseline Parameters:** λ̂ = {analysis_result.lambda_hat:.2f} {analysis_result.measurement_unit}/int &nbsp;·&nbsp; α = {analysis_result.alpha:g} &nbsp;·&nbsp; k_crit = {k_crit_val}  
**Audit Note:** {audit_note}

---

## 1. Review Triage Overview
- **Total Reviewed Intervals:** `{total_reviews}`
- **Reviewed — Likely Benign:** `{benign_count}` ({benign_count/max(1, total_reviews):.1%})
- **Suspected Attacks:** `{suspected_count}` ({suspected_count/max(1, total_reviews):.1%})
- **Confirmed Attacks (Evidence-Supported):** `{confirmed_count}` ({confirmed_count/max(1, total_reviews):.1%})
- **Pending / Unreviewed:** `{unreviewed_count}`

---

## 2. Interval Review Summary Table
| Interval | Observed Count | P-Value | Likely Cause | Assessment | Confidence |
| :--- | :--- | :--- | :--- | :--- | :--- |
{table_md}

---

## 3. Methodological Advisory
Statistical Poisson burst detection identifies mathematical deviations from stationary arrival expectations. True threat qualification requires correlating volume anomalies with flow duration, endpoint authentication logs, and network protocol context.
"""
    return md
