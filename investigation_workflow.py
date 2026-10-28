"""
investigation_workflow.py
BurstPulse Analyst Investigation & Automated Demonstration Engine.

Provides:
- Data models for analyst classifications (likely cause, assessment, confidence).
- Review persistence with strict dataset isolation and demo/genuine segregation.
- Automated demonstration generator operating deterministically on actual detector results.
- Transparent investigative recommendations and automated markdown report generators.
- Graph marker styling and classification mapping.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Dict, Any, List, Optional, Tuple
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
    """Represents a human or demo analyst review for a single observation interval."""
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalystReview":
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
    - Deterministically selects up to 4 representative flagged intervals.
    - Applies distinct realistic scenarios (benign surge, suspected scan/DDoS, misconfiguration).
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
            "num_demo_records": 0
        }

    # Reset any prior demo records first to avoid duplication
    reset_demo_reviews(session_state, dataset_key)
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
            "num_demo_records": 0
        }

    total_candidates = len(candidate_positions)
    if total_candidates == 1:
        chosen_positions = [candidate_positions[0]]
    elif total_candidates == 2:
        chosen_positions = [candidate_positions[0], candidate_positions[1]]
    elif total_candidates == 3:
        chosen_positions = [candidate_positions[0], candidate_positions[1], candidate_positions[2]]
    else:
        # First available, peak burst, median, and final burst among candidates
        cand_counts = [flagged.iloc[p]["packet_count"] for p in candidate_positions]
        peak_idx = int(np.argmax(cand_counts))
        chosen_positions = sorted(list({
            candidate_positions[0],
            candidate_positions[peak_idx],
            candidate_positions[total_candidates // 2],
            candidate_positions[-1]
        }))

    scenarios = [
        {
            "cause": LikelyCause.SCHEDULED_BACKUP.value,
            "assessment": AnalystAssessment.LIKELY_BENIGN.value,
            "confidence": 0.9,
            "notes": "DEMO — Scheduled incremental VM snapshot. Traffic matches authorized storage controller transfer window."
        },
        {
            "cause": LikelyCause.POSSIBLE_DDOS.value,
            "assessment": AnalystAssessment.SUSPECTED_ATTACK.value,
            "confidence": 0.8,
            "notes": "DEMO — Unusually concentrated arrival surge exceeding 3.5× baseline rate. Destination service latency elevated."
        },
        {
            "cause": LikelyCause.POSSIBLE_SCANNING.value,
            "assessment": AnalystAssessment.CONFIRMED_ATTACK.value,
            "confidence": 0.95,
            "notes": "DEMO — Corroborated with multi-port sweep alerts. Rapid SYN packet generation across sequential ports."
        },
        {
            "cause": LikelyCause.MISCONFIGURATION.value,
            "assessment": AnalystAssessment.LIKELY_BENIGN.value,
            "confidence": 0.7,
            "notes": "DEMO — Application client retry storm caused by transient gateway timeout; confirmed non-malicious."
        }
    ]

    created_reviews: List[AnalystReview] = []

    for idx, row_pos in enumerate(chosen_positions):
        row = flagged.iloc[row_pos]
        scenario = scenarios[idx % len(scenarios)]
        interval_num = int(row["interval"])
        cnt = int(row["packet_count"])
        p_val = float(row["upper_tail_probability"])
        gt_label = str(row.get("attack_type", row.get("scenario_label", ""))) or None

        rev = AnalystReview(
            dataset_key=dataset_key,
            interval=interval_num,
            observed_count=cnt,
            measurement_unit=analysis_result.measurement_unit,
            lambda_hat=analysis_result.lambda_hat,
            p_value=p_val,
            alpha=analysis_result.alpha,
            k_crit=analysis_result.k_crit,
            likely_cause=scenario["cause"],
            assessment=scenario["assessment"],
            notes=scenario["notes"],
            evidence="Automated Demo: Synthetically populated investigative scenario mapped to actual detector outlier.",
            confidence=scenario["confidence"],
            is_demo=True,
            ground_truth_label=gt_label
        )
        save_analyst_review(session_state, rev)
        created_reviews.append(rev)

    return {
        "success": True,
        "message": f"Successfully initialized {len(created_reviews)} demonstration review records mapped to real detected bursts.",
        "num_demo_records": len(created_reviews),
        "demo_intervals": [r.interval for r in created_reviews]
    }


def generate_single_burst_report_markdown(
    review: AnalystReview,
    dataset_name: str,
    timestamp_str: Optional[str] = None
) -> str:
    """
    Generate an authoritative, standalone investigation report in Markdown
    for a single observation interval.
    """
    diff = review.observed_count - review.lambda_hat
    mult = (review.observed_count / review.lambda_hat) if review.lambda_hat > 0 else 1.0
    demo_badge = " [DEMO — SIMULATED ANALYST REVIEW]" if review.is_demo else ""
    recommendations = get_investigative_recommendations(review.likely_cause)
    recs_md = "\n".join([f"- {r}" for r in recommendations])

    status_note = (
        "This report represents a simulated demonstration record generated deterministically for evaluation purposes."
        if review.is_demo else
        "This report represents an authentic analyst review recorded during live traffic evaluation."
    )
    notes_text = review.notes.strip() if review.notes and review.notes.strip() else "No supplementary analyst notes entered."
    evidence_text = review.evidence.strip() if review.evidence and review.evidence.strip() else "No supplementary evidence logs attached."
    ground_truth_text = review.ground_truth_label or "None / Unlabeled"
    ts_label = f"({timestamp_str})" if timestamp_str else ""

    md = f"""# BurstPulse Network Traffic Investigation Report{demo_badge}

**Generated At:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Dataset Reference:** {dataset_name}  
**Investigation Target:** Interval #{review.interval:,} {ts_label}  
**Analyst Assessment Status:** **{review.assessment}** (Confidence: {int(review.confidence * 100)}%)  

---

## 1. Executive Summary
During the evaluated observation window, interval **#{review.interval:,}** recorded **{review.observed_count:,} {review.measurement_unit}**, representing an absolute deviation of **{diff:+.1f} {review.measurement_unit}** (**{mult:.2f}×** above the stationary baseline arrival rate λ̂ = {review.lambda_hat:.2f} {review.measurement_unit}/interval).

Under the null hypothesis (H₀: stationary Poisson arrival process), the upper-tail survival probability of observing {review.observed_count:,} or more arrivals purely by chance is:
P(X ≥ {review.observed_count:,} | λ̂ = {review.lambda_hat:.2f}) = {review.p_value:.3e}

Because p = {review.p_value:.3e} < α = {review.alpha:g} (satisfying critical cutoff k ≥ k_crit = {review.k_crit}), the statistical detector rejected H₀ and flagged this interval as an improbable volume surge.

---

## 2. Statistical & Detector Telemetry
| Metric | Recorded Value | Theoretical / Baseline Context |
| :--- | :--- | :--- |
| **Observation Index** | `#{review.interval:,}` | Discrete temporal evaluation interval |
| **Observed Count (k)** | **{review.observed_count:,} {review.measurement_unit}** | Evaluated arrival volume |
| **Baseline Arrival Rate (λ̂)** | `{review.lambda_hat:.4f} {review.measurement_unit}/int` | Calibrated on uncontaminated baseline partition |
| **Critical Cutoff (k_crit)** | `{review.k_crit} {review.measurement_unit}` | Integer threshold where P(X ≥ k) < α |
| **Upper-Tail Probability (p)** | `{review.p_value:.4e}` | Cumulative Poisson survival: `poisson.sf(k-1, λ̂)` |
| **Decision Cutoff (α)** | `{review.alpha:g}` | Configured statistical significance level |
| **Statistical Decision** | `REJECT H₀ (Flagged Burst)` | Improbable under stationary baseline rate |
| **Ground-Truth Marker** | `{ground_truth_text}` | Labeled scenario marker (kept separate from analyst review) |

---

## 3. Analyst Investigation & Findings
- **Assigned Likely Cause:** `{review.likely_cause}`
- **Analyst Assessment:** **`{review.assessment}`**
- **Assessment Confidence:** `{int(review.confidence * 100)}%`
- **Analyst Notes:**
> {notes_text}
- **Supporting Telemetry / Evidence:**
> {evidence_text}

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
        rows.append(
            f"| #{r.interval:,} | {r.observed_count:,} {r.measurement_unit} | {r.p_value:.2e} | "
            f"{r.likely_cause} | **{r.assessment}**{demo_str} | {int(r.confidence*100)}% |"
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
