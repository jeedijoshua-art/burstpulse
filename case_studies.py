"""
case_studies.py
Historical Incident Case Studies and Threat Analysis.
Applied Mathematics Micro Project — Real-World Traffic & Historical Incidents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CaseStudy:
    """Structured analytical case study for a documented network incident or attack scenario."""
    id: str
    title: str
    incident_date: str
    source_org: str
    official_reference: str
    official_url: str
    threat_category: str
    data_availability: str
    what_source_establishes: str
    available_traffic_data: str
    poisson_model_applicability: str
    model_limitations_and_caveats: str
    key_takeaways: List[str] = field(default_factory=list)


CASE_STUDIES: Dict[str, CaseStudy] = {
    "caida_2007": CaseStudy(
        id="caida_2007",
        title="CAIDA 2007: Coordinated Infrastructure DDoS Attack",
        incident_date="August 4, 2007 (20:50:08 – 21:56:16 UTC)",
        source_org="Center for Applied Internet Data Analysis (CAIDA), UC San Diego",
        official_reference="The CAIDA UCSD 'DDoS Attack 2007' Dataset",
        official_url="https://catalog.caida.org/dataset/ddos_attack_2007",
        threat_category="Volumetric Distributed Denial-of-Service (SYN & ICMP Floods)",
        data_availability="Restricted by CAIDA Acceptable Use Agreement (AUA) — Research / Non-profit only",
        what_source_establishes=(
            "Captures approximately one hour of authentic, coordinated flood traffic directed at a single victim server. "
            "Confirms that distributed botnets can overwhelm ingress line rates with massive bursts of spoofed SYN and ICMP packets."
        ),
        available_traffic_data=(
            "Approximately 21 GB uncompressed PCAP files split into 5-minute segments. "
            "IP addresses anonymized via CryptoPAn prefix-preserving mapping; payload data completely removed. "
            "Non-attack background traffic was largely filtered out prior to release."
        ),
        poisson_model_applicability=(
            "When binned into 1-second intervals via `tshark -r ddos.pcap -q -z io,stat,1`, the Poisson upper-tail detector "
            "can identify onset transitions and intra-attack intensity shifts where packet arrival counts spike by orders of magnitude."
        ),
        model_limitations_and_caveats=(
            "1. Because non-attack background traffic was filtered out during dataset creation, normal baseline arrival rate "
            "estimation (λ̂) cannot be obtained from the trace itself without prior baseline traffic.\n"
            "2. Poisson models assume independent arrivals, whereas coordinated botnet floods exhibit synchronized attack pulses."
        ),
        key_takeaways=[
            "Authentic historical benchmark for volumetric DDoS research",
            "Demonstrates rapid surge from quiescent state to maximum line saturation",
            "Illustrates the necessity of separate baseline calibration periods before flood onset",
            "Access strictly governed by CAIDA AUA to prevent misuse"
        ]
    ),

    "mirai_dyn_2016": CaseStudy(
        id="mirai_dyn_2016",
        title="Mirai Botnet: Dyn Managed DNS Outage (Critical Infrastructure Disruption)",
        incident_date="Friday, October 21, 2016 (Morning & Afternoon EDT)",
        source_org="Dyn (now Oracle Cloud Infrastructure) / CISA / US-CERT",
        official_reference="CISA Alert TA16-288A: Heightened DDoS Threat Posed by Mirai and Other Botnets",
        official_url="https://www.cisa.gov/news-events/alerts/2016/10/21/heightened-ddos-threat-posed-mirai-and-other-botnets",
        threat_category="IoT Botnet Volumetric & Application-Layer DNS Flood",
        data_availability="Contextual Case Study (No Public Packet Trace Released by Dyn)",
        what_source_establishes=(
            "Demonstrated that compromised consumer IoT devices (IP cameras, DVRs) with default credentials "
            "can be weaponized into an unprecedented volumetric botnet exceeding 1 Tbps, causing widespread collateral "
            "failure across recursive DNS resolvers and cloud services (Twitter, GitHub, Spotify, Netflix)."
        ),
        available_traffic_data=(
            "Qualitative post-mortems, CISA security advisories, and bandwidth telemetry graphs. "
            "Dyn never released raw packet PCAPs publicly due to confidential customer queries and privacy obligations. "
            "Strict academic integrity dictates treating this strictly as a contextual case study rather than a simulated packet dataset."
        ),
        poisson_model_applicability=(
            "Provides vital theoretical understanding of DNS queue dynamics. In recursive resolvers, query arrivals normally "
            "approximate Poisson behavior during off-peak hours, but catastrophic multi-vector surges cause immediate buffer overflow."
        ),
        model_limitations_and_caveats=(
            "1. No raw packet PCAP is publicly available; attempting to run automated packet detection on this event would require "
            "fabricating packet numbers, which violates academic honesty.\n"
            "2. Mirai combined volumetric floods (UDP/SYN) with valid recursive DNS queries ('water torture'), illustrating that "
            "packet volume alone cannot distinguish valid resolver retries from malicious query generation."
        ),
        key_takeaways=[
            "Watershed event in Internet history: IoT botnets weaponized for multi-hundred Gbps floods",
            "Illustrates cascading failure in DNS hierarchy when authoritative servers become unreachable",
            "Emphasizes the difference between volume anomaly detection and semantic query validation",
            "Highlighted in this project to demonstrate the real-world operational importance of burst monitoring"
        ]
    ),

    "cic_ids_2017_portscan": CaseStudy(
        id="cic_ids_2017_portscan",
        title="CIC-IDS2017: Friday Automated Reconnaissance & PortScan",
        incident_date="Friday, July 7, 2017 (12:30 – 15:40 ADT)",
        source_org="Canadian Institute for Cybersecurity (CIC), University of New Brunswick",
        official_reference="Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization (ICISSP 2018)",
        official_url="https://www.unb.ca/cic/datasets/ids-2017.html",
        threat_category="Reconnaissance / Port Scanning (Nmap probe sweeps)",
        data_availability="Available for Local Analysis (Bundled Curated 200-Interval Slice)",
        what_source_establishes=(
            "Documents an empirical transition from a clean benign morning baseline (B-Profile human-generated traffic) "
            "into high-rate automated network scanning using Nmap. Verified by ground-truth packet and flow labels."
        ),
        available_traffic_data=(
            "Full Friday PCAP (~8.3 GB) and CICFlowMeter feature CSVs. A curated 200-interval sequence (10-second bins) "
            "is provided in `data/cic_ids2017_portscan_sample.csv` with verified attack tags (`PortScan` vs `BENIGN`)."
        ),
        poisson_model_applicability=(
            "Directly testable in this application. The baseline is calibrated on the clean morning partition (intervals 1 to 120), "
            "and the Poisson upper-tail detector correctly flags the port scanning probe surges during the afternoon evaluation window."
        ),
        model_limitations_and_caveats=(
            "1. The Poisson model flags volume spikes as statistical bursts; it does not identify that the packets were SYN scans.\n"
            "2. Low-rate / slow reconnaissance (e.g. 1 probe per minute) would blend into the baseline rate and evade Poisson volume detection."
        ),
        key_takeaways=[
            "Demonstrates the full end-to-end pipeline from raw flow records to statistical burst classification",
            "Enables empirical calculation of precision, recall, and false alarms against verified ground-truth labels",
            "Cleanly illustrates that high packet volume is a strong indicator of automated port sweeps",
            "Fully reproducible inside this project's interactive analyzer"
        ]
    )
}


def get_case_study(study_id: str) -> Optional[CaseStudy]:
    """Retrieve a case study by ID."""
    return CASE_STUDIES.get(study_id)


def list_case_studies() -> List[CaseStudy]:
    """Return all available historical case studies."""
    return list(CASE_STUDIES.values())
