"""
dataset_registry.py
Dataset Registry and Metadata Catalog for Network Traffic Burst Analysis.
Applied Mathematics Micro Project — 5 Research-Inspired Synthetic Benchmark Datasets.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Dict, List, Optional
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


class ProvenanceCategory(str, Enum):
    """Explicit provenance classification for network traffic datasets."""
    SYNTHETIC_SIMULATION = "Synthetic Simulation"
    VERIFIED_REAL_CAPTURE = "Verified Real Capture"
    CONTROLLED_RESEARCH = "Controlled Research Benchmark"
    HISTORICAL_INCIDENT = "Historical Incident Study"
    RESTRICTED_EXTERNAL = "Restricted External Dataset"


# Backward-compatibility alias
DatasetCategory = ProvenanceCategory


class AvailabilityStatus(str, Enum):
    """Local availability and accessibility status for a dataset."""
    LOCAL_READY = "Available for Local Analysis"
    EXTERNAL_DOWNLOAD = "Public External Download"
    RESTRICTED_ACCESS = "Restricted Access (AUA / Registration Required)"
    CONTEXTUAL_ONLY = "Contextual Case Study (No Public Packet Trace)"


@dataclass
class DatasetEntry:
    """Detailed metadata record for a network traffic dataset."""
    id: str
    name: str
    category: ProvenanceCategory
    local_file_path: str
    num_observations: int
    description: str
    expected_findings: str
    limitations: List[str]
    has_labels: bool = False
    label_names: List[str] = field(default_factory=list)
    source_org: str = "BurstPulse Research Benchmark Suite"
    official_url: str = "http://localhost:8501"
    publication_date: str = "2026"
    trace_date: str = "Reproducible Synthetic Simulation"
    documented_event: str = ""
    traffic_nature: str = ""
    file_format: str = "CSV (interval, timestamp, packet_count, flow_count, ...)"
    approx_size: str = ""
    license_and_access: str = "Open Academic Project (MIT License)"
    availability: AvailabilityStatus = AvailabilityStatus.LOCAL_READY
    verified_original_identifier: str = ""
    original_format: str = "Synthetic Research-Inspired CSV"
    actual_time_resolution: str = "1-second discrete intervals"
    measurement_unit: str = "packets_and_flows_per_1s_interval"
    interval_seconds: int = 1
    clean_baseline_intervals: int = 3000
    clean_baseline_pct: int = 30
    is_synthetic: bool = True
    semantics: str = "Interval-level packet and flow counts"
    preprocessing_method: str = "Pre-aggregated fixed-duration intervals."
    citation: str = "Applied Mathematics Micro Project: Network Traffic Burst Analysis (2026)."
    academic_notes: str = ""
    key_features: List[str] = field(default_factory=list)
    download_instructions: Optional[str] = None
    provenance_statements: List[str] = field(default_factory=list)

    @property
    def provenance_category(self) -> ProvenanceCategory:
        return self.category

    @property
    def resolved_path(self) -> Path:
        """Resolve local file path robustly relative to project root."""
        p = Path(self.local_file_path)
        if not p.is_absolute():
            return PROJECT_ROOT / p
        return p

    @property
    def is_analyzable(self) -> bool:
        if not self.local_file_path:
            return False
        p = self.resolved_path
        return p.is_file() and p.stat().st_size > 0

    def is_available(self) -> bool:
        """Backward-compatibility check for dataset file availability."""
        return self.is_analyzable


# Load provenance statements if available
PROVENANCE_STATEMENTS: List[str] = [
    "All five datasets were generated programmatically with fixed random seeds.",
    "They are NOT genuine downloaded research datasets.",
    "Folder names indicate research-inspired test profiles only.",
    "No claim is made that any record originated from the named institutions, testbeds or datasets (CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15, BoT-IoT, TON_IoT).",
    "The data is intended for software development, demos, statistical testing and educational use.",
    "Genuine research-data validation requires obtaining and processing the original datasets.",
    "Attack labels describe simulated scenarios; a statistical burst detection is not proof of an attack."
]

try:
    prov_file = PROJECT_ROOT / "BurstPulse_Real_Datasets" / "DATA_PROVENANCE.json"
    if prov_file.exists():
        prov_json = json.loads(prov_file.read_text(encoding="utf-8"))
        if "statements" in prov_json:
            PROVENANCE_STATEMENTS = prov_json["statements"]
except Exception:
    pass


# EXACTLY FIVE BUILT-IN RESEARCH-INSPIRED DATASETS
BUILTIN_DATASETS: Dict[str, DatasetEntry] = {
    "cic_ids2017_style": DatasetEntry(
        id="cic_ids2017_style",
        name="CIC-IDS2017-style (Synthetic Benchmark)",
        category=ProvenanceCategory.SYNTHETIC_SIMULATION,
        local_file_path="BurstPulse_Real_Datasets/CIC-IDS2017/network_traffic.csv",
        num_observations=10000,
        description="Synthetic CIC-IDS2017-style benchmark profile with stable baseline (λ ≈ 60 packets/s), shaped multiplicative surges, DoS bursts, brute-force windows, and port scanning.",
        expected_findings="Clean 30% baseline partition (λ̂ ≈ 60.00 pkts/s, D_base ≈ 1.02) isolates attack windows and volumetric bursts while maintaining low false alarm rates on stable benign periods.",
        limitations=[
            "Pure synthetic simulation generated via fixed random seed (seed=20170001); not authentic UNB testbed capture.",
            "Statistical burst flags denote volume rarity, not confirmed malicious intent.",
            "Does not contain packet payloads or hardware-level frame timings."
        ],
        has_labels=True,
        label_names=["BENIGN", "SIM_DDoS_Window", "SIM_DoS_Burst", "SIM_BruteForce_Window", "SIM_PortScan_Window"],
        documented_event="Simulated DDoS, DoS bursts, Brute Force, and PortScan events outside the 30% baseline calibration window.",
        traffic_nature="Synthetic Poisson and Binomial integer arrivals generated with seed=20170001.",
        approx_size="10,000 records (~650 KB)",
        verified_original_identifier="generate_datasets.py v1.0.0 (CIC-IDS2017 profile, seed=20170001)",
        actual_time_resolution="1-second discrete intervals",
        measurement_unit="packets_and_flows_per_1s_interval",
        interval_seconds=1,
        clean_baseline_intervals=3000,
        clean_baseline_pct=30,
        is_synthetic=True,
        academic_notes="Establishes normal baseline behavior where the Poisson equi-dispersion assumption Var(X) ≈ E[X] holds during baseline calibration.",
        provenance_statements=PROVENANCE_STATEMENTS
    ),

    "cse_cic_ids2018_style": DatasetEntry(
        id="cse_cic_ids2018_style",
        name="CSE-CIC-IDS2018-style (Synthetic Benchmark)",
        category=ProvenanceCategory.SYNTHETIC_SIMULATION,
        local_file_path="BurstPulse_Real_Datasets/CSE-CIC-IDS2018/network_traffic.csv",
        num_observations=10000,
        description="Synthetic CSE-CIC-IDS2018-style benchmark profile over 5-second intervals (~13.9 h) with multi-regime diurnal background, overdispersed segments, sustained DDoS floods, and botnet activity.",
        expected_findings="Demonstrates sustained attack surge detection against shifting background regimes and highlights overdispersion effects in night/business-hour transitions.",
        limitations=[
            "Synthetic multi-regime simulation; not genuine AWS testbed capture (seed=20180002).",
            "Heavier-tailed overdispersed background regimes produce elevated burst flags under simple Poisson assumptions.",
            "Multi-regime diurnal wander violates stationary rate assumption across long windows."
        ],
        has_labels=True,
        label_names=["BENIGN", "SIM_DDoS_Sustained", "SIM_Botnet_Window", "SIM_WebAttack_Window", "SIM_DoS_Surge", "SIM_BruteForce_Burst"],
        documented_event="Multi-regime shifts, sustained 400-interval DDoS flood, botnet traffic, and web attack windows.",
        traffic_nature="Synthetic multi-regime gamma-Poisson and Poisson arrivals generated with seed=20180002.",
        approx_size="10,000 records (~670 KB)",
        verified_original_identifier="generate_datasets.py v1.0.0 (CSE-CIC-IDS2018 profile, seed=20180002)",
        actual_time_resolution="5-second discrete intervals",
        measurement_unit="packets_and_flows_per_5s_interval",
        interval_seconds=5,
        clean_baseline_intervals=3000,
        clean_baseline_pct=30,
        is_synthetic=True,
        academic_notes="Demonstrates continuous burst detection across sustained multi-interval attack campaigns and illustrates variance clustering challenges.",
        provenance_statements=PROVENANCE_STATEMENTS
    ),

    "unsw_nb15_style": DatasetEntry(
        id="unsw_nb15_style",
        name="UNSW-NB15-style (Synthetic Benchmark)",
        category=ProvenanceCategory.SYNTHETIC_SIMULATION,
        local_file_path="BurstPulse_Real_Datasets/UNSW-NB15/network_traffic.csv",
        num_observations=10000,
        description="Synthetic UNSW-NB15-style benchmark profile featuring slowly wandering background rates, gradual anomaly ramps, and reconnaissance activity that heavily elevates flow counts relative to packet counts.",
        expected_findings="Contrasting packet count versus flow count analysis reveals reconnaissance scans that cause pronounced surges in flow arrivals while packet arrival surges remain modest.",
        limitations=[
            "Synthetic profile with simulated wander; not authentic IXIA PerfectStorm captures (seed=20150003).",
            "Gradual anomaly ramps may evade thresholding until traffic volume reaches peak ramp.",
            "Wandering background rates induce mild baseline overdispersion (D_base ≈ 2.08)."
        ],
        has_labels=True,
        label_names=["BENIGN", "SIM_Recon_Scan", "SIM_Exploit_Burst", "SIM_Generic_Anomaly", "SIM_Fuzzers_Window"],
        documented_event="Reconnaissance scanning, exploit surges, fuzzers, and gradual anomaly ramps.",
        traffic_nature="Synthetic non-stationary Poisson process with AR(1) wander and Binomial flow generation.",
        approx_size="10,000 records (~660 KB)",
        verified_original_identifier="generate_datasets.py v1.0.0 (UNSW-NB15 profile, seed=20150003)",
        actual_time_resolution="1-second discrete intervals",
        measurement_unit="packets_and_flows_per_1s_interval",
        interval_seconds=1,
        clean_baseline_intervals=3000,
        clean_baseline_pct=30,
        is_synthetic=True,
        academic_notes="Demonstrates the crucial diagnostic contrast between packet volume and flow volume in reconnaissance anomaly detection.",
        provenance_statements=PROVENANCE_STATEMENTS
    ),

    "bot_iot_style": DatasetEntry(
        id="bot_iot_style",
        name="BoT-IoT-style (Synthetic Benchmark)",
        category=ProvenanceCategory.SYNTHETIC_SIMULATION,
        local_file_path="BurstPulse_Real_Datasets/BoT-IoT/network_traffic.csv",
        num_observations=10000,
        description="Synthetic BoT-IoT-style benchmark profile modeling low-rate IoT device traffic (λ ≈ 12 packets/s) with periodic telemetry transmissions, massive volumetric floods (up to 30x baseline), and exponential recovery tails.",
        expected_findings="Extreme volumetric surges produce p-values approaching machine zero, achieving near 100% recall during active flood campaigns while post-attack recovery exhibits lingering elevated counts.",
        limitations=[
            "Synthetic device traffic; not physical smart-home testbed captures (seed=20180004).",
            "Periodic telemetry pulses create slight baseline dispersion (D_base ≈ 2.25).",
            "Real volumetric floods may saturate upstream queues, capping arrival rates."
        ],
        has_labels=True,
        label_names=["BENIGN", "SIM_DDoS_TCP_Flood", "SIM_DDoS_UDP_Flood", "SIM_DoS_HTTP_Flood", "SIM_Keylogger_Exfil"],
        documented_event="Massive IoT botnet volumetric DDoS and DoS floods with exponential recovery decay.",
        traffic_nature="Synthetic IoT device simulation with periodic telemetry spikes and volumetric flood surges.",
        approx_size="10,000 records (~655 KB)",
        verified_original_identifier="generate_datasets.py v1.0.0 (BoT-IoT profile, seed=20180004)",
        actual_time_resolution="1-second discrete intervals",
        measurement_unit="packets_and_flows_per_1s_interval",
        interval_seconds=1,
        clean_baseline_intervals=3000,
        clean_baseline_pct=30,
        is_synthetic=True,
        academic_notes="Demonstrates severe volumetric anomalies where burst arrival counts exceed baseline by order of magnitude.",
        provenance_statements=PROVENANCE_STATEMENTS
    ),

    "ton_iot_style": DatasetEntry(
        id="ton_iot_style",
        name="TON_IoT-style (Synthetic Benchmark)",
        category=ProvenanceCategory.SYNTHETIC_SIMULATION,
        local_file_path="BurstPulse_Real_Datasets/TON_IoT/network_traffic.csv",
        num_observations=10000,
        description="Synthetic TON_IoT-style benchmark profile over 2-second intervals with cyclic sensor behavior, multi-rate operating modes, intermittent burst trains (6-interval surges), and noise segments.",
        expected_findings="Intermittent pulse bursts test burst detector responsiveness across repeated episodic attacks, while device cyclic variation illustrates non-stationary background challenges.",
        limitations=[
            "Synthetic telemetry profile; not authentic UNSW Canberra testbed packets (seed=20190005).",
            "Cyclic variations and intermittent bursts introduce dispersion challenging constant-rate assumptions.",
            "Short transient pulses may recover before external alerting mechanisms react."
        ],
        has_labels=True,
        label_names=["BENIGN", "SIM_Backdoor_Surge", "SIM_Injection_Window", "SIM_Ransomware_Burst", "SIM_Scanning_Periodic"],
        documented_event="Intermittent backdoor surges, injection windows, ransomware bursts, and periodic scanning.",
        traffic_nature="Synthetic sensor cycle simulation with episodic burst pulses and gamma-Poisson noise.",
        approx_size="10,000 records (~660 KB)",
        verified_original_identifier="generate_datasets.py v1.0.0 (TON_IoT profile, seed=20190005)",
        actual_time_resolution="2-second discrete intervals",
        measurement_unit="packets_and_flows_per_2s_interval",
        interval_seconds=2,
        clean_baseline_intervals=3000,
        clean_baseline_pct=30,
        is_synthetic=True,
        academic_notes="Highlights episodic intermittent bursts and demonstrates dispersion metrics under cyclic telemetry patterns.",
        provenance_statements=PROVENANCE_STATEMENTS
    )
}

# The main dataset registry
DATASET_REGISTRY: Dict[str, DatasetEntry] = BUILTIN_DATASETS

# Backward-compatibility alias map for tests and alternate naming schemes
DATASET_ALIASES: Dict[str, str] = {
    # Catalog folder names
    "CIC-IDS2017": "cic_ids2017_style",
    "CSE-CIC-IDS2018": "cse_cic_ids2018_style",
    "UNSW-NB15": "unsw_nb15_style",
    "BoT-IoT": "bot_iot_style",
    "TON_IoT": "ton_iot_style",

    # Exact hyphenated style names from prompt
    "CIC-IDS2017-style": "cic_ids2017_style",
    "CSE-CIC-IDS2018-style": "cse_cic_ids2018_style",
    "UNSW-NB15-style": "unsw_nb15_style",
    "BoT-IoT-style": "bot_iot_style",
    "TON_IoT-style": "ton_iot_style",

    # Snake-case short names
    "cic_ids2017": "cic_ids2017_style",
    "cse_cic_ids2018": "cse_cic_ids2018_style",
    "unsw_nb15": "unsw_nb15_style",
    "bot_iot": "bot_iot_style",
    "ton_iot": "ton_iot_style",

    # Legacy dataset IDs for backward compatibility with earlier tests
    "normal_network_traffic": "cic_ids2017_style",
    "network_traffic_with_bursts": "cse_cic_ids2018_style",
    "port_scan_simulation": "unsw_nb15_style",
    "ddos_style_traffic_surge": "bot_iot_style",
    "overdispersed_network_traffic": "ton_iot_style",

    # Legacy test aliases
    "synthetic_demo": "cic_ids2017_style",
    "synthetic_portscan_demo": "unsw_nb15_style",
    "synthetic_overdispersed_demo": "ton_iot_style",
    "cic_ids_2017_portscan": "unsw_nb15_style",
    "mawi_samplepoint_f": "ton_iot_style"
}


def get_dataset(dataset_id: str) -> Optional[DatasetEntry]:
    """Retrieve a built-in dataset record by its identifier with alias and case-insensitive resolution."""
    if not dataset_id:
        return None
    if dataset_id in BUILTIN_DATASETS:
        return BUILTIN_DATASETS[dataset_id]
    if dataset_id in DATASET_ALIASES:
        target = DATASET_ALIASES[dataset_id]
        if target in BUILTIN_DATASETS:
            return BUILTIN_DATASETS[target]

    # Normalized lookup fallback
    norm = dataset_id.lower().replace("-", "_").replace(" ", "_")
    for k, v in BUILTIN_DATASETS.items():
        if k.lower().replace("-", "_") == norm:
            return v
    for a, target in DATASET_ALIASES.items():
        if a.lower().replace("-", "_") == norm and target in BUILTIN_DATASETS:
            return BUILTIN_DATASETS[target]

    return None


def list_builtin_datasets() -> List[DatasetEntry]:
    """Return exactly the five built-in research-inspired dataset entries."""
    return list(BUILTIN_DATASETS.values())


def list_datasets(*args, **kwargs) -> List[DatasetEntry]:
    """List datasets (for backward compatibility)."""
    return list_builtin_datasets()


def get_local_datasets() -> List[DatasetEntry]:
    """Return all built-in datasets available locally."""
    return list_builtin_datasets()
