"""
tests/test_dataset_registry.py
Unit tests for the 5 Research-Inspired Synthetic Dataset Registry, Metadata, and Case Studies Catalog.
"""

from pathlib import Path
import pytest
import pandas as pd
from dataset_registry import (
    BUILTIN_DATASETS,
    DATASET_REGISTRY,
    DATASET_ALIASES,
    ProvenanceCategory,
    DatasetCategory,
    AvailabilityStatus,
    get_dataset,
    list_builtin_datasets,
    get_local_datasets
)
from case_studies import CASE_STUDIES, get_case_study, list_case_studies


def test_registry_contains_exactly_five_builtin_datasets():
    """Verify that the built-in registry contains exactly the five research-inspired local datasets."""
    expected_ids = [
        "cic_ids2017_style",
        "cse_cic_ids2018_style",
        "unsw_nb15_style",
        "bot_iot_style",
        "ton_iot_style"
    ]
    assert len(BUILTIN_DATASETS) == 5, f"Expected exactly 5 built-in datasets, got {len(BUILTIN_DATASETS)}"
    for d_id in expected_ids:
        assert d_id in BUILTIN_DATASETS, f"Missing required built-in dataset: {d_id}"


def test_all_five_dataset_files_exist_and_have_valid_schemas():
    """Verify all 5 built-in dataset files exist on disk, are non-empty, and contain required columns."""
    for d_id, entry in BUILTIN_DATASETS.items():
        assert entry.local_file_path is not None, f"Dataset {d_id} has no file path"
        p = entry.resolved_path
        assert p.exists(), f"File {p} does not exist"
        assert p.stat().st_size > 0, f"File {p} is empty"
        assert entry.is_analyzable is True

        # Validate CSV contents
        df = pd.read_csv(p)
        assert len(df) == 10000, f"Dataset {d_id} should have exactly 10,000 rows"
        assert "packet_count" in df.columns, f"Dataset {d_id} missing packet_count column"
        assert "flow_count" in df.columns, f"Dataset {d_id} missing flow_count column"
        assert "timestamp" in df.columns, f"Dataset {d_id} missing timestamp column"
        assert "has_attack_label" in df.columns, f"Dataset {d_id} missing has_attack_label column"
        assert "is_synthetic" in df.columns, f"Dataset {d_id} missing is_synthetic column"
        assert bool(df["is_synthetic"].all()) is True, f"Dataset {d_id} must have is_synthetic True everywhere"
        assert (df["packet_count"] >= 0).all(), f"Dataset {d_id} has negative counts"
        assert (df["packet_count"] % 1 == 0).all(), f"Dataset {d_id} has fractional counts"
        assert (df["flow_count"] >= 0).all(), f"Dataset {d_id} has negative flow counts"


def test_all_five_datasets_have_complete_descriptions_and_limitations():
    """Ensure every built-in dataset has informative descriptions, expectations, and limitations."""
    for d_id, entry in BUILTIN_DATASETS.items():
        assert len(entry.name) > 0
        assert len(entry.description) > 20, f"Description too short for {d_id}"
        assert len(entry.expected_findings) > 20, f"Expected findings too short for {d_id}"
        assert len(entry.limitations) >= 1, f"Missing limitations for {d_id}"
        assert entry.num_observations == 10000
        assert entry.category == ProvenanceCategory.SYNTHETIC_SIMULATION
        assert entry.clean_baseline_pct == 30
        assert entry.clean_baseline_intervals == 3000
        assert len(entry.provenance_statements) > 0


def test_alias_resolution_for_backward_compatibility():
    """Verify alias mapping allows legacy and alternate test IDs to resolve gracefully."""
    # Exact hyphenated style IDs
    assert get_dataset("CIC-IDS2017-style") is not None
    assert get_dataset("CIC-IDS2017-style").id == "cic_ids2017_style"
    assert get_dataset("CSE-CIC-IDS2018-style").id == "cse_cic_ids2018_style"
    assert get_dataset("UNSW-NB15-style").id == "unsw_nb15_style"
    assert get_dataset("BoT-IoT-style").id == "bot_iot_style"
    assert get_dataset("TON_IoT-style").id == "ton_iot_style"

    # Catalog folder names
    assert get_dataset("CIC-IDS2017").id == "cic_ids2017_style"
    assert get_dataset("CSE-CIC-IDS2018").id == "cse_cic_ids2018_style"
    assert get_dataset("UNSW-NB15").id == "unsw_nb15_style"
    assert get_dataset("BoT-IoT").id == "bot_iot_style"
    assert get_dataset("TON_IoT").id == "ton_iot_style"

    # Legacy IDs from initial demo
    assert get_dataset("normal_network_traffic") is not None
    assert get_dataset("network_traffic_with_bursts") is not None
    assert get_dataset("port_scan_simulation") is not None
    assert get_dataset("ddos_style_traffic_surge") is not None
    assert get_dataset("overdispersed_network_traffic") is not None

    # Legacy test aliases
    assert get_dataset("synthetic_demo") is not None
    assert get_dataset("synthetic_portscan_demo") is not None
    assert get_dataset("synthetic_overdispersed_demo") is not None
    assert get_dataset("cic_ids_2017_portscan") is not None


def test_no_external_download_required_for_builtin_datasets():
    """Verify all 5 built-in datasets are marked LOCAL_READY and do not require external downloads."""
    for entry in list_builtin_datasets():
        assert entry.availability == AvailabilityStatus.LOCAL_READY
        assert entry.local_file_path is not None
        assert entry.resolved_path.exists()


def test_case_studies_registry_remains_available():
    """Verify historical incident case studies catalog remains accessible for reference."""
    studies = list_case_studies()
    assert len(studies) >= 3
    study_ids = [s.id for s in studies]
    assert "caida_2007" in study_ids
    assert "mirai_dyn_2016" in study_ids
