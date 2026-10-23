"""
tests/test_dataset_registry.py
Unit tests for the 5 Built-in Dataset Registry, Metadata, and Case Studies Catalog.
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
    """Verify that the built-in registry contains exactly the five specified local datasets."""
    expected_ids = [
        "normal_network_traffic",
        "network_traffic_with_bursts",
        "port_scan_simulation",
        "ddos_style_traffic_surge",
        "overdispersed_network_traffic"
    ]
    assert len(BUILTIN_DATASETS) == 5, f"Expected exactly 5 built-in datasets, got {len(BUILTIN_DATASETS)}"
    for d_id in expected_ids:
        assert d_id in BUILTIN_DATASETS, f"Missing required built-in dataset: {d_id}"


def test_all_five_dataset_files_exist_and_have_valid_schemas():
    """Verify all 5 built-in dataset files exist on disk, are non-empty, and contain required columns."""
    for d_id, entry in BUILTIN_DATASETS.items():
        assert entry.local_file_path is not None, f"Dataset {d_id} has no file path"
        p = Path(entry.local_file_path)
        assert p.exists(), f"File {entry.local_file_path} does not exist"
        assert p.stat().st_size > 0, f"File {entry.local_file_path} is empty"
        assert entry.is_analyzable is True

        # Validate CSV contents
        df = pd.read_csv(p)
        assert len(df) >= 200, f"Dataset {d_id} should have at least 200 rows"
        assert "packet_count" in df.columns, f"Dataset {d_id} missing packet_count column"
        assert (df["packet_count"] >= 0).all(), f"Dataset {d_id} has negative counts"
        assert (df["packet_count"] % 1 == 0).all(), f"Dataset {d_id} has fractional counts"


def test_all_five_datasets_have_complete_descriptions_and_limitations():
    """Ensure every built-in dataset has informative descriptions, expectations, and limitations."""
    for d_id, entry in BUILTIN_DATASETS.items():
        assert len(entry.name) > 0
        assert len(entry.description) > 20, f"Description too short for {d_id}"
        assert len(entry.expected_findings) > 20, f"Expected findings too short for {d_id}"
        assert len(entry.limitations) >= 1, f"Missing limitations for {d_id}"
        assert entry.num_observations == len(pd.read_csv(entry.local_file_path))
        assert entry.category == ProvenanceCategory.SYNTHETIC_SIMULATION


def test_alias_resolution_for_backward_compatibility():
    """Verify alias mapping allows legacy test IDs to resolve gracefully."""
    assert get_dataset("synthetic_demo") is not None
    assert get_dataset("synthetic_demo").id == "normal_network_traffic"

    assert get_dataset("synthetic_portscan_demo") is not None
    assert get_dataset("synthetic_portscan_demo").id == "port_scan_simulation"

    assert get_dataset("synthetic_overdispersed_demo") is not None
    assert get_dataset("synthetic_overdispersed_demo").id == "overdispersed_network_traffic"

    assert get_dataset("cic_ids_2017_portscan") is not None
    assert get_dataset("cic_ids_2017_portscan").id == "port_scan_simulation"


def test_no_external_download_required_for_builtin_datasets():
    """Verify all 5 built-in datasets are marked LOCAL_READY and do not require external downloads."""
    for entry in list_builtin_datasets():
        assert entry.availability == AvailabilityStatus.LOCAL_READY
        assert entry.local_file_path is not None


def test_case_studies_registry_remains_available():
    """Verify historical incident case studies catalog remains accessible for reference."""
    studies = list_case_studies()
    assert len(studies) >= 3
    study_ids = [s.id for s in studies]
    assert "caida_2007" in study_ids
    assert "mirai_dyn_2016" in study_ids
