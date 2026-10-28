"""
tests/test_deployment_regression.py
Regression tests for Streamlit Community Cloud production deployment:
1. Validates .streamlit/config.toml server configurations (0.0.0.0, port 8501, headless).
2. Validates .python-version declaration.
3. Verifies path resolution robustness across all 5 benchmark datasets independent of CWD.
4. Verifies graceful error handling for missing/malformed datasets without unhandled crashes.
5. Verifies all 5 datasets load and calculate Poisson upper-tail probabilities successfully.
"""

from pathlib import Path
import os
import tomllib
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

from dataset_registry import (
    BUILTIN_DATASETS,
    PROJECT_ROOT,
    get_dataset,
    list_builtin_datasets
)
from burst_detector import analyze_traffic, compute_upper_tail_probability

APP_PATH = str(PROJECT_ROOT / "app.py")


def test_streamlit_config_toml_configured_for_cloud_deployment():
    """Verify .streamlit/config.toml exists and configures address 0.0.0.0, port 8501, and headless."""
    config_path = PROJECT_ROOT / ".streamlit" / "config.toml"
    assert config_path.is_file(), ".streamlit/config.toml must exist"

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    assert "server" in config, "config.toml must have [server] section"
    server_cfg = config["server"]
    assert server_cfg.get("headless") is True, "server.headless must be True"
    assert server_cfg.get("address") == "0.0.0.0", "server.address must be 0.0.0.0 to prevent IPv6 health check resets"
    assert server_cfg.get("port") == 8501, "server.port must be 8501"


def test_python_version_file_specified():
    """Verify .python-version file exists and specifies a supported Python version."""
    py_ver_path = PROJECT_ROOT / ".python-version"
    assert py_ver_path.is_file(), ".python-version file must exist"
    content = py_ver_path.read_text(encoding="utf-8").strip()
    assert content in ["3.11", "3.12"], f"Expected supported Python version (3.11 or 3.12), got {content}"


def test_all_five_bundled_benchmark_datasets_exist_and_resolve_from_any_cwd():
    """Verify that resolved_path accurately locates all 5 benchmark datasets regardless of CWD."""
    original_cwd = os.getcwd()
    try:
        # Change CWD to parent directory to simulate arbitrary runtime execution CWD
        os.chdir(PROJECT_ROOT.parent)

        datasets = list_builtin_datasets()
        assert len(datasets) == 5, f"Expected 5 built-in datasets, found {len(datasets)}"

        for entry in datasets:
            p = entry.resolved_path
            assert p.is_file(), f"Dataset {entry.id} file does not exist at resolved path: {p}"
            assert p.stat().st_size > 50_000, f"Dataset {entry.id} file is suspiciously small ({p.stat().st_size} bytes)"
            assert entry.is_analyzable is True
    finally:
        os.chdir(original_cwd)


def test_missing_dataset_handled_gracefully_in_app():
    """Verify that if a dataset path is unavailable, the app stops gracefully without crashing."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.query_params["dataset"] = "non_existent_dataset_key"
    at.run()

    # The fallback should resolve to a valid default dataset without throwing unhandled exceptions
    assert not at.exception, f"App threw unexpected exception: {[e.value for e in at.exception]}"


def test_all_five_datasets_produce_valid_poisson_burst_analyses():
    """Verify that each of the 5 benchmark datasets executes the full Poisson analysis pipeline."""
    for entry in list_builtin_datasets():
        df = pd.read_csv(entry.resolved_path)
        assert len(df) == 10000, f"Expected 10,000 rows for {entry.id}"

        # Analyze at clean baseline 30% and alpha = 0.001
        res = analyze_traffic(
            df,
            baseline_pct=entry.clean_baseline_pct,
            alpha=0.001,
            count_col="packet_count"
        )

        assert res.lambda_hat > 0, f"Baseline lambda must be positive for {entry.id}"
        assert res.baseline_count == entry.clean_baseline_intervals
        assert res.evaluated_count == 7000
        assert len(res.evaluated_data) == 7000

        # Check upper-tail probability values are within valid probability interval [0, 1]
        probs = res.evaluated_data["upper_tail_probability"].values
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert res.burst_count >= 0
