"""
tests/test_app_integration.py
End-to-end integration tests using Streamlit's official AppTest framework.
Verifies the Landing Page, 5 Built-in Datasets, Drag-and-Drop CSV Upload,
Live Recalculation, Data-Driven Verification, and Removal of External Pages.
"""

from pathlib import Path
import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
NORMAL_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "normal_network_traffic.csv"
BURSTS_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "network_traffic_with_bursts.csv"
PORTSCAN_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "port_scan_simulation.csv"
DDOS_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "ddos_style_traffic_surge.csv"
OVERDISPERSED_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "overdispersed_network_traffic.csv"


def test_landing_page_renders_hero_and_ctas():
    """Verify landing page renders first with headline, badges, and navigation CTAs."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception, f"App threw unexpected exception: {[e.value for e in at.exception]}"

    button_labels = [b.label for b in at.button]
    assert any("Open Analyzer" in lbl for lbl in button_labels)
    assert any("Launch Analyzer" in lbl for lbl in button_labels)


def test_landing_page_to_analyzer_and_back_navigation():
    """Verify bidirectional navigation between Landing Page and Analyzer."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Click Launch Analyzer
    hero_btn = [b for b in at.button if "Launch Analyzer" in b.label][0]
    hero_btn.click().run()
    assert not at.exception

    assert len(at.title) >= 1
    assert "Network Traffic Burst Analysis" in at.title[0].value
    assert len(at.metric) >= 6

    # Click Back to Product Overview
    back_btn = [b for b in at.button if "Back to Product Overview" in b.label][0]
    back_btn.click().run()
    assert not at.exception
    assert any("Launch Analyzer" in b.label for b in at.button)


def test_analyzer_has_exactly_two_data_source_modes():
    """Verify the analyzer presents exactly two data source modes: Built-in and CSV Upload."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    source_radios = [r for r in at.radio if "Data Source Selection" in r.label]
    assert len(source_radios) == 1
    options = source_radios[0].options
    assert len(options) == 2
    assert "Built-in Datasets" in options
    assert "Upload Your Own CSV" in options


def test_analyzer_has_exactly_five_builtin_datasets_in_dropdown():
    """Verify the Built-in dropdown contains exactly the five specified local datasets."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    assert len(at.selectbox) >= 1
    ds_selector = at.selectbox[0]
    assert "Select a Network Traffic Dataset" in ds_selector.label
    assert len(ds_selector.options) == 5

    formatted_options = " ".join(ds_selector.options)
    assert "cic_ids2017_style" in formatted_options or "CIC-IDS2017" in formatted_options
    assert "cse_cic_ids2018_style" in formatted_options or "CSE-CIC-IDS2018" in formatted_options
    assert "unsw_nb15_style" in formatted_options or "UNSW-NB15" in formatted_options
    assert "bot_iot_style" in formatted_options or "BoT-IoT" in formatted_options
    assert "ton_iot_style" in formatted_options or "TON_IoT" in formatted_options

    # Ensure external research datasets are NOT in the dropdown
    assert "CIC-IDS2017 Research Benchmark" not in formatted_options
    assert "MAWI Backbone Traffic" not in formatted_options
    assert "CAIDA" not in formatted_options


def test_all_five_builtin_datasets_analyze_immediately():
    """Verify all 5 built-in datasets load, analyze, and display complete metrics immediately."""
    for idx in range(5):
        at = AppTest.from_file(APP_PATH, default_timeout=15)
        at.query_params["page"] = "analyzer"
        at.run()
        assert not at.exception

        at.selectbox[0].select_index(idx).run()
        assert not at.exception

        metric_map = {m.label: m.value for m in at.metric}
        assert "Total Observations" in metric_map
        assert "Total Packets" in metric_map or "Total Flows" in metric_map
        assert "Baseline Mean (λ̂)" in metric_map
        assert "Evaluated Intervals" in metric_map
        assert "Potential Bursts" in metric_map
        assert "Peak Evaluated Count" in metric_map


def test_selecting_dataset_updates_description_without_navigation():
    """Verify changing dataset selection updates the inline description without navigating away."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    # Default is cic_ids2017_style
    assert at.selectbox[0].value == "cic_ids2017_style"

    # Switch to BoT-IoT (index 3)
    at.selectbox[0].select_index(3).run()
    assert not at.exception

    assert at.selectbox[0].value == "bot_iot_style"
    assert any("Network Traffic Burst Analysis" in t.value for t in at.title)

    metric_map = {m.label: m.value for m in at.metric}
    assert metric_map["Total Observations"] == "10,000 int"


def test_changing_baseline_or_alpha_recalculates_analysis():
    """Verify changing alpha or baseline slider recalculates metrics data-driven."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    # On normal traffic at alpha=0.001
    initial_bursts = at.metric[4].value  # Potential Bursts

    # Set permissive threshold alpha=0.05
    at.select_slider[0].set_value(0.05).run()
    assert not at.exception

    # Change baseline slider from 30% to 90%
    at.slider[0].set_value(90).run()
    assert not at.exception

    metric_map = {m.label: m.value for m in at.metric}
    assert "Baseline Mean (λ̂)" in metric_map
    assert "Evaluated Intervals" in metric_map
    assert metric_map["Evaluated Intervals"] == "1,000 int"  # 10% of 10,000


def test_synthetic_labels_preserved_and_contingency_displayed():
    """Verify datasets with simulation labels display contingency analysis."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    # Select CIC-IDS2017 (index 0)
    at.selectbox[0].select_index(0).run()
    assert not at.exception

    metric_map = {m.label: m.value for m in at.metric}
    assert "True Positives (Overlap)" in metric_map
    assert "Statistical False Alarms" in metric_map
    assert "Empirical Recall" in metric_map


def test_user_csv_upload_interval_counts():
    """Verify uploading interval-count CSV analyzes user data without fallback to synthetic."""
    with open(NORMAL_CSV_PATH, "rb") as f:
        file_bytes = f.read()

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    # Switch radio to "Upload Your Own CSV"
    at.radio[0].set_value("Upload Your Own CSV").run()
    assert not at.exception

    at.file_uploader[0].upload("user_traffic.csv", file_bytes).run()
    assert not at.exception

    metric_map = {m.label: m.value for m in at.metric}
    assert "Total Observations" in metric_map
    assert metric_map["Total Observations"] == "250 int"
    assert "Total Packets" in metric_map


def test_user_csv_upload_packet_event_timestamps():
    """Verify uploading packet timestamp stream aggregates correctly into intervals."""
    # 4 packets: 2 at second 0, 0 at second 1, 2 at second 2
    packet_stream_csv = b"timestamp,src_ip\n2026-10-15 10:00:00,10.0.0.1\n2026-10-15 10:00:00.5,10.0.0.2\n2026-10-15 10:00:02,10.0.0.3\n2026-10-15 10:00:02.5,10.0.0.4\n" + \
        b"\n".join([f"2026-10-15 10:00:{i:02d},10.0.0.5".encode() for i in range(3, 20)])

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    at.radio[0].set_value("Upload Your Own CSV").run()
    at.file_uploader[0].upload("packet_stream.csv", packet_stream_csv).run()
    assert not at.exception

    metric_map = {m.label: m.value for m in at.metric}
    assert "Total Packets" in metric_map


def test_invalid_csv_shows_helpful_error_without_crash():
    """Verify malformed CSV shows helpful validation error without crash or silent substitution."""
    bad_csv = b"interval,packet_count\n1,20\n2,-5\n3,18\n4,22\n5,21\n6,20\n7,19\n8,24\n9,22\n10,21\n"

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    at.radio[0].set_value("Upload Your Own CSV").run()
    at.file_uploader[0].upload("bad_traffic.csv", bad_csv).run()
    assert not at.exception

    error_texts = [e.value for e in at.error]
    assert any("cannot be negative" in txt for txt in error_texts)


def test_query_params_routing_and_direct_urls():
    """Verify root URL, ?page=landing, and ?page=analyzer route correctly."""
    at_root = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at_root.exception
    assert any("Launch Analyzer" in b.label for b in at_root.button)

    at_analyzer = AppTest.from_file(APP_PATH, default_timeout=15)
    at_analyzer.query_params["page"] = "analyzer"
    at_analyzer.run()
    assert not at_analyzer.exception
    assert any("Network Traffic Burst Analysis" in t.value for t in at_analyzer.title)

    at_landing = AppTest.from_file(APP_PATH, default_timeout=15)
    at_landing.query_params["page"] = "landing"
    at_landing.run()
    assert not at_landing.exception
    assert any("Launch Analyzer" in b.label for b in at_landing.button)


def test_no_external_dataset_pages_in_analyzer():
    """Verify that external dataset download pages and CAIDA AUA procedures are completely removed from analyzer."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    all_markdown = " ".join([m.value for m in at.markdown])
    assert "Acceptable Use Agreement" not in all_markdown
    assert "Request Data Access" not in all_markdown
    assert "s3://cse-cic-ids2018/" not in all_markdown


def test_user_csv_upload_flow_records_aggregation():
    """Verify uploading flow-level CSV aggregates packet totals across flows into discrete intervals."""
    # Generate flow records spanning 150 seconds (15 ten-second intervals)
    flow_rows = [f"2026-10-15 10:{i//60:02d}:{i%60:02d},10,5".encode() for i in range(150)]
    flow_csv = b"timestamp,total fwd packets,total backward packets\n" + b"\n".join(flow_rows)

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    at.radio[0].set_value("Upload Your Own CSV").run()
    at.file_uploader[0].upload("flows.csv", flow_csv).run()
    assert not at.exception

    metric_map = {m.label: m.value for m in at.metric}
    assert "Total Packets" in metric_map


def test_unlabeled_user_data_does_not_receive_fabricated_classification_metrics():
    """Verify unlabeled user data computes statistical bursts but does NOT fabricate precision/recall metrics."""
    with open(NORMAL_CSV_PATH, "rb") as f:
        file_bytes = f.read()

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    at.radio[0].set_value("Upload Your Own CSV").run()
    at.file_uploader[0].upload("normal_unlabeled.csv", file_bytes).run()
    assert not at.exception

    metric_map = {m.label: m.value for m in at.metric}
    assert "Total Observations" in metric_map
    # Ensure precision, recall, and true positives are NOT rendered when no labels exist
    assert "True Positives (Overlap)" not in metric_map
    assert "Empirical Recall" not in metric_map
    assert "Precision" not in metric_map


def test_antenna_signal_animation_renders_on_dataset_selection():
    """Verify the antenna signal animation banner renders upon selecting built-in dataset."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    iframe_docs = [x.proto.srcdoc for x in at._tree if hasattr(x, "proto") and hasattr(x.proto, "srcdoc")]
    assert len(iframe_docs) >= 1
    doc = iframe_docs[0]
    assert "antenna-signal-container" in doc
    assert "Signal Synchronized" in doc
    assert "antenna-svg-wrap active-signal" in doc
    # Confirm no raw SVG/CSS/HTML code is leaked into markdown text elements
    all_markdown = " ".join([m.value for m in at.markdown])
    assert "@keyframes" not in all_markdown
    assert "<svg" not in all_markdown


def test_antenna_signal_stays_idle_on_slider_recalculation():
    """Verify adjusting baseline/alpha sliders does not re-trigger wave animation (stays in idle/synchronized state)."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    # First run triggers active-signal on initial load
    first_docs = [x.proto.srcdoc for x in at._tree if hasattr(x, "proto") and hasattr(x.proto, "srcdoc")]
    assert "antenna-svg-wrap active-signal" in first_docs[0]

    # Rerun with slider change only (same dataset ID)
    at.select_slider[0].set_value(0.05).run()
    assert not at.exception

    second_docs = [x.proto.srcdoc for x in at._tree if hasattr(x, "proto") and hasattr(x.proto, "srcdoc")]
    assert "antenna-svg-wrap idle-signal" in second_docs[0]
    assert "Signal Synchronized" in second_docs[0]


def test_antenna_signal_renders_on_csv_upload():
    """Verify antenna signal animation triggers on successful user CSV upload."""
    with open(NORMAL_CSV_PATH, "rb") as f:
        file_bytes = f.read()

    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    at.radio[0].set_value("Upload Your Own CSV").run()
    at.file_uploader[0].upload("user_traffic.csv", file_bytes).run()
    assert not at.exception

    iframe_docs = [x.proto.srcdoc for x in at._tree if hasattr(x, "proto") and hasattr(x.proto, "srcdoc")]
    assert len(iframe_docs) >= 1
    doc = iframe_docs[0]
    assert "antenna-signal-container" in doc
    assert "CSV Signal Locked" in doc
    assert "antenna-svg-wrap active-signal" in doc


def test_landing_page_mathematical_foundation_renders_st_latex():
    """Verify Mathematical Foundation uses native st.latex for equations without raw unrendered blocks."""
    at = AppTest.from_file(APP_PATH, default_timeout=15).run()
    assert not at.exception

    # Landing page must contain native st.latex equations
    latex_vals = [elem.value for elem in at.latex]
    assert len(latex_vals) >= 5, f"Expected at least 5 st.latex elements on landing page, got {len(latex_vals)}"

    # 1. Poisson PMF
    assert any(r"\frac{\lambda^k e^{-\lambda}}{k!}" in val for val in latex_vals)
    # 2. Upper-tail cumulative probability
    assert any(r"P(X \ge k) = 1 - P(X \le k - 1)" in val for val in latex_vals)
    assert any(r"\sum_{j=k}^{\infty} \frac{\lambda^j e^{-\lambda}}{j!}" in val for val in latex_vals)
    # 3. Baseline estimation and classification rule
    assert any(r"\hat{\lambda} = \frac{1}{m}\sum_{i=1}^{m} x_i" in val for val in latex_vals)
    assert any(r"\text{Flag as potential burst if } p < \alpha" in val for val in latex_vals)


def test_analyzer_mathematical_expander_renders_st_latex():
    """Verify analyzer math expander also renders native st.latex equations."""
    at = AppTest.from_file(APP_PATH, default_timeout=15)
    at.query_params["page"] = "analyzer"
    at.run()
    assert not at.exception

    latex_vals = [elem.value for elem in at.latex]
    assert len(latex_vals) >= 5, f"Expected at least 5 st.latex elements in analyzer, got {len(latex_vals)}"
    assert any(r"\frac{\lambda^k e^{-\lambda}}{k!}" in val for val in latex_vals)
    assert any(r"\hat{\lambda} = \frac{1}{m} \sum_{i=1}^{m} x_i" in val for val in latex_vals)


