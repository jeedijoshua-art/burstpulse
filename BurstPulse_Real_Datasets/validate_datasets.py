#!/usr/bin/env python3
"""
Validates the BurstPulse synthetic dataset pack, writes VALIDATION_REPORT.txt and builds
BurstPulse_Real_Datasets.zip (deterministic entry order/timestamps).

Usage: python validate_datasets.py [pack_dir]      (pack_dir = the BurstPulse_Real_Datasets folder)
Optional: set BURSTPULSE_REPO=/path/to/burstpulse to also run the repo-compatibility check.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
sys.path.insert(0, str(ROOT))
import generate_datasets as gen  # noqa: E402

FOLDERS = ["CIC-IDS2017", "CSE-CIC-IDS2018", "UNSW-NB15", "BoT-IoT", "TON_IoT"]
REQUIRED = ["timestamp", "interval", "packet_count", "flow_count", "measurement_unit", "has_attack_label",
            "attack_type", "scenario", "source_dataset_style", "is_synthetic"]
MIN_ROWS = 5000
ZIP_NAME = "BurstPulse_Real_Datasets.zip"
TOP = "BurstPulse_Real_Datasets"
EXPECTED_FILES = (["README.md", "DATA_PROVENANCE.json", "DATASET_CATALOG.csv", "VALIDATION_REPORT.txt",
                   "generate_datasets.py", "validate_datasets.py"]
                  + [f"{f}/{n}" for f in FOLDERS for n in ("network_traffic.csv", "dataset_info.json")])

results = []     # (check, status, detail)
warnings = []


def rec(name, ok, detail=""):
    results.append((name, "PASS" if ok else "FAIL", detail))
    return ok


def load(folder):
    return pd.read_csv(ROOT / folder / "network_traffic.csv")


def check_dataset_files():
    frames = {f: load(f) for f in FOLDERS}
    infos = {f: json.loads((ROOT / f / "dataset_info.json").read_text()) for f in FOLDERS}
    catalog = pd.read_csv(ROOT / "DATASET_CATALOG.csv")

    def per(name, fn, detail_fn=None):
        bad = [f for f in FOLDERS if not fn(frames[f], infos[f])]
        rec(name, not bad, (detail_fn() if detail_fn else "all 5 datasets") if not bad else f"failed: {bad}")

    # 1 required columns
    per("1. Required columns present", lambda d, i: all(c in d.columns for c in REQUIRED))
    # 2 min rows
    rows = {f: len(frames[f]) for f in FOLDERS}
    rec("2. Minimum row count (>= 5,000)", all(v >= MIN_ROWS for v in rows.values()), f"rows: {rows}")
    # 3 types
    def types_ok(d, i):
        ts = pd.to_datetime(d.timestamp, utc=True, errors="coerce", format="ISO8601")
        return (ts.notna().all() and pd.api.types.is_integer_dtype(d.packet_count)
                and pd.api.types.is_integer_dtype(d.flow_count) and pd.api.types.is_integer_dtype(d.interval)
                and d.has_attack_label.dtype == bool and d.is_synthetic.dtype == bool
                and d.timestamp.str.endswith("+00:00").all())
    per("3. Types: ISO 8601 UTC timestamps, integer counts/interval, boolean labels", types_ok)
    # 4 negative / fractional
    per("4. No negative or fractional packet/flow counts",
        lambda d, i: (d.packet_count >= 0).all() and (d.flow_count >= 0).all()
        and (d.packet_count % 1 == 0).all() and (d.flow_count % 1 == 0).all())
    # extra: flows <= packets and zero-packet => zero flows
    per("4b. flow_count <= packet_count, flow_count >= 1 whenever packet_count > 0",
        lambda d, i: (d.flow_count <= d.packet_count).all() and ((d.packet_count == 0) <= (d.flow_count == 0)).all()
        and ((d.packet_count > 0) <= (d.flow_count >= 1)).all())
    # 5 missing
    per("5. No missing values (any column, incl. empty strings)",
        lambda d, i: not d.isna().any().any() and not (d.select_dtypes("object") == "").any().any())
    # 6 duplicates
    per("6. No duplicate interval identifiers (and timestamps)",
        lambda d, i: d.interval.is_unique and d.timestamp.is_unique)
    # 7 monotonic + semantics
    def mono(d, i):
        ts = pd.to_datetime(d.timestamp, utc=True, format="ISO8601")
        step = ts.diff().dropna().dt.total_seconds()
        return (d.interval.diff().dropna() == 1).all() and d.interval.iloc[0] == 1 \
            and (step == i["interval_seconds"]).all() and f"{i['interval_seconds']}s_interval" in i["measurement_unit"] \
            and (d.measurement_unit == i["measurement_unit"]).all()
    per("7. Strictly increasing timestamps at the documented interval_seconds; unit matches metadata", mono)
    # 8 labels
    def labels(d, i):
        types = set(d.attack_type.unique())
        documented = set(i["attack_type_counts"].keys())
        atk = d.attack_type != "BENIGN"
        return (types == documented and (d.has_attack_label == atk).all()
                and all(t == "BENIGN" or t.startswith("SIM_") for t in types)
                and set(d.scenario.unique()) == set(i["scenario_counts"].keys())
                and (d.scenario == d.scenario_label).all()
                and not d[d.period == "baseline"].has_attack_label.any()
                and d.has_attack_label.any() and (~d.has_attack_label).any())
    per("8. Label values valid: attack_type BENIGN/SIM_*, has_attack_label == (attack_type != BENIGN), "
        "documented scenarios, clean baseline", labels)
    # 9 is_synthetic
    per("9. is_synthetic always True (rows, metadata, style column)",
        lambda d, i: d.is_synthetic.all() and i["is_synthetic"] is True
        and d.source_dataset_style.nunique() == 1 and d.source_dataset_style.iloc[0] == i["source_dataset_style"])
    # 10 reproducibility
    regenerated = gen.generate_all_frames()
    regen2 = gen.generate_all_frames()
    ok10, det10 = True, []
    for f in FOLDERS:
        b_file = (ROOT / f / "network_traffic.csv").read_bytes()
        b1, b2 = gen.csv_bytes(regenerated[f][0]), gen.csv_bytes(regen2[f][0])
        same = b_file == b1 == b2
        ok10 &= same
        det10.append(f"{f}:{hashlib.sha256(b_file).hexdigest()[:12]}{'=' if same else '!='}regen")
    rec("10. Reproducible: two fresh regenerations are byte-identical to files on disk", ok10, "; ".join(det10))
    # 11 catalog + metadata
    ok11, det11 = True, []
    for f in FOLDERS:
        c = catalog[catalog.dataset_folder == f]
        d, i = frames[f], infos[f]
        good = (len(c) == 1 and int(c.row_count.iloc[0]) == len(d) == i["row_count"]
                and c.file_path.iloc[0] == f"{f}/network_traffic.csv"
                and c.timestamp_start.iloc[0] == d.timestamp.iloc[0] == i["timestamp_start"]
                and c.timestamp_end.iloc[0] == d.timestamp.iloc[-1] == i["timestamp_end"]
                and int(c.random_seed.iloc[0]) == i["random_seed"]
                and c.measurement_unit.iloc[0] == i["measurement_unit"]
                and bool(c.is_synthetic.iloc[0]) is True
                and i["attack_label_intervals"] == int(d.has_attack_label.sum())
                and abs(i["packet_count_stats"]["mean"] - d.packet_count.mean()) < 1e-3
                and abs(i["flow_count_stats"]["variance"] - d.flow_count.var(ddof=1)) < 1e-3)
        ok11 &= good
        det11.append(f"{f}={len(d)}")
    cols_ok = list(catalog.columns) == ["dataset_folder", "display_name", "file_path", "row_count", "timestamp_start",
                                        "timestamp_end", "measurement_unit", "scenarios_included", "random_seed",
                                        "is_synthetic", "provenance_description"]
    prov = json.loads((ROOT / "DATA_PROVENANCE.json").read_text())
    prov_ok = prov["is_synthetic"] is True and any("NOT genuine" in s for s in prov["statements"])
    rec("11. Catalog, metadata, provenance consistent with CSVs", ok11 and cols_ok and prov_ok and len(catalog) == 5,
        ", ".join(det11))
    return frames, infos


def check_repo_compat(frames):
    repo = os.environ.get("BURSTPULSE_REPO")
    if not repo or not Path(repo, "data_preprocessor.py").exists():
        warnings.append("13. BurstPulse repo compatibility check SKIPPED (BURSTPULSE_REPO not set / repo not found).")
        results.append(("13. BurstPulse repo compatibility (inspect_schema, prepare_interval_dataset, analyze_traffic)",
                        "SKIPPED", "repo not available in this run"))
        return
    sys.path.insert(0, repo)
    from data_preprocessor import inspect_schema, prepare_interval_dataset, SchemaType
    from burst_detector import analyze_traffic
    ok, det = True, []
    for f, df in frames.items():
        schema, _ = inspect_schema(df)
        pre = prepare_interval_dataset(df, interval_sec=json.loads((ROOT / f / "dataset_info.json").read_text())["interval_seconds"])
        a = analyze_traffic(df, baseline_pct=30, alpha=0.001)
        good = (schema == SchemaType.INTERVAL_COUNTS and a.has_labels and a.baseline_count == 3000 and pre.total_intervals == len(df)
                and pre.has_labels and int(pre.dataframe.has_attack_label.sum()) == int(df.has_attack_label.sum()))
        ok &= good
        det.append(f"{f}: schema={schema.name}, baseline={a.baseline_count}, lambda_hat={a.lambda_hat:.2f}, bursts={a.burst_count}, labels_detected={a.has_labels}")
    rec("13. BurstPulse repo compatibility (inspect_schema, prepare_interval_dataset, analyze_traffic @30% baseline)",
        ok, " | ".join(det))


def zip_dir(src_root, zip_path):
    """Deterministic zip: sorted entries, fixed timestamps, DEFLATE."""
    zip_path = Path(zip_path)
    if zip_path.exists():
        zip_path.unlink()
    files = sorted(p for p in Path(src_root).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            arc = f"{TOP}/{p.relative_to(src_root).as_posix()}"
            zi = zipfile.ZipInfo(arc, date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, p.read_bytes())
    return zip_path


def verify_zip(zip_path, label):
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            bad = z.testzip()
            names = sorted(n for n in z.namelist() if not n.endswith("/"))
            z.extractall(tmp)
        base = Path(tmp) / TOP
        expected = sorted(f"{TOP}/{x}" for x in EXPECTED_FILES)
        missing = [n for n in expected if n not in names]
        extra = [n for n in names if n not in expected]
        same = all((base / n.split("/", 1)[1]).read_bytes() == (ROOT / n.split("/", 1)[1]).read_bytes()
                   for n in names if (ROOT / n.split("/", 1)[1]).exists()
                   and n.split("/", 1)[1] != "VALIDATION_REPORT.txt")
        empties = [n for n in names if (base / n.split("/", 1)[1]).stat().st_size == 0]
        rows = {f: len(pd.read_csv(base / f / "network_traffic.csv")) for f in FOLDERS}
        ok = bad is None and not missing and not extra and same and not empties and all(v >= MIN_ROWS for v in rows.values())
        return ok, (f"{label}: corrupt={bad}, missing={missing}, extra={extra}, files_identical_to_source={same}, "
                    f"empty_files={empties}, extracted_rows={rows}")


def write_report(path, zip_detail=None):
    lines = ["BurstPulse synthetic dataset pack - VALIDATION REPORT",
             "=" * 60,
             f"generator_version: {gen.GENERATOR_VERSION}",
             "NOTE: all data is SYNTHETIC; see README.md and DATA_PROVENANCE.json.", ""]
    for name, status, detail in results:
        lines.append(f"[{status}] {name}")
        if detail:
            lines.append(f"        {detail}")
    lines += ["", "Warnings:"] + ([f"  - {w}" for w in warnings] if warnings else ["  (none)"])
    lines += ["", "Dataset row counts: " + ", ".join(
        f"{f}={len(load(f))}" for f in FOLDERS)]
    n_fail = sum(1 for r in results if r[1] == "FAIL")
    lines += ["", f"OVERALL: {'ALL EXECUTED CHECKS PASSED' if n_fail == 0 else str(n_fail) + ' CHECK(S) FAILED'}"]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return n_fail


def main():
    frames, _ = check_dataset_files()
    check_repo_compat(frames)
    # 12: zip creation + extraction on a staging archive (report excluded from the byte-compare by design)
    Path(ROOT / "VALIDATION_REPORT.txt").write_text("staging\n")
    with tempfile.TemporaryDirectory() as tmp:
        stage = zip_dir(ROOT, Path(tmp) / ZIP_NAME)
        ok, detail = verify_zip(stage, "staging archive")
        rec("12. ZIP created and extracted successfully into a temp dir; contents match source", ok,
            detail + f"; staging size={stage.stat().st_size} bytes")
    n_fail = write_report(ROOT / "VALIDATION_REPORT.txt")
    if n_fail:
        print("VALIDATION FAILED"); print((ROOT / "VALIDATION_REPORT.txt").read_text()); sys.exit(1)
    final = zip_dir(ROOT, ROOT.parent / ZIP_NAME)
    ok2, d2 = verify_zip(final, "final archive")
    print((ROOT / "VALIDATION_REPORT.txt").read_text())
    print("FINAL ZIP:", final, final.stat().st_size, "bytes;", d2)
    if not ok2:
        sys.exit(1)


if __name__ == "__main__":
    main()
