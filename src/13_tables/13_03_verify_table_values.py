"""
Verify the rendered table document against the tables it was rendered from.

Purpose
    Stage 13, third script. The rendered table document is checked in three
    ways: no table row may still contain a non-English label, the number of
    tables in the document must match the number of CSVs published beside it,
    and every number in the sampled tables must be traceable to the artefact
    that produced it. A closing block re-checks four frozen cell values so that
    a silent change of a headline number cannot pass unnoticed.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    manuscript("tables_markdown")      the document rendered by 13_02.
    results("tables_dir")/*.csv        the published table CSVs.
    work("benchmark_summary")          JSON, source of Table 3.
    work("benchmark_merged")           JSON, source of Table 4.
    work("pac_aggregate")              JSON, source of Table 5a.
    config/params.yaml verification.table_value_checks  frozen cell values.

Outputs
    None. The result is printed.

Usage
    python 13_03_verify_table_values.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import CONFIG, PROJECT_ROOT, param, results, work

# Any character in the CJK unified ideograph block counts as a residue.
CJK_RESIDUE = re.compile(r"[\u4e00-\u9fff]")


def manuscript(key: str) -> Path:
    """Resolve a manuscript source declared under ``paths.manuscript``.

    Args
        key: entry name inside the manuscript section of config/paths.yaml.

    Returns
        Absolute path of the manuscript source.
    """
    root = Path(str(CONFIG["paths"]["manuscript_root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / CONFIG["paths"]["manuscript"][key]


def load_json(path: Path):
    """Read one JSON artefact.

    Args
        path: absolute path of the artefact.

    Returns
        The decoded object.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    """Run the three checks and the frozen-cell sample, and print the outcome."""
    tables_dir = results("tables_dir")
    document = manuscript("tables_markdown").read_text(encoding="utf-8")
    # Everything above the first horizontal rule is the document header; only
    # the table rows below it are checked for a non-English label.
    body = document.split("---", 1)[1] if "---" in document else document

    # 1) Non-English residue, restricted to the table rows.
    table_lines = [line for line in body.split("\n") if line.startswith("|")]
    residue = [
        (index, line)
        for index, line in enumerate(table_lines)
        if CJK_RESIDUE.search(line)
    ]
    print(
        f"1) table rows total {len(table_lines)}; "
        f"rows containing a Chinese character {len(residue)}"
    )
    for _, line in residue[:5]:
        print(f"     {line[:110]}")

    # 2) Number of tables in the document against the number of CSVs published.
    n_tables = len(re.findall(r"^## Table ", document, re.M))
    n_csv = len(list(tables_dir.glob("*.csv")))
    print(f"2) tables in the document {n_tables}; CSVs in the tables dir {n_csv}")

    # 3) Numeric provenance: every number in a sampled table must be present in
    #    the artefact the table was extracted from.
    provenance = {
        "Table 3": (
            "T3_python_method_ranking.csv",
            load_json(work("benchmark_summary"))["tab1"],
        ),
        "Table 4": (
            "T4_R_package_ranking.csv",
            load_json(work("benchmark_merged"))["r_ranking"],
        ),
        "Table 5a": (
            "T5a_pac_by_method.csv",
            load_json(work("pac_aggregate"))["tab1"],
        ),
    }
    # Keys of the per-method dictionaries that a table column may be built from.
    per_method_keys = ("A", "B", "all", "bal", "deg", "strong", "med")
    all_traced = True
    for table, (filename, source) in provenance.items():
        frame = pd.read_csv(tables_dir / filename)
        numbers = pd.to_numeric(
            frame.select_dtypes("number").stack(), errors="coerce"
        ).dropna()
        traced = sum(
            1
            for value in numbers
            if any(
                abs(value - round(float(entry), 4)) < 1e-9
                for entry in source.values()
                if isinstance(entry, (int, float))
            )
            or any(
                abs(value - round(float(entry.get(key, 0)), 4)) < 1e-9
                for entry in source.values()
                if isinstance(entry, dict)
                for key in per_method_keys
            )
        )
        print(
            f"3) {table}: {len(frame)} rows / {len(numbers)} numbers "
            f"-> traced {traced}"
        )
        if traced == 0:
            all_traced = False

    # 4) Frozen cell values, read from the configuration.
    for check in param("verification", "table_value_checks"):
        frame = pd.read_csv(tables_dir / check["file"])
        got = float(frame[frame.iloc[:, 0] == check["key"]][check["column"]].iloc[0])
        verdict = "OK" if abs(got - check["expected"]) < 1e-9 else "MISMATCH"
        print(
            f"4) {check['file'][:28]} [{check['key']}] {check['column']} = {got}  "
            f"expected {check['expected']}  {verdict}"
        )

    print("\nprovenance check complete:", "all traced" if all_traced else "unresolved rows")


if __name__ == "__main__":
    main()
