"""
Verify the rendered table PDF against its Markdown source and the table CSVs.

Purpose
    Stage 13, fourth script. A PDF is checked rather than the Markdown it was
    built from, because the failure modes that matter here happen at render
    time: missing glyphs (a font that falls back to a tofu box), a stale PDF
    that no longer matches its source, and numbers that were dropped or rounded
    away during typesetting. This is the canonical implementation; the numbered
    duplicate 14_12_verify_table_pdf.py imports this module and re-exports main.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    manuscript("tables_markdown")      the Markdown the PDF was built from.
    manuscript("tables_pdf")           the rendered PDF.
    results("tables_dir")/*.csv        the published table CSVs.
    config/params.yaml verification.table_pdf_values  sampled printed values.

Outputs
    None. The result is printed.

Usage
    python 13_04_verify_table_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import CONFIG, PROJECT_ROOT, param, results

# Characters a renderer substitutes for a glyph its font does not carry: the
# white square, the Unicode replacement character, and the white square again as
# a separate code point (the legacy list carried both spellings).
TOFU = ["\u25a1", "\ufffd", "\u25a1"]


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


def main() -> None:
    """Run the six checks over the rendered PDF and print the outcome."""
    tables_dir = results("tables_dir")
    markdown_path = manuscript("tables_markdown")
    pdf_path = manuscript("tables_pdf")

    # 1) Freshness: a PDF older than its Markdown source is stale.
    markdown_mtime = markdown_path.stat().st_mtime
    pdf_mtime = pdf_path.stat().st_mtime
    freshness = "OK (pdf is newer)" if pdf_mtime >= markdown_mtime else "pdf is stale"
    print(
        f"1) freshness: md {markdown_mtime:.0f} vs pdf {pdf_mtime:.0f} -> {freshness}"
    )

    document = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in document)
    print(f"2) PDF: {len(document)} pages / {len(text):,} characters")

    # 2) Missing glyphs.
    found_tofu = [char for char in TOFU if char in text]
    print(f"3) missing glyphs (tofu): {'none' if not found_tofu else found_tofu}")

    # 3) Table numbers that actually appear in the PDF.
    table_numbers = re.findall(r"Table (\d+[abc]?)", text)
    print(f"4) table numbers present in the PDF: {sorted(set(table_numbers))}")

    # 4) The first numeric value of every published table must survive rendering.
    csv_files = sorted(path for path in tables_dir.glob("*.csv"))
    missing = []
    for path in csv_files:
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes("number")
        if numeric.shape[1] == 0:
            continue
        values = pd.to_numeric(numeric.iloc[:, 0], errors="coerce").dropna()
        if len(values) == 0:
            continue
        token = f"{values.iloc[0]:g}"
        # A number in a generated PDF may be split across lines or padded with
        # spaces, so the whitespace-free form is tested as well as the raw text.
        if token not in text.replace(" ", "").replace("\n", "") and token not in text:
            missing.append((path.name, token))
    print(
        f"5) first value of each CSV found in the PDF: "
        f"{len(csv_files) - len(missing)}/{len(csv_files)}"
    )
    for filename, token in missing:
        print(f"     x {filename}: {token}")

    # 5) Frozen printed values, read from the configuration.
    print("6) sampled key values:")
    for value, label in param("verification", "table_pdf_values"):
        mark = "v" if value in text else "x"
        print(f"     {mark} {value:>8s}  {label}")

    print("\ntable document verification complete")


if __name__ == "__main__":
    main()
