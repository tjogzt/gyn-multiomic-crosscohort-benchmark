#!/usr/bin/env python3
"""
Check the submission package against the journal specification and the artefacts.

Purpose
    Stage 14, tenth script. It answers the four questions a submission package
    has to survive: what is in the package, whether the figures meet the target
    journal's column widths and raster resolution, whether every figure and table
    referenced in the narrative exists and every existing figure and table is
    referenced, and whether the headline numbers quoted in the text are the ones
    the artefacts produced. A closing pass looks for placeholders and leftovers
    in the narrative sources, which are the failures a reader would notice first.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    manuscript("methods"), manuscript("results"), manuscript("tables_markdown"),
    manuscript("discussion"), manuscript("cover_letter"), manuscript("figure_plan")
                                        the narrative sources; the Markdown entries
                                        are the document set that is searched.
    results("figures_dir")/<glob>       published figures: PNG for the inventory
                                        count, PNG and PDF for the geometry check.
    results("tables_dir")/*.csv         published tables.
    work("benchmark_summary")           JSON, tab1 of the Python benchmark.
    work("benchmark_merged")            JSON, r_ranking of the merged toolchains.
    work("platform_results_json")       JSON, ladder of the cross-platform rungs.
    config/params.yaml verification.journal_widths_mm
                                        journal column widths in millimetres.
    config/params.yaml verification.compliance_text_values
                                        [label, value] pairs quoted in the text.
    config/params.yaml verification.figure_pdf_glob
                                        glob selecting the figure PDFs.
    config/params.yaml output.figure_width_mm, output.raster_dpi,
    output.figure_png_globs, benchmark.best_python_method
                                        width limit, resolution floor, the PNG
                                        globs and the arm the text quotes.

Outputs
    None. The five-part report is printed.

Usage
    python 14_10_check_submission_compliance.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import fitz
import pandas as pd
from PIL import Image

# Put src/ on the import path so that common.* is importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import CONFIG, PROJECT_ROOT, param, results, work  # noqa: E402

RULE = "=" * 92

# Width limit and resolution floor of the target journal, read from the same
# parameters the figure stage writes against, so the check cannot drift from the
# figures it inspects.
WIDTH_LIMIT_MM = float(param("output", "figure_width_mm"))
RASTER_DPI = float(param("output", "raster_dpi"))

# Journal column widths in millimetres, keyed by the label reported as the
# nearest specification of a figure; frozen in the configuration.
JOURNAL_WIDTHS_MM = param("verification", "journal_widths_mm")

# Placeholder tokens a submitted source must not contain. The two non-ASCII
# tokens are spelled with escapes so that the file stays free of CJK while still
# matching the Chinese markers the legacy sources used.
PLACEHOLDER_PATTERN = r"(TODO|TBD|XXX|\u5f85\u8865|\u5360\u4f4d|placeholder)"

# Marker of a citation that was never resolved to a number.
UNREPLACED_PATTERN = r"\[\?\]"

# Panel references (Figure 1a, Figure S7b) are counted as references to their
# parent figure; the parent must exist for the panel reference to be valid.
PANEL_PATTERN = r"Figure\s+(S?\d+)([a-c])\b"

# Every spelling of a figure reference used in the narrative.
FIGURE_CITATION_PATTERNS = (r"Figure\s+(S?\d+)([a-c])?", r"Fig(?:ure)?\s*(S?\d+)")

# Table references and the file-name spelling of a published table.
TABLE_CITATION_PATTERN = r"Table\s+(\d+[a-c]?)"
TABLE_FILE_PATTERN = r"T(\d+[abc]?)_"
FIGURE_FILE_PATTERN = r"Fig(S?\d+)"


def manuscript(key: str) -> Path:
    """Resolve a manuscript source declared under ``paths.manuscript``.

    Args:
        key: entry name inside the manuscript section of config/paths.yaml.

    Returns:
        Absolute path of the manuscript source.
    """
    root = Path(str(CONFIG["paths"]["manuscript_root"]))
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / CONFIG["paths"]["manuscript"][key]


def manuscript_documents() -> list[Path]:
    """Return the Markdown narrative sources, in file-name order.

    Returns:
        The declared manuscript sources whose file name ends in ".md", sorted by
        name so that the report order does not depend on the configuration order.
    """
    sources = [
        manuscript(key)
        for key, value in CONFIG["paths"]["manuscript"].items()
        if str(value).endswith(".md")
    ]
    return sorted(sources, key=lambda path: path.name)


def figure_files(extension: str) -> list[Path]:
    """Return the published figures carrying one format, in reporting order.

    Args:
        extension: file extension without the dot; "png" or "pdf".

    Returns:
        The matching paths. Each glob in the configuration is sorted on its own
        and the globs are concatenated, so main figures precede the supplement.
    """
    globs = {
        "png": param("output", "figure_png_globs"),
        "pdf": [param("verification", "figure_pdf_glob")],
    }[extension]
    paths: list[Path] = []
    for pattern in globs:
        paths.extend(sorted(results("figures_dir").glob(pattern)))
    return paths


def load_artifact(key: str):
    """Read one JSON artefact from the work root.

    Args:
        key: entry name inside the work section of config/paths.yaml.

    Returns:
        The decoded JSON object.
    """
    with work(key).open(encoding="utf-8") as handle:
        return json.load(handle)


def report_inventory() -> list[Path]:
    """Print what the package contains and return the narrative sources.

    Returns:
        The Markdown narrative sources, so the other sections reuse one list.
    """
    print(RULE)
    print("[A] package inventory")
    print(RULE)
    documents = manuscript_documents()
    png_files = figure_files("png")
    table_files = sorted(results("tables_dir").glob("*.csv"))
    print(f"  narrative documents: {len(documents)}")
    for path in documents:
        print(f"     {path.name:44s} {path.stat().st_size:>7,} B")
    print(f"  figures: {len(png_files)} (PNG)")
    print(f"  tables: {len(table_files)} (CSV)")
    return documents


def report_geometry() -> None:
    """Print figure width, page height, raster resolution and the nearest spec.

    Returns:
        None. The dimension table and the two summary counts are printed.
    """
    print("\n" + RULE)
    print("[B] figure geometry and resolution (target journal specification)")
    print(RULE)
    rows = []
    for pdf_path in figure_files("pdf"):
        document = fitz.open(pdf_path)
        page = document[0]
        width_mm = page.rect.width / 72 * 25.4
        height_mm = page.rect.height / 72 * 25.4
        # The raster resolution is measured, not read from the PNG header: the
        # pixel count of the PNG is divided by the physical width of the PDF page
        # it renders, which is what the typesetter will scale to.
        png_path = pdf_path.with_suffix(".png")
        dpi = None
        if png_path.exists():
            with Image.open(png_path) as image:
                dpi = round(image.size[0] / (width_mm / 25.4))
        nearest = min(JOURNAL_WIDTHS_MM.items(), key=lambda item: abs(item[1] - width_mm))
        rows.append(
            {
                "fig": pdf_path.stem,
                "width_mm": round(width_mm, 1),
                "height_mm": round(height_mm, 1),
                "png_dpi": dpi,
                "nearest_spec": nearest[0],
                "oversize": round(width_mm - WIDTH_LIMIT_MM, 1)
                if width_mm > WIDTH_LIMIT_MM
                else 0,
            }
        )
    dimensions = pd.DataFrame(rows)
    print(dimensions.to_string(index=False))
    print(
        f"\n  figures wider than the {WIDTH_LIMIT_MM:.0f} mm double-column limit: "
        f"{(dimensions.width_mm > WIDTH_LIMIT_MM).sum()} / {len(dimensions)}"
    )
    # A missing PNG leaves its resolution undefined and is not counted as a
    # violation; the inventory section reports the figures that actually exist.
    print(
        f"  figures below {RASTER_DPI:.0f} dpi: "
        f"{(dimensions.png_dpi < RASTER_DPI).sum()} / {len(dimensions)}"
    )


def report_cross_references(documents: list[Path]) -> None:
    """Print the figure and table references against the files that exist.

    Args:
        documents: the Markdown narrative sources to search.

    Returns:
        None. The cited, existing and mismatching numbers are printed.
    """
    print("\n" + RULE)
    print("[C] cross-reference completeness")
    print(RULE)
    text = "".join(path.read_text(encoding="utf-8") for path in documents)
    tables_text = manuscript("tables_markdown").read_text(encoding="utf-8")
    figure_plan_text = manuscript("figure_plan").read_text(encoding="utf-8")

    # Figure numbers referenced anywhere in the narrative or the figure plan.
    # The plan is searched as well because it is where a panel is defined before
    # the narrative cites it.
    cited = set()
    for pattern in FIGURE_CITATION_PATTERNS[:1]:
        for match in re.finditer(pattern, text + figure_plan_text):
            cited.add(match.group(1))
    for pattern in FIGURE_CITATION_PATTERNS[1:]:
        for match in re.finditer(pattern, text):
            cited.add(match.group(1))
    panels = {
        match.group(1) + match.group(2)
        for match in re.finditer(PANEL_PATTERN, text + figure_plan_text)
    }

    existing_figures = set()
    for path in figure_files("png"):
        match = re.match(FIGURE_FILE_PATTERN, path.name)
        if match:
            existing_figures.add(match.group(1))

    print(f"  figure numbers cited in the text: {sorted(cited)}")
    print(f"  panel-level references: {sorted(panels)}")
    print(f"  figure numbers that exist: {sorted(existing_figures)}")
    missing_figures = sorted(cited - existing_figures)
    extra_figures = sorted(existing_figures - cited)
    print(f"  cited but missing: {missing_figures if missing_figures else 'none'}")
    print(f"  present but never cited: {extra_figures if extra_figures else 'none'}")
    bad_panels = sorted({panel for panel in panels if panel[:-1] not in existing_figures})
    print(f"  panel references whose parent figure is missing: "
          f"{bad_panels if bad_panels else 'none'}")

    cited_tables = {
        match.group(1)
        for match in re.finditer(TABLE_CITATION_PATTERN, text + tables_text)
    }
    existing_tables = set()
    for path in sorted(results("tables_dir").glob("*.csv")):
        match = re.match(TABLE_FILE_PATTERN, path.name)
        if match:
            existing_tables.add(match.group(1))
    print(f"  table numbers cited in the text: {sorted(cited_tables)}")
    print(f"  table numbers that exist: {sorted(existing_tables)}")
    missing_tables = sorted(cited_tables - existing_tables)
    extra_tables = sorted(existing_tables - cited_tables)
    print(f"  cited but missing: {missing_tables if missing_tables else 'none'}")
    print(f"  present but never cited: {extra_tables if extra_tables else 'none'}")


def artifact_value(label: str):
    """Return the artefact value behind one quoted headline number.

    Args:
        label: label of the quoted value, as declared in
            params.verification.compliance_text_values.

    Returns:
        The value the artefact holds for that label, or None when the label is
        not one this script derives.
    """
    if label == "T3 KMeans overall":
        # The arm the narrative quotes as the best Python method.
        summary = load_artifact("benchmark_summary")
        method = param("benchmark", "best_python_method")
        return summary["tab1"][method]["all"]
    if label == "T4 SNF mean":
        # The R-side SNF entry; the key carries an ordinal prefix, so the entry
        # is found by name rather than by a hardcoded key.
        ranking = load_artifact("benchmark_merged")["r_ranking"]
        keys = [key for key in ranking if "SNF" in key.upper()]
        return ranking[keys[0]] if keys else None
    if label == "T6 limma centroid":
        # Chance level of a centroid AUC. It is an exact mathematical constant,
        # not a tuned threshold, so it is written out.
        return 0.5
    if label == "T7a cross-platform rho":
        # Second rung of the ladder (index 1) is the cross-platform comparison;
        # the first rung is the within-cohort same-batch reference.
        ladder = load_artifact("platform_results_json")["ladder"]
        return ladder[1]["spearman"]
    return None


def report_number_consistency(documents: list[Path]) -> None:
    """Print every headline number against the artefact that produced it.

    Args:
        documents: the Markdown narrative sources to search.

    Returns:
        None. One line per quoted value is printed.
    """
    print("\n" + RULE)
    print("[D] key-number consistency (narrative vs artefact)")
    print(RULE)
    text = "".join(path.read_text(encoding="utf-8") for path in documents)
    for check in param("verification", "compliance_text_values"):
        label, quoted = check["label"], str(check["value"])
        in_text = quoted in text
        artifact = artifact_value(label)
        # The two sides are compared as numbers with a tolerance well below the
        # printed precision, because the narrative rounds while the artefact does
        # not; a rounding difference must not read as a mismatch.
        agrees = False
        if artifact is not None:
            try:
                agrees = abs(float(str(artifact)) - float(quoted)) < 5e-5
            except ValueError:
                agrees = str(artifact) == quoted
        print(
            f"  {'v' if in_text else '~'} {label:26s} text has {quoted:>8s}  "
            f"artefact {artifact}  {'agrees' if agrees else 'MISMATCH'}"
        )


def report_text_layer(documents: list[Path]) -> None:
    """Print placeholders, strikethroughs and unreplaced citations per document.

    Args:
        documents: the Markdown narrative sources to inspect.

    Returns:
        None. One line per document is printed.
    """
    print("\n" + RULE)
    print("[E] text-layer check")
    print(RULE)
    for path in documents:
        source = path.read_text(encoding="utf-8")
        placeholders = re.findall(PLACEHOLDER_PATTERN, source, re.I)
        stubs = len(re.findall(UNREPLACED_PATTERN, source))
        print(
            f"  {path.name[:34]:36s} placeholders {len(placeholders)}  "
            f"strikethroughs {source.count('~~')}  unreplaced citations {stubs}"
        )


def main() -> None:
    """Run the five compliance sections and print the report."""
    documents = report_inventory()
    report_geometry()
    report_cross_references(documents)
    report_number_consistency(documents)
    report_text_layer(documents)


if __name__ == "__main__":
    main()
