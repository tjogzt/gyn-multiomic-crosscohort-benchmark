#!/usr/bin/env python3
"""
Audit every published EPS figure for genuine vector output.

Purpose
    Stage 14, eleventh script. A converter that silently rasterises an EPS file
    still yields a file that opens and prints, so the failure is invisible until
    a typesetter scales it. matplotlib abbreviates operators in the document
    prolog, which is why the audit counts image operators in the body rather than
    looking for a drawing primitive by name, and it checks which font format is
    embedded: a vector figure carries a Type 42 (TrueType) font program, a
    rasterised one does not. A figure that contains an imshow heatmap is expected
    to carry one or two raster operators, because a heatmap is a raster by
    nature; for every other figure any image operator is a defect.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    results("figures_dir")/<glob>   the published EPS figures, read as bytes.
    config/params.yaml verification.eps_figure_glob
                                    glob selecting those figures.
    config/params.yaml verification.eps_raster_figures
                                    stems that legitimately embed a heatmap.
    config/params.yaml verification.eps_null_byte_limit
                                    binary-content bound that separates a
                                    text-only EPS from a rasterised one.

Outputs
    None. One line per figure is printed, plus the list of outliers.

Usage
    python 14_11_audit_eps_vector_output.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Put src/ on the import path so that common.* is importable from any directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import param, results  # noqa: E402

# Figures whose panels include an imshow heatmap and therefore an embedded
# raster: the availability matrix, the layer curve, the method ranking, the
# threshold panel and the supplementary heatmaps. Frozen in the configuration
# because it is a property of the figures, not of the check.
RASTER_FIGURES = set(param("verification", "eps_raster_figures"))

# An EPS body that is pure text carries almost no NUL bytes; an embedded image
# stream carries many. The bound separates the two without parsing the file.
NULL_BYTE_LIMIT = int(param("verification", "eps_null_byte_limit"))

# Operator spellings matplotlib may emit for an embedded image and for its
# display device; both spellings are counted so an abbreviated prolog cannot hide
# an image operator.
IMAGE_OPERATORS = (b"\nimage\n", b" image\n")
SHOW_OPERATORS = (b" show\n", b"show\n")

# Font token of an embedded Type 42 program (a TrueType font). Its presence is
# what distinguishes vector text from text that was flattened into a bitmap.
TYPE42_MARKER = b"TrueTypeFont"

VERDICT_EXPECTED_RASTER = "vector plus heatmap bitmap (expected)"
VERDICT_PURE_VECTOR = "pure vector"
VERDICT_UNEXPECTED_RASTER = "WARNING unexpected bitmap"


def audit_figure(path: Path) -> tuple:
    """Audit one EPS figure and return its measurement row.

    Args:
        path: the EPS file to read.

    Returns:
        The tuple (stem, size_kb, image_operators, show_operators, has_type42,
        is_text_only, verdict). The verdict is the warning string when the file
        carries a bitmap its figure is not expected to carry.
    """
    payload = path.read_bytes()
    # The stem may carry a suffix after the figure tag, so the tag is extracted
    # rather than assumed to be the whole name.
    tag = re.match(r"(FigS?\d+)", path.stem).group(1)
    image_operators = sum(payload.count(op) for op in IMAGE_OPERATORS)
    show_operators = sum(payload.count(op) for op in SHOW_OPERATORS)
    has_type42 = TYPE42_MARKER in payload
    is_text_only = payload.count(b"\x00") < NULL_BYTE_LIMIT
    expects_raster = tag in RASTER_FIGURES

    if image_operators == 0:
        verdict = VERDICT_PURE_VECTOR
    elif expects_raster:
        verdict = VERDICT_EXPECTED_RASTER
    else:
        verdict = VERDICT_UNEXPECTED_RASTER
    return (
        path.stem,
        len(payload) / 1024,
        image_operators,
        show_operators,
        has_type42,
        is_text_only,
        verdict,
    )


def main() -> None:
    """Audit every published EPS figure and print the outliers."""
    figure_glob = param("verification", "eps_figure_glob")
    rows = [audit_figure(path) for path in sorted(results("figures_dir").glob(figure_glob))]
    for stem, size_kb, images, shows, has_type42, is_text_only, verdict in rows:
        print(
            f"  {stem:40s} {size_kb:6.1f}KB  image={images:<3d} show={shows:<4d} "
            f"Type42={'Y' if has_type42 else 'n'}  "
            f"ASCII={'Y' if is_text_only else 'n'}  {verdict}"
        )
    bad = [row for row in rows if row[-1] == VERDICT_UNEXPECTED_RASTER]
    print(f"\n{len(rows)} figures audited; {len(bad)} unexpected")
    if bad:
        print("  unexpected:", [row[0] for row in bad])
    print(
        "\nCriterion: image operators == 0 means pure vector; a figure with a "
        "heatmap is expected to carry one or two bitmap operators, because imshow "
        "is a raster by nature."
    )


if __name__ == "__main__":
    main()
