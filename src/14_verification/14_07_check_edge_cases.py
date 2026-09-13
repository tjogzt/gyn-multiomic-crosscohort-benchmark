"""
Check that the boundary labels of Figures 1 to 3 survive PDF rendering.

Purpose
    Stage 14, seventh script. A figure is saved with a tight bounding box, which
    lets the canvas grow to hold every label; whether that worked can only be
    told from the rendered PDF, not from the plotting code. The script opens each
    of the first three figure PDFs, confirms that the boundary tokens the figures
    must carry are actually in the extracted text, and flags any text block that
    ends up flush against a page edge, which is the signature of a label that was
    clipped by the canvas.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    results("figures_dir")/<stem>.pdf       rendered PDF of each figure, one per
                                            stem in verification.figure_1_3_stems.
    config/params.yaml verification.figure_1_3_tokens
                                            boundary tokens that must be present.
    config/params.yaml verification.figure_1_3_stems
                                            file-name stems of those figures.

Outputs
    None. The per-figure report is printed.

Usage
    python 14_07_check_edge_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import param, results

# Distance in points below which a text block counts as touching a page edge. It
# absorbs the rounding of the page box, not a real margin.
EDGE_TOLERANCE_PT = 1.0

# Number of characters of the offending block echoed next to the warning.
BLOCK_ECHO_LENGTH = 40


def check_figure(stem: str, tokens: list[str]) -> None:
    """Report one figure PDF: page geometry, boundary tokens and edge blocks.

    Args
        stem: file-name stem of the figure; the PDF is <stem>.pdf inside the
            results figures directory.
        tokens: boundary tokens the figure must contain.

    Returns
        None. The report for this figure is printed.
    """
    path = results("figures_dir") / f"{stem}.pdf"
    document = fitz.open(path)
    text = " ".join(page.get_text() for page in document)
    # All pages are compared against the first page box, because a fraction
    # figure is sized once and every page shares that geometry.
    first_page = document[0].rect
    print(
        f"\u2500\u2500 {stem}.pdf  {len(document)} pages  "
        f"page {first_page.width:.0f}x{first_page.height:.0f} pt"
    )

    # 1) Every boundary token must be present in the extracted text.
    for token in tokens:
        mark = "\u2713" if token in text else "\u2717"
        print(f"     {mark} {token}")

    # 2) Any text block flush against the page edge is evidence of clipping.
    for page_index, page in enumerate(document):
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            if (
                x1 > first_page.width - EDGE_TOLERANCE_PT
                or y1 > first_page.height - EDGE_TOLERANCE_PT
                or x0 < EDGE_TOLERANCE_PT
                or y0 < EDGE_TOLERANCE_PT
            ):
                print(
                    f"     ! page {page_index} text block touches the edge "
                    f"x0={x0:.1f} y0={y0:.1f} x1={x1:.1f} y1={y1:.1f} : "
                    f"{block[4][:BLOCK_ECHO_LENGTH]!r}"
                )
    print()


def main() -> None:
    """Run the edge and boundary-token check over each figure of Figures 1 to 3."""
    tokens = param("verification", "figure_1_3_tokens")
    for stem in param("verification", "figure_1_3_stems"):
        check_figure(stem, tokens)


if __name__ == "__main__":
    main()
