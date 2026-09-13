"""
Make every figure-generation source write its EPS natively from matplotlib.

Purpose
    EPS obtained by converting the figure PDF with ghostscript comes out
    rasterised: the file carries only a handful of path operators plus one
    bitmap image operator, so the vector content is gone and the publisher's
    vector-output audit fails. This module therefore guarantees that each
    figure-generation source writes the EPS itself through matplotlib, with
    TrueType font embedding enabled ("pdf.fonttype": 42 and "ps.fonttype": 42),
    which keeps the text as embedded TrueType outlines instead of Type 3 bitmaps.
    It is stage 15 (exports) of the pipeline: it patches the copies of the
    figure sources that the PDF build stages under work.pdfbuild_dir, so that
    the EPS is written in the same rendering pass as the PDF and never passes
    through a converter.

Author
    Tao Zhu

Created
    2026-09-13

Design
    Two text-level edits are applied per source file, in this order:
      1. add the font-type settings to the rcParams block, unless the file
         already mentions "ps.fonttype";
      2. append one fig.savefig(... .eps ...) call after every
         fig.savefig(... .pdf ...) call, with the same indentation, so that both
         vector formats come out of the same figure object.
    Edit 2 is applied to every PDF statement it finds, exactly as the original
    helper did: running this module twice over the same tree therefore adds a
    second EPS call per figure, so it is run once per staged tree. Edit 1 is
    skipped when the file already carries the marker, and a file that needs no
    edit at all is left untouched rather than rewritten.

Inputs
    paths.work.pdfbuild_dir (directory) -- staging directory of the figure and
        PDF build; it holds the figure-generation sources named in
        params.output.figure_sources.
    params.output.figure_sources (list of str) -- basenames of the
        figure-generation sources to patch.
    params.output.raster_dpi (int) -- raster resolution carried by the rcParams
        anchor line of those sources; used to build the exact text matched.

Outputs
    The staged figure sources are rewritten in place when a change is needed.
    No new file is created.

Usage
    python 15_01_export_eps_native.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Allow the module to be run from any directory: src/ holds the common package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import param, work  # noqa: E402

# PostScript/PDF font-type marker for embedded TrueType outlines. The value 42
# is fixed by the file formats themselves (Type 42), so it is a format constant
# rather than a tunable analysis parameter, and it is not read from params.yaml.
FONT_TYPE_TRUE_TYPE = 42

# Lines that already set the PostScript font type are left alone. The marker is
# exactly the setting written below, so a file that was patched in an earlier run
# is not patched a second time.
_FONT_TYPE_MARKER = "ps.fonttype"

# Whole fig.savefig(...) statement that writes the PDF, e.g.
#     fig.savefig(f"{OUT}/Fig1_availability_comparability.pdf", bbox_inches="tight", ...)
_SAVEFIG_PDF_STATEMENT = re.compile(
    r'^[ \t]*fig\.savefig\(f"\{OUT\}/[A-Za-z0-9_]+\.pdf".*\)$', flags=re.M
)

# The figure stem inside that statement; only the extension has to be changed.
_SAVEFIG_PDF_STEM = re.compile(r'f"\{OUT\}/([A-Za-z0-9_]+)\.pdf"')


def add_font_type_settings(source: str, dpi: int, font_type: int) -> str:
    """Add the PDF/PS font-type settings to the rcParams block of a source.

    Embedding TrueType (Type 42) keeps the glyphs as outlines; matplotlib's
    default Type 3 output stores glyphs as bitmaps and is rejected by journals.

    Args
        source: full text of a figure-generation source.
        dpi: raster resolution that the anchor line of that source carries; it
            is used to describe the exact line that is matched.
        font_type: font-type marker to write, see FONT_TYPE_TRUE_TYPE.

    Returns
        The patched text, or ``source`` unchanged when it already sets the font
        type. Every occurrence of the anchor line is treated, as in the original
        helper.
    """
    if _FONT_TYPE_MARKER in source:
        return source
    anchor = (
        f'"figure.dpi": {dpi}, "savefig.dpi": {dpi}, '
        f'"axes.unicode_minus": False,'
    )
    replacement = (
        anchor
        + "\n    "
        + f'"pdf.fonttype": {font_type}, "ps.fonttype": {font_type},'
    )
    return source.replace(anchor, replacement)


def _append_native_eps_call(match: re.Match) -> str:
    """Return a PDF savefig statement followed by the matching EPS statement.

    Args
        match: a match of _SAVEFIG_PDF_STATEMENT.

    Returns
        The original statement plus one fig.savefig call for the ``.eps`` file of
        the same stem, indented like the statement. The ``{OUT}`` placeholders
        stay literal because they are resolved when the patched source runs.
    """
    line = match.group(0)
    found = _SAVEFIG_PDF_STEM.search(line)
    if not found:  # a statement whose path is not a f"{OUT}/<stem>.pdf" literal
        return line
    stem = found.group(1)
    indent = line[: len(line) - len(line.lstrip())]
    return (
        line
        + f'\n{indent}fig.savefig(f"{{OUT}}/{stem}.eps", '
        f'bbox_inches="tight", facecolor="white")'
    )


def patch_source(path: Path, dpi: int) -> bool:
    """Apply both edits to one figure-generation source.

    Args
        path: source file to patch, rewritten in place.
        dpi: raster resolution used to locate the rcParams anchor line.

    Returns
        True when the file changed. False when neither edit had anything to do
        (no rcParams anchor line and no PDF statement); such a file is left
        untouched instead of being rewritten with identical bytes.
    """
    original = path.read_text(encoding="utf-8")
    patched = add_font_type_settings(original, dpi, FONT_TYPE_TRUE_TYPE)
    patched = _SAVEFIG_PDF_STATEMENT.sub(_append_native_eps_call, patched)
    if patched == original:
        return False
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    """Patch every staged figure source listed in the configuration."""
    stage_dir = work("pdfbuild_dir")
    sources = param("output", "figure_sources")
    dpi = param("output", "raster_dpi")

    print(f"Native EPS export: {len(sources)} figure source(s) staged in {stage_dir}")
    for name in sources:
        changed = patch_source(stage_dir / name, dpi)
        print(f"  {name}: {'updated' if changed else 'unchanged'}")
    print(
        "EPS is written by matplotlib itself with Type 42 fonts; "
        "ghostscript conversion, which would rasterise the figure, is not used."
    )


if __name__ == "__main__":
    main()
