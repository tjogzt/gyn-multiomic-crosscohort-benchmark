"""
Re-export every published figure PNG as an LZW-compressed TIFF at 300 dpi.

Purpose
    The publisher's figure system accepts TIFF rasters at a stated resolution and
    rejects figures whose resolution tag is missing, while the pipeline itself
    renders PNG for preview. This module is stage 15 (exports) of the pipeline:
    it reads the PNG of every published figure from results.figures_dir, encodes
    it losslessly as an LZW TIFF beside it, and stamps the configured resolution
    into the TIFF so that the physical print width of each figure is unambiguous.
    Pixels are never resampled: the resolution tag records the resolution at
    which the figure was rendered, so the physical width in mm is derived from
    the pixel count rather than imposed on it.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    results.figures_dir (directory) -- published figure directory; holds the PNG
        rasters written by the figure stage.
    params.output.figure_png_globs (list of str) -- glob patterns, relative to
        that directory, selecting the PNG rasters to convert. Every pattern is
        sorted on its own, so the reporting order is the order of the list.
    params.output.raster_dpi (int) -- resolution stamped into the TIFF and used
        to convert pixels into millimetres.

Outputs
    <figures_dir>/<stem>.tif (TIFF, RGB, LZW, resolution raster_dpi) for every
    PNG selected by the patterns.

Usage
    python 15_02_export_tiff_300dpi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# Allow the module to be run from any directory: src/ holds the common package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import ensure_dir, param, results  # noqa: E402

# Millimetres per inch, used to state the print width of a raster in the
# reporting line; a unit conversion, not a tunable parameter.
MM_PER_INCH = 25.4

# Pillow's identifier for LZW compression, which is lossless. The publisher's
# figure system expects LZW-compressed TIFF, so it is fixed here.
TIFF_COMPRESSION = "tiff_lzw"


def export_tiff(png_path: Path, dpi: int) -> tuple[int, int | None, float]:
    """Encode one PNG raster as an LZW TIFF written beside it.

    Args
        png_path: source PNG. The TIFF is written to the same directory with the
            same stem, so a figure's raster formats sit together.
        dpi: resolution written into the TIFF metadata and used to derive the
            physical print width.

    Returns
        ``(size in bytes, resolution read back from the TIFF or None when the
        tag is absent, physical width in mm)``; the value read back is taken from
        the written file, so a silently dropped resolution tag is visible in the
        report rather than assumed.
    """
    tif_path = png_path.with_suffix(".tif")
    with Image.open(png_path) as im:
        width_mm = im.size[0] / dpi * MM_PER_INCH
        # Convert to RGB because the TIFF required here carries no alpha
        # channel; the figures are rendered opaque on white, so no visible
        # content is lost, and the pixels are not resampled.
        im.convert("RGB").save(
            tif_path,
            format="TIFF",
            compression=TIFF_COMPRESSION,
            dpi=(dpi, dpi),
        )
    with Image.open(tif_path) as written:
        tag = written.info.get("dpi", (None,))[0]
        read_back = round(float(tag)) if tag is not None else None
    return tif_path.stat().st_size, read_back, width_mm


def main() -> None:
    """Convert every figure PNG selected by the configuration into a TIFF."""
    fig_dir = ensure_dir(results("figures_dir"))
    dpi = param("output", "raster_dpi")
    globs = param("output", "figure_png_globs")

    # Each pattern is sorted separately so that main figures are reported before
    # the supplement, matching the order in which the patterns are configured.
    pngs: list[Path] = []
    for pattern in globs:
        pngs.extend(sorted(fig_dir.glob(pattern)))

    print(f"{'stem':40s} {'TIF':>9s}  dpi  width_mm")
    for png in pngs:
        size_bytes, read_back, width_mm = export_tiff(png, dpi)
        print(f"{png.stem:40s} {size_bytes/1024:8.0f}K {read_back:>4} {width_mm:.1f}")
    print(f"\nTIFF written: {len(pngs)}")


if __name__ == "__main__":
    main()
