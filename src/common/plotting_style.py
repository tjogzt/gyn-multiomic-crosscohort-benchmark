#!/usr/bin/env python3
"""
Shared plotting style, semantic colour palette and figure geometry harness.

Purpose
    Hold the single definition of the published figure style and of the
    geometry check that every rendered figure has to pass. All figure scripts of
    stage 12 import their colours, type sizes, label vocabularies and save
    routine from here, so a style change is made once and cannot drift between
    figures. Colour is never decorative: each palette entry is bound to a
    toolchain or to a verdict and keeps that meaning in every figure, which is
    why the palette lives with the style rather than in the configuration.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    None. The three style settings that the target journal constrains
    (output.min_font_pt, output.raster_dpi, output.figure_formats) are read from
    config/params.yaml through common.config.param, so each of them is stated in
    exactly one place and can be changed without touching a figure script.

Outputs
    None, except save(), which writes one file per requested format into the
    directory returned by results("figures_dir").

Usage
    from common.plotting_style import C_PY, check, save
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# src/common/plotting_style.py sits one level below src/; putting src/ on the
# import path lets this module load common.config from any working directory.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.config import ensure_dir, param, results  # noqa: E402

import matplotlib  # noqa: E402

# Figures are written to disk and never displayed, so a file-only backend is
# selected before pyplot is imported anywhere.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# --- Semantic colour palette -------------------------------------------------
# These six colours encode meaning, not aesthetics, and are therefore fixed
# constants of the style rather than configuration: the same colour must carry
# the same meaning in every figure of the paper.
C_PY = "#3D6BA8"    # azurite: Python self-implemented toolchain
C_R = "#C23531"     # cinnabar: R official-package toolchain
C_NEU = "#177CB0"   # indigo: neutral / reference
C_BAD = "#9D2933"   # rouge: failed / degenerate / undecidable
C_GY = "#999999"    # grey: auxiliary (raw values, no correction applied)
C_OK = "#3F6B3F"    # pine: passing / usable range
C_SPAN = "#E8F0E8"  # pale wash used behind "usable" intervals

# --- Type sizes and line weights ---------------------------------------------
# Fixed style constants. The enforced floor on published type size is
# output.min_font_pt (see MIN_FONT_PT below); every size used here is at or
# above it.
FONT_SIZE_PT = 8.5
TICK_SIZE_PT = 8
TITLE_SIZE_PT = 9
LEGEND_SIZE_PT = 8
AXIS_LINEWIDTH = 0.7

# --- Settings taken from the configuration ----------------------------------
MIN_FONT_PT = float(param("output", "min_font_pt"))
RASTER_DPI = int(param("output", "raster_dpi"))
FIGURE_FORMATS = tuple(param("output", "figure_formats"))

# Two text bounding boxes count as overlapping when their intersection over the
# smaller box exceeds this fraction. The value defines the check, so it is kept
# next to the check itself.
OVERLAP_IOU = 0.12

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": FONT_SIZE_PT,
    "axes.labelsize": FONT_SIZE_PT,
    "axes.titlesize": TITLE_SIZE_PT,
    "xtick.labelsize": TICK_SIZE_PT,
    "ytick.labelsize": TICK_SIZE_PT,
    "legend.fontsize": LEGEND_SIZE_PT,
    "axes.linewidth": AXIS_LINEWIDTH,
    "xtick.major.width": AXIS_LINEWIDTH,
    "ytick.major.width": AXIS_LINEWIDTH,
    "figure.dpi": RASTER_DPI,
    "savefig.dpi": RASTER_DPI,
    # Matplotlib defaults to U+2212 for the minus sign; Arial does not carry it
    # in every weight, so the ASCII hyphen is used instead.
    "axes.unicode_minus": False,
    # Font type 42 embeds TrueType outlines in PDF and PostScript output. The
    # publisher requires embeddable, editable text, and type 3 (the matplotlib
    # default for PostScript) is rejected.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- Label vocabularies ------------------------------------------------------
# The benchmark artefacts key several dictionaries by the labels that were used
# while the study was built. Some of those keys contain a non-ASCII character
# because the model, the correction arm or the layer was originally named in
# Chinese; the keys below are written as \uXXXX escapes so that this source file
# stays pure ASCII while matching the artefact dictionaries exactly. The value
# each key maps to is the English label printed in the figures. Nothing here is
# renamed in the artefacts: the keys are data, and translating them would break
# every lookup.

# The display-name vocabularies live in config/legacy_label_map.yaml, not here:
# their keys are the labels carried by the stage artefacts, some of which are not
# ASCII, so they are DATA rather than code. Loading them from common.config also
# keeps this module free of any non-English literal.
from common.config import LABELS  # noqa: E402

EN_METHOD = LABELS.get("figure_method", {})
EN_LAYER = LABELS.get("figure_layer", {})
TERM = LABELS.get("figure_term", {})


def EN(s):
    """Return the English label of an artefact label.

    Args:
        s: a method key, a layer key or an already-English label.

    Returns:
        str: the label to print. Method and layer keys are looked up in
        EN_METHOD and EN_LAYER; anything else keeps its text, with the numeric
        prefix and the implementation suffixes stripped so that an unmapped key
        still renders as a readable name.
    """
    s = str(s)
    if s in EN_METHOD:
        return EN_METHOD[s]
    if s in EN_LAYER:
        return EN_LAYER[s]
    return s.split("_", 1)[1].replace("(concat)", "").replace("-lite", "") if "_" in s else s


def tr(s):
    """Rewrite the strategy/verdict wording of a string into English.

    Args:
        s: any string that may contain a term listed in TERM.

    Returns:
        str: the input with every TERM key replaced by its English value.
    """
    s = str(s)
    for k, v in TERM.items():
        s = s.replace(k, v)
    return s


# --- Artefact access ---------------------------------------------------------


def load_json(path):
    """Read a JSON artefact written by an earlier stage.

    Args:
        path: the file to read, normally a path returned by
            common.config.work or common.config.data.

    Returns:
        The decoded object (dict, list, scalar), exactly as stored.
    """
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


# --- Plotting helpers --------------------------------------------------------


def cbar(im, ax, **kw):
    """Attach a compact colour bar to an image panel.

    Args:
        im: the mappable drawn by imshow.
        ax: the axes the colour bar takes its space from.
        **kw: forwarded to matplotlib.pyplot.colorbar.

    Returns:
        The created Colorbar, so the caller can label it.
    """
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035, **kw)
    cb.ax.tick_params(labelsize=TICK_SIZE_PT)
    return cb


def nospines(ax, which=("top", "right")):
    """Hide the given spines of an axes.

    Args:
        ax: the axes to modify.
        which: the spines to hide; the top and right spines carry no data and
            are removed by default.
    """
    for sp in which:
        ax.spines[sp].set_visible(False)


# --- Geometry verification ---------------------------------------------------


def check(fig, name, min_pt=None, verbose=True):
    """Measure a drawn figure and report every geometry violation.

    The three failures the check enforces are: type smaller than the journal
    floor, pairwise text overlap, and text leaving the canvas. The content
    coverage of the canvas is reported as an observation but is not a failure.

    Args:
        fig: the matplotlib Figure to inspect. It is drawn here, so the figure
            must be complete before the call.
        name: label used in the report line, normally the figure id ("Fig1").
        min_pt: minimum accepted type size in points; defaults to
            output.min_font_pt from params.yaml.
        verbose: when True, print one summary line and one line per problem.

    Returns:
        list[str]: one message per problem found. An empty list means the
        figure passes.
    """
    if min_pt is None:
        min_pt = MIN_FONT_PT
    probs = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    # (label, x0, y0, x1, y1, fontsize, axes index) for every visible text.
    items = []
    for ai, ax in enumerate(fig.get_axes()):
        objs = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        # A panel hidden with axis("off") keeps its tick labels as artists, but
        # they are never drawn; measuring them produces phantom overlaps and
        # false positives, so they are excluded.
        if ax.axison:
            objs += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in objs:
            s = t.get_text()
            if not s or not s.strip():
                continue
            if not t.get_visible():
                continue
            try:
                bb = t.get_window_extent(renderer=r)
            except Exception:
                # Some artists cannot be measured while they are being laid out;
                # they are skipped rather than failing the whole check.
                continue
            if bb.width <= 0 or bb.height <= 0:
                continue
            items.append((s[:36], bb.x0, bb.y0, bb.x1, bb.y1, t.get_fontsize(), ai))
    # 1) type size below the floor
    small = [(it[0], it[5]) for it in items if it[5] < min_pt - 1e-6]
    if small:
        probs.append(f"font<{min_pt}pt: " + "; ".join(f"{s}({fs:.1f})" for s, fs in small[:6]))
    # 2) pairwise overlap, measured as intersection over the smaller box
    ov = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
            if ix > 0 and iy > 0:
                aa = (a[3] - a[1]) * (a[4] - a[2])
                ab = (b[3] - b[1]) * (b[4] - b[2])
                iou = (ix * iy) / min(aa, ab) if min(aa, ab) > 0 else 0
                if iou > OVERLAP_IOU:
                    ov.append((a[0], b[0], round(iou, 2)))
    if ov:
        probs.append(f"overlap {len(ov)}: " + "; ".join(f"'{a}'x'{b}'({v})" for a, b, v in ov[:6]))
    # 3) content outside the canvas (2 px tolerance for antialiasing)
    out = [(s, f"ax{ai}", round(x1), round(y1)) for s, x0, y0, x1, y1, _, ai in items
           if x0 < -2 or y0 < -2 or x1 > W + 2 or y1 > H + 2]
    if out:
        probs.append(f"out-of-canvas {len(out)}: {out[:4]}")
    # 4) observation: how much of the canvas the text actually covers
    if items:
        gx0 = min(i[1] for i in items)
        gx1 = max(i[3] for i in items)
        gy0 = min(i[2] for i in items)
        gy1 = max(i[4] for i in items)
        info = f"coverage {((gx1 - gx0) * (gy1 - gy0)) / (W * H) * 100:.1f}% | {len(items)} texts | {W}x{H}px"
    else:
        info = "no text"
    if verbose:
        print(f"  {'✓' if not probs else '✗'} {name}: {info}")
        for p in probs:
            print(f"        · {p}")
    return probs


def save(fig, stem, name, formats=None):
    """Check and write one figure in every format the submission requires.

    Args:
        fig: the completed Figure.
        stem: file name without extension; the formats are appended to it.
        name: label passed to check() for the console report.
        formats: extensions to write; defaults to output.figure_formats from
            params.yaml.

    Returns:
        list[str]: the problems reported by check() for this figure.
    """
    if formats is None:
        formats = FIGURE_FORMATS
    out_dir = ensure_dir(results("figures_dir"))
    probs = check(fig, name)
    for ext in formats:
        if ext == "tif":
            continue
        # PDF and EPS stay vector and go to production; PNG is the preview.
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    if "tif" in formats:
        # The journal systems ask for TIFF. It is produced from the PNG so that
        # the rasterisation is the same one that was geometry-checked, and it is
        # written LZW-compressed (lossless) with the resolution in the metadata.
        from PIL import Image
        with Image.open(out_dir / f"{stem}.png") as im:
            im.convert("RGB").save(out_dir / f"{stem}.tif", format="TIFF",
                                   compression="tiff_lzw", dpi=(RASTER_DPI, RASTER_DPI))
    plt.close(fig)
    return probs
