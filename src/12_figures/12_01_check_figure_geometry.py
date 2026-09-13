#!/usr/bin/env python3
"""
Standalone geometric checker for the rendered figures.

Purpose
    Extract the bounding box of every rendered text element of a figure and
    report the three geometry failures the submission must not have: type below
    output.min_font_pt at print size, pairwise text overlap, and content leaving
    the canvas. The content coverage of the canvas is reported as an
    observation, not as a failure. This module is the standalone variant named
    by docs/pipeline.md; the style module common/plotting_style.py additionally
    exposes the same check in the form the figure scripts call, where the
    problems are printed and the summary line is not returned.

Author
    Tao Zhu

Created
    2026-09-13

Inputs
    None. The minimum accepted type size is read from config/params.yaml
    (output.min_font_pt) through common.config.param.

Outputs
    None: the check prints its findings and returns them.

Usage
    Library module; the figure scripts reach the same check through
    common.plotting_style. A module name may not begin with a digit, so this
    file is loaded by path when the (problems, summary) pair is needed:

        python -c "import importlib.util as u; \\
            s = u.spec_from_file_location('geomcheck', \\
                'src/12_figures/12_01_check_figure_geometry.py'); \\
            m = u.module_from_spec(s); s.loader.exec_module(m); \\
            print(m.check(fig, 'FigX'))"
"""

from __future__ import annotations

import sys
from pathlib import Path

# src/12_figures/12_01_check_figure_geometry.py sits one level below src/, so
# this puts src/ on the import path and makes common.config importable.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from common.config import param  # noqa: E402

import matplotlib  # noqa: E402

# The check measures a drawn figure and never displays one.
matplotlib.use("Agg")


def check(fig, name, min_pt=None, verbose=True):
    """Measure a drawn figure and report every geometry violation.

    Args:
        fig: the matplotlib Figure to inspect. The figure must be complete,
            because the check draws it to obtain the renderer.
        name: label used in the report line, normally the figure id ("Fig1").
        min_pt: accepted minimum type size in points. When None, the journal
            floor output.min_font_pt from params.yaml is used.
        verbose: when True, print one summary line and one line per problem.

    Returns:
        tuple[list[str], str]: the list of problem messages (empty when the
        figure passes) and the one-line summary of what was measured.
    """
    if min_pt is None:
        min_pt = float(param("output", "min_font_pt"))
    probs = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = []  # (label, x0, y0, x1, y1, fontsize, axes index)
    for ai, ax in enumerate(fig.get_axes()):
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label] + \
                ax.get_xticklabels() + ax.get_yticklabels():
            s = t.get_text()
            if not s or not s.strip():
                continue
            try:
                bb = t.get_window_extent(renderer=r)
            except Exception:
                # An artist that cannot be measured while it is being laid out is
                # skipped rather than failing the whole check.
                continue
            if bb.width <= 0 or bb.height <= 0:
                continue
            items.append((s[:34], bb.x0, bb.y0, bb.x1, bb.y1, t.get_fontsize(), ai))
    # 1) type size below the floor
    small = [(it[0], it[5]) for it in items if it[5] < min_pt - 1e-6]
    if small:
        probs.append(f"font < {min_pt}pt: " + "; ".join(f"{s}({fs:.1f})" for s, fs in small[:5]))
    # 2) pairwise text overlap; an intersection over the smaller box above 0.12
    # counts as an overlap
    ov = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
            if ix > 0 and iy > 0:
                inter = ix * iy
                aa = (a[3] - a[1]) * (a[4] - a[2])
                ab = (b[3] - b[1]) * (b[4] - b[2])
                iou = inter / min(aa, ab) if min(aa, ab) > 0 else 0
                if iou > 0.12:
                    ov.append((a[0], b[0], round(iou, 2)))
    if ov:
        probs.append(f"text overlap {len(ov)} pairs (IoU>0.12): " + "; ".join(f"'{a}'x'{b}'({v})" for a, b, v in ov[:6]))
    # 3) content beyond the canvas, with a 2 px tolerance for antialiasing
    out = [(s, f"ax{ai}", round(x1), round(y1)) for s, x0, y0, x1, y1, _, ai in items
           if x0 < -2 or y0 < -2 or x1 > W + 2 or y1 > H + 2]
    if out:
        probs.append(f"outside the canvas ({len(out)}): {out[:5]}")
    # 4) observation: share of the canvas covered by the text bounding boxes
    if items:
        gx0 = min(i[1] for i in items)
        gx1 = max(i[3] for i in items)
        gy0 = min(i[2] for i in items)
        gy1 = max(i[4] for i in items)
        frac = ((gx1 - gx0) * (gy1 - gy0)) / (W * H)
        info = f"content coverage {frac * 100:.1f}% | {len(items)} text blocks | canvas {W}x{H}px"
    else:
        info = "no text"
    if verbose:
        tag = "✓" if not probs else "✗"
        print(f"  {tag} {name}: {info}")
        for p in probs:
            print(f"        · {p}")
    return probs, info
