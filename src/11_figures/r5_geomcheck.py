#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r5：图形几何校验——文本重叠 / 越界 / 缺字 / 画布留白"""
import sys, warnings, io, contextlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def check(fig, name, min_pt=8.0, verbose=True):
    """min_pt: 允许的最小字号（pt）。返回 (问题列表, 统计)"""
    probs = []
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = []   # (label, x0,y0,x1,y1, fs)
    for ai, ax in enumerate(fig.get_axes()):
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label] + \
                 ax.get_xticklabels() + ax.get_yticklabels():
            s = t.get_text()
            if not s or not s.strip(): continue
            try:
                bb = t.get_window_extent(renderer=r)
            except Exception:
                continue
            if bb.width <= 0 or bb.height <= 0: continue
            items.append((s[:34], bb.x0, bb.y0, bb.x1, bb.y1, t.get_fontsize(), ai))
    # 1) 字号
    small = [(it[0], it[5]) for it in items if it[5] < min_pt - 1e-6]
    if small:
        probs.append(f"字号 < {min_pt}pt: " + "; ".join(f"{s}({fs:.1f})" for s, fs in small[:5]))
    # 2) 文本两两重叠（面积 IoU > 0.12 视为重叠）
    ov = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
            if ix > 0 and iy > 0:
                inter = ix * iy
                aa = (a[3]-a[1])*(a[4]-a[2]); ab = (b[3]-b[1])*(b[4]-b[2])
                iou = inter / min(aa, ab) if min(aa, ab) > 0 else 0
                if iou > 0.12:
                    ov.append((a[0], b[0], round(iou, 2)))
    if ov:
        probs.append(f"文本重叠 {len(ov)} 对（IoU>0.12）: " + "; ".join(f"'{a}'×'{b}'({v})" for a, b, v in ov[:6]))
    # 3) 越界（超出画布）
    out = [(s, f"ax{ai}", round(x1), round(y1)) for s, x0, y0, x1, y1, _, ai in items
           if x0 < -2 or y0 < -2 or x1 > W + 2 or y1 > H + 2]
    if out:
        probs.append(f"越出画布 {len(out)} 处: {out[:5]}")
    # 4) 画布留白（文字内容包围盒占画布比例）
    if items:
        gx0 = min(i[1] for i in items); gx1 = max(i[3] for i in items)
        gy0 = min(i[2] for i in items); gy1 = max(i[4] for i in items)
        frac = ((gx1-gx0)*(gy1-gy0))/(W*H)
        info = f"内容占比 {frac*100:.1f}% | 文本块 {len(items)} | 画布 {W}×{H}px"
    else:
        info = "无文本"
    if verbose:
        tag = "✓" if not probs else "✗"
        print(f"  {tag} {name}: {info}")
        for p in probs: print(f"        · {p}")
    return probs, info
