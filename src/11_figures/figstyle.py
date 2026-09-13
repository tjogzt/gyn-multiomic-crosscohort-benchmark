#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""figstyle.py — 全局统一样式与几何校验（图件共用）
配色语义（不可逐图更换）：
  C_PY  石青 #3D6BA8 = Python 自实现工具链      C_R   朱砂 #C23531 = R 规范实现工具链
  C_NEU 靛青 #177CB0 = 中性/参照                C_BAD 胭脂 #9D2933 = 失效/退化/不可判
  C_GY  灰   #999999 = 辅助（原始值、无校正）    C_OK  松绿 #3F6B3F = 通过/可用区间
"""
import json, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_PY, C_R, C_NEU, C_BAD, C_GY, C_OK = "#3D6BA8", "#C23531", "#177CB0", "#9D2933", "#999999", "#3F6B3F"
C_SPAN = "#E8F0E8"
T = "/tmp/gyn_retyping"
OUT = "/Users/taozhu/my researches/retyping"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
EN_METHOD = {
    "1_SNF": "SNF", "2_NMF(concat)": "NMF", "3_KMeans(concat)": "KMeans", "4_PCA(concat)": "PCA",
    "5_谱嵌入(concat)": "Spectral", "6_共联共识": "Co-assoc.", "7_MCCA-lite": "MCCA",
    "8_MOFA-lite": "MOFA", "2_intNMF等价": "intNMF*", "6_MOFA2": "MOFA2",
    "5_iClusterPlus": "iCluster", "3_MCIA等价(MFA)": "MCIA*", "4_mixOmics": "mixOmics",
}
EN_LAYER = {"表达": "mRNA", "拷贝数": "CNA", "甲基化": "Methylation", "蛋白": "Protein",
            "mRNA": "mRNA", "miRNA": "miRNA", "CNA": "CNA", "meth": "Methylation",
            "RPPA": "RPPA", "甲基化450K": "Methylation450K"}
def EN(s):
    s = str(s)
    if s in EN_METHOD: return EN_METHOD[s]
    if s in EN_LAYER: return EN_LAYER[s]
    return s.split("_", 1)[1].replace("(concat)", "").replace("-lite", "") if "_" in s else s

# ── 术语表：中→英（图内一律英文；目标刊为英文刊） ──
TERM = {
    "基线": "baseline", "分位数标准化": "quantile norm.", "Δ参考校正": "Δ-reference",
    "ComBat(池化)": "ComBat (pooled)", "恒等性验证": "identity check",
    "可重复性天花板": "reproducibility ceiling", "数值恒等阈值": "identity threshold",
    "退化阈值": "degeneracy threshold", "实质下限": "substantive floor",
    "null 95 分位": "null 95th pct", "队列内连自己都复现不了": "within-cohort: cannot reproduce itself",
    "门槛": "threshold", "基线值": "baseline",
}
def tr(s):
    s = str(s)
    for k, v in TERM.items(): s = s.replace(k, v)
    return s

def J(p): return json.load(open(f"{T}/{p}"))
def cbar(im, ax, **kw):
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035, **kw)
    cb.ax.tick_params(labelsize=8); return cb
def nospines(ax, which=("top", "right")):
    for sp in which: ax.spines[sp].set_visible(False)

def check(fig, name, min_pt=8.0, verbose=True):
    """几何校验：字号 / 文本重叠 / 越出画布 / 内容占比"""
    probs = []
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    items = []
    for ai, ax in enumerate(fig.get_axes()):
        objs = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        if ax.axison:                      # axis("off") 的刻度不可见，不参与几何校验
            objs += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        for t in objs:
            s = t.get_text()
            if not s or not s.strip(): continue
            if not t.get_visible(): continue
            try: bb = t.get_window_extent(renderer=r)
            except Exception: continue
            if bb.width <= 0 or bb.height <= 0: continue
            items.append((s[:36], bb.x0, bb.y0, bb.x1, bb.y1, t.get_fontsize(), ai))
    small = [(it[0], it[5]) for it in items if it[5] < min_pt - 1e-6]
    if small:
        probs.append(f"font<{min_pt}pt: " + "; ".join(f"{s}({fs:.1f})" for s, fs in small[:6]))
    ov = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            iy = max(0, min(a[4], b[4]) - max(a[2], b[2]))
            if ix > 0 and iy > 0:
                aa = (a[3]-a[1])*(a[4]-a[2]); ab = (b[3]-b[1])*(b[4]-b[2])
                iou = (ix*iy)/min(aa, ab) if min(aa, ab) > 0 else 0
                if iou > 0.12: ov.append((a[0], b[0], round(iou, 2)))
    if ov:
        probs.append(f"overlap {len(ov)}: " + "; ".join(f"'{a}'x'{b}'({v})" for a, b, v in ov[:6]))
    out = [(s, f"ax{ai}", round(x1), round(y1)) for s, x0, y0, x1, y1, _, ai in items
           if x0 < -2 or y0 < -2 or x1 > W + 2 or y1 > H + 2]
    if out: probs.append(f"out-of-canvas {len(out)}: {out[:4]}")
    if items:
        gx0 = min(i[1] for i in items); gx1 = max(i[3] for i in items)
        gy0 = min(i[2] for i in items); gy1 = max(i[4] for i in items)
        info = f"coverage {((gx1-gx0)*(gy1-gy0))/(W*H)*100:.1f}% | {len(items)} texts | {W}x{H}px"
    else: info = "no text"
    if verbose:
        print(f"  {'✓' if not probs else '✗'} {name}: {info}")
        for p in probs: print(f"        · {p}")
    return probs

def save(fig, stem, name, formats=("png", "pdf", "eps", "tif")):
    """输出投稿所需全部格式：
       PDF/EPS = 矢量（供排版）；PNG = 预览；TIF = 300dpi 位图（期刊系统常要求）
    """
    probs = check(fig, name)
    for ext in formats:
        if ext == "tif":
            continue
        fig.savefig(f"{OUT}/{stem}.{ext}", bbox_inches="tight", facecolor="white")
    # TIFF：由 PNG 转（LZW 无损 + 写入 dpi 元数据）
    if "tif" in formats:
        from PIL import Image
        with Image.open(f"{OUT}/{stem}.png") as im:
            w_in = im.size[0] / 300.0
            im.convert("RGB").save(f"{OUT}/{stem}.tif", format="TIFF",
                                   compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)
    return probs
