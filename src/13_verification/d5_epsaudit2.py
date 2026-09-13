#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d5：EPS 审计（修正版）——matplotlib 在 prolog 缩写算子，须查图像算子与字体类型"""
import os, re, glob
R = "/Users/taozhu/my researches/retyping"
HEATMAP_FIGS = {"Fig1", "Fig2", "Fig3", "Fig5", "FigS1", "FigS3", "FigS6"}  # 含 imshow 热图
rows = []
for eps in sorted(glob.glob(f"{R}/Fig*.eps")):
    b = open(eps, "rb").read()
    stem = os.path.splitext(os.path.basename(eps))[0]
    tag = re.match(r"(FigS?\d+)", stem).group(1)
    n_img = b.count(b"\nimage\n") + b.count(b" image\n")
    n_show = b.count(b" show\n") + b.count(b"show\n")
    t42 = b.count(b"TrueTypeFont") > 0
    text_only = b.count(b"\x00") < 50
    expect_raster = tag in HEATMAP_FIGS
    verdict = ("矢量+热图位图(预期)" if (n_img > 0 and expect_raster)
               else "纯矢量 ✅" if n_img == 0
               else "⚠ 非预期位图")
    rows.append((stem, len(b) / 1024, n_img, n_show, t42, text_only, verdict))
    print(f"  {stem:40s} {len(b)/1024:6.1f}KB  image={n_img:<3d} show={n_show:<4d} "
          f"Type42={'Y' if t42 else 'n'}  ASCII={'Y' if text_only else 'n'}  {verdict}")
bad = [r for r in rows if "⚠" in r[-1]]
print(f"\n共 {len(rows)} 张；异常 {len(bad)}")
if bad: print("  异常:", [r[0] for r in bad])
print("\n判据：image 算子 = 0 → 纯矢量；含热图的图预期有 1–2 个位图算子（imshow 本质为栅格）。")
