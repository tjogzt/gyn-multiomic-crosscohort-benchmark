#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d4：仅生成 TIFF（EPS 由 matplotlib 原生输出，勿被 gs 覆盖）"""
import os, glob
from PIL import Image
R = "/Users/taozhu/my researches/retyping"
pngs = sorted(glob.glob(f"{R}/Fig[0-9]_*.png")) + sorted(glob.glob(f"{R}/FigS*.png"))
print(f"{'stem':40s} {'TIF':>9s}  dpi  width_mm")
for png in pngs:
    stem = os.path.splitext(os.path.basename(png))[0]
    tif = f"{R}/{stem}.tif"
    with Image.open(png) as im:
        mm = im.size[0] / 300 * 25.4
        im.convert("RGB").save(tif, format="TIFF", compression="tiff_lzw", dpi=(300, 300))
    with Image.open(tif) as im2:
        d = im2.info.get("dpi", (None,))[0]
        dpi = round(float(d)) if d is not None else None
    print(f"{stem:40s} {os.path.getsize(tif)/1024:8.0f}K {dpi:>4} {mm:.1f}")
print(f"\nTIFF 共 {len(pngs)} 张")
