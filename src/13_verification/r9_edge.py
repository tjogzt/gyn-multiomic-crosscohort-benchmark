#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r9：验证 越界标签 是否真的进入输出文件（tight bbox 会外扩画布）"""
import fitz, re, os
R = "/Users/taozhu/my researches/retyping"
for f in ["Fig1_availability_comparability", "Fig2_layer_negative_marginal", "Fig3_method_not_resolvable"]:
    p = f"{R}/{f}.pdf"
    d = fitz.open(p)
    t = " ".join(pg.get_text() for pg in d)
    pg0 = d[0].rect
    print(f"── {f}.pdf  {len(d)}页  页面 {pg0.width:.0f}×{pg0.height:.0f}pt")
    # 边界标签
    for tok in ["0.6", "0.8", "−0.066", "0.412", "0.895", "0.208", "0.0118"]:
        print(f"     {'✓' if tok in t else '✗'} {tok}")
    # 检查是否有文本块贴到页面边缘（被裁的迹象）
    for pi, pg in enumerate(d):
        for b in pg.get_text("blocks"):
            x0, y0, x1, y1 = b[:4]
            if x1 > pg0.width - 1.0 or y1 > pg0.height - 1.0 or x0 < 1.0 or y0 < 1.0:
                print(f"     ⚠ 页{pi} 文本块贴边 x0={x0:.1f} y0={y0:.1f} x1={x1:.1f} y1={y1:.1f} : {b[4][:40]!r}")
    print()
