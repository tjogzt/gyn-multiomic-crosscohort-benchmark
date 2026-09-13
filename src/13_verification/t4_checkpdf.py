#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t4：F1_Tables.pdf 针对性校验（缺字 / 页数 / 数值命中 / 时效）"""
import fitz, re, os
import pandas as pd
R = "/Users/taozhu/my researches/retyping"; TD = f"{R}/tables"
mp, pp = f"{R}/F1_Tables.md", f"{R}/F1_Tables.pdf"
st_m, st_p = os.stat(mp).st_mtime, os.stat(pp).st_mtime
print(f"1) 时效: md {st_m:.0f} vs pdf {st_p:.0f} → {'OK（pdf 更新）' if st_p >= st_m else '✗ pdf 过期'}")

d = fitz.open(pp)
txt = " ".join(pg.get_text() for pg in d)
print(f"2) PDF: {len(d)} 页 / {len(txt):,} 字符")

TOFU = ["□", "\ufffd", "\u25a1"]
found = [c for c in TOFU if c in txt]
print(f"3) 缺字（tofu）: {'无' if not found else found}")

# 表标题
tabs = re.findall(r"Table (\d+[abc]?)", txt)
print(f"4) PDF 中出现的表号: {sorted(set(tabs))}")

# 数值命中：每个 CSV 的首个数值必须出现在 PDF
miss = []
for f in sorted(os.listdir(TD)):
    if not f.endswith(".csv"): continue
    df = pd.read_csv(f"{TD}/{f}")
    numdf = df.select_dtypes("number")
    if numdf.shape[1] == 0: continue
    nums = pd.to_numeric(numdf.iloc[:, 0], errors="coerce").dropna()
    if len(nums) == 0: continue
    v = f"{nums.iloc[0]:g}"
    if v not in txt.replace(" ", "").replace("\n", "") and v not in txt:
        miss.append((f, v))
print(f"5) 各 CSV 首值在 PDF 中命中: {len([f for f in os.listdir(TD) if f.endswith('.csv')]) - len(miss)}/{len([f for f in os.listdir(TD) if f.endswith('.csv')])}")
for f, v in miss: print(f"     ✗ {f}: {v}")

# 关键值抽查
KEY = [("0.3947", "KMeans overall"), ("0.0118", "SNF overall"), ("0.2081", "SNF R-side"),
       ("0.2890", "PAC<0.30 bin"), ("0.5000", "limma centroid"), ("0.5217", "Harmony"),
       ("0.8473", "cross-platform Δ"), ("0.2968", "network"), ("306", "core set"),
       ("479", "G2 required n")]
print("6) 关键值抽查:")
for v, lab in KEY:
    print(f"     {'✓' if v in txt else '✗'} {v:>8s}  {lab}")
print("\n表文档校验完成")
