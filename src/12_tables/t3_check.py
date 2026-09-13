#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""t3：表文档质量核验——中文残留 / 数值溯源 / 表数"""
import re, os, json
import pandas as pd
R = "/Users/taozhu/my researches/retyping"; TD = f"{R}/tables"; T = "/tmp/gyn_retyping"
md = open(f"{R}/F1_Tables.md", encoding="utf-8").read()
body = md.split("---", 1)[1] if "---" in md else md   # 跳过题头

# 1) 中文残留（排除说明性中文段落，只看表格行）
tablines = [l for l in body.split("\n") if l.startswith("|")]
zh = [(i, l) for i, l in enumerate(tablines) if re.search(r"[\u4e00-\u9fff]", l)]
print(f"1) 表格行总数 {len(tablines)}；含中文的行 {len(zh)}")
for i, l in zh[:5]: print(f"     {l[:110]}")

# 2) 表数与 CSV 数
ntab = len(re.findall(r"^## Table ", md, re.M))
ncsv = len([f for f in os.listdir(TD) if f.endswith(".csv")])
print(f"2) 文档表数 {ntab}；tables/ 下 CSV {ncsv}")

# 3) 数值溯源：每个数值必须能在对应 CSV 中找到
def J(p): return json.load(open(f"{T}/{p}"))
MAP = {
    "Table 3": ("T3_python_method_ranking.csv", J("bench_summary.json")["tab1"]),
    "Table 4": ("T4_R_package_ranking.csv", J("bench_merged.json")["r_ranking"]),
    "Table 5a": ("T5a_pac_by_method.csv", J("pac_agg.json")["tab1"]),
}
ok = True
for tab, (fn, src) in MAP.items():
    df = pd.read_csv(f"{TD}/{fn}")
    n = len(df)
    nums = pd.to_numeric(df.select_dtypes("number").stack(), errors="coerce").dropna()
    found = sum(1 for v in nums if any(abs(v - round(float(sv), 4)) < 1e-9
                                       for sv in src.values() if isinstance(sv, (int, float)))
                or any(abs(v - round(float(sv.get(k, 0)), 4)) < 1e-9 for sv in src.values()
                       if isinstance(sv, dict) for k in ("A", "B", "all", "bal", "deg", "strong", "med")))
    print(f"3) {tab}: {n} 行 / {len(nums)} 个数值 → 溯源命中 {found}")
    if found == 0: ok = False

# 4) 关键单值抽查
CHK = [("T3", "KMeans", "overall_ARI", 0.3947), ("T3", "SNF", "overall_ARI", 0.0118),
       ("T4", "SNF", "mean_ARI", 0.2081), ("T5b", "[0.0, 0.3)", "mean_transfer_ARI", 0.2890),
       ("T6", None, "centroid_auc", None)]
for fn, key, col, exp in [("T3_python_method_ranking.csv", "KMeans", "overall_ARI", 0.3947),
                          ("T3_python_method_ranking.csv", "SNF", "overall_ARI", 0.0118),
                          ("T4_R_package_ranking.csv", "SNF", "mean_ARI", 0.2081),
                          ("T5b_pac_bins_to_ARI.csv", "[0.0, 0.3)", "mean_transfer_ARI", 0.2890)]:
    d = pd.read_csv(f"{TD}/{fn}")
    got = float(d[d.iloc[:, 0] == key][col].iloc[0])
    print(f"4) {fn[:28]} [{key}] {col} = {got}  期望 {exp}  {'OK' if abs(got-exp) < 1e-9 else 'MISMATCH'}")
print("\n表格文档核验完成")
