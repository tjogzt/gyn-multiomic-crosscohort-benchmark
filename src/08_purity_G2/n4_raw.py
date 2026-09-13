#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n4：看 dis 纯度列的原始取值"""
import pandas as pd
B = "/Volumes/tjogzt4T/data"
d = pd.read_csv(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
for c in ["Tumor_purity", "Purity_Immune", "Purity_Cancer", "Purity_Stroma"]:
    if c not in d.columns: 
        print(f"{c}: 列不存在"); continue
    s = d[c]
    print("=" * 70); print(f"{c}  | dtype={s.dtype} | 非空 {s.notna().sum()}/{len(s)}")
    vals = s.dropna().astype(str)
    print("  唯一值前 12:", list(vals.unique())[:12])
    num = pd.to_numeric(vals.str.replace(",", ""), errors="coerce")
    print(f"  去逗号后数值化成功: {num.notna().sum()}/{len(vals)}")
    if num.notna().sum():
        print(f"    均值 {num.mean():.4f} 范围 [{num.min():.4f},{num.max():.4f}]")
