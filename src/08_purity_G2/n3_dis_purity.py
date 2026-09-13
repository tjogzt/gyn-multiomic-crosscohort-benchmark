#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n3：诊断 dis 的 Tumor_purity 列"""
import numpy as np, pandas as pd
B = "/Volumes/tjogzt4T/data"
p = f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt"
d = pd.read_csv(p, sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
print("形状:", d.shape)
print("\nTumor_purity 列:")
s = d["Tumor_purity"]
print("  dtype:", s.dtype, "| 非空:", s.notna().sum(), "| 唯一值前 10:", list(s.dropna().unique())[:10])
print("  前 8 个原始值:", list(s.head(8)))
print("\n尝试数值化后的统计:")
n = pd.to_numeric(s, errors="coerce")
print("  非空:", n.notna().sum(), "| 均值:", n.mean(), "| 范围:", (n.min(), n.max()))
print("\n其他可能含纯度的列名（搜 'pur'）：", [c for c in d.columns if "pur" in str(c).lower()])
print("\nProteomics_Aliquot_ID 样例:", list(d["Proteomics_Aliquot_ID"].astype(str).head(3)))
print("idx 样例:", list(d["idx"].astype(str).head(3)))
print("\nTumor_Normal 分布:"); print(d["Proteomics_Tumor_Normal"].value_counts().to_string())
print("\n全表非空计数（前 25 列）:")
print(d.notna().sum().head(25).to_string())
