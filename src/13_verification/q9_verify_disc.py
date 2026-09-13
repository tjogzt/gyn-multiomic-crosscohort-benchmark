#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q9：Discussion 引用核验——每个 DOI / 卷期页码须可溯源"""
import json, re, os
T = "/tmp/gyn_retyping"; R = "/Users/taozhu/my researches/retyping"

md = open(os.path.join(R, "F1_Discussion.md")).read()
refsec = md.split("## 参考文献", 1)[1] if "## 参考文献" in md else md

# --- 已核验集合（来自 Crossref / PubMed 实测返回）---
VERIFIED = {}
def add(doi, src, meta):
    if doi: VERIFIED[doi.lower()] = (src, meta)

for f, src in [("disc_refs.json", "crossref-r1"), ("disc_refs2.json", "crossref-r2")]:
    p = os.path.join(T, f)
    if os.path.exists(p):
        for r in json.load(open(p)):
            add(r.get("doi"), src, f"{r.get('first_author')} {r.get('year')} {r.get('journal')}")
p = os.path.join(T, "disc_refs3.json")
if os.path.exists(p):
    for r in json.load(open(p)):
        top = r.get("top") or {}
        dois = [x["value"] for x in (top.get("articleids") or []) if x.get("idtype") == "doi"]
        add(dois[0] if dois else None, "pubmed-r3", str(top.get("title"))[:60])
# 手工确认（q6/q7 实测摘要 + q8）
add("10.1016/j.cell.2016.05.069", "pubmed-efetch", "Zhang H Cell 2016 (摘要实测)")
add("10.1038/s41467-020-20430-7", "crossref-q8", "Cantini L 2021 Nat Commun")
add("10.1016/j.ccell.2023.07.007", "pubmed-q3", "Dou Y Cancer Cell 2023")
add("10.1038/s41588-024-01703-z", "pubmed-q3", "Hu Z Nat Genet 2024")
add("10.1073/pnas.1208949110", "pubmed-q4", "Mo Q PNAS 2013")
add("10.1093/bioinformatics/btp659", "crossref-r2", "Shen R Bioinformatics 2009")
add("10.1007/bf01908075", "crossref-r2", "Hubert Arabie 1985")
add("10.1038/nature12113", "ref_verify2", "Kandoth Nature 2013")
add("10.1038/s41592-019-0619-0", "crossref-r2", "Korsunsky Nat Methods 2019")
add("10.1016/j.cell.2025.10.043", "pubmed-efetch", "Cell 2025 erratum")

# --- 抽取 Discussion 中的 DOI ---
dois = re.findall(r"doi:(10\.[^\s）)。,（*]+)", refsec)
print(f"参考文献节 DOI 数: {len(dois)}  （去重后 {len(set(d.lower() for d in dois))}）\n")
bad = []
for d in dict.fromkeys(dois):
    k = d.lower().rstrip(".")
    if k in VERIFIED: print(f"  ✓ {d:44s} [{VERIFIED[k][0]}] {VERIFIED[k][1]}")
    else: bad.append(d); print(f"  ✗ {d:44s} 不在已核验集合")
print(f"\n未溯源: {len(bad)}")

# --- 编号与条目数一致性 ---
nums = re.findall(r"^(\d+)\. ", refsec, re.M)
print(f"参考文献编号: {len(nums)} 条，范围 {nums[0]}–{nums[-1]}", "✓ 连续" if nums == [str(i) for i in range(1, len(nums)+1)] else "✗ 不连续")
# 正文引用
cited = re.findall(r"\[([0-9,\s\-–]+)\]", md)
print(f"正文方括号引用: {len(cited)} 处")
json.dump(dict(dois=dois, bad=bad), open(os.path.join(T, "disc_verify.json"), "w"), ensure_ascii=False, indent=1)
