#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建全部 PDF 并做缺失标识符/缺字校验"""
import os, subprocess, re, fitz

BASE = "/Users/taozhu/my researches/retyping"
TMP = "/tmp/gyn_retyping/pdfbuild"; os.makedirs(TMP, exist_ok=True)

MAP = [("\ufe0f", ""), ("\u26a0", "\u25b2"), ("\u274c", "\u00d7"),
       ("\u2705", "\u221a"), ("\u23f3", "\u2026"), ("\U0001f4cc", "\u2591"),
       ("\u25d0", "\u25cb"), ("\u2b50", "\u2605"), ("\U0001f6a9", "\u2691"),
       ("\u2b07", "\u2193"), ("\U0001f7e1", "\u25cf"), ("\U0001f534", "\u25cf")]

DOCS = ["方向F_妇科肿瘤再分型_进展与选题论证", "四方团队审查裁决报告_方向F",
        "外部队列数据获取清单_G0", "方向F_预注册计划_v0.1",
        "外部队列获取执行报告_S1S5", "跨队列对齐测试报告",
        "跨队列对齐测试报告_v2", "方向F_预注册计划_v0.2",
        "对齐策略验证与EEEC获取报告", "多组学整合基准报告", "方向F_预注册计划_v0.3", "EEEC表型映射与队列解析报告"]

# 长无空格 token 集合（用于缺失校验）
TOKENS = {
 "外部队列数据获取清单_G0": ["PDC000357", "phs003152.v1.p1", "10.7937/6rda-p940",
    "PXD046507", "HRA018093", "PXD068004", "PXD015903", "phs001287",
    "PDC000125", "PDC000439"],
 "方向F_预注册计划_v0.1": ["PDC000125/126/226", "PDC000357-362", "phs003152.v1.p1",
    "PXD046507", "PXD068004", "10.7937/6rda-p940", "NCT05640999"],
 "方向F_预注册计划_v0.2": ["PDC000439/441/443/445", "PDC000125/126/226", "PDC000357-362",
    "phs003152.v1.p1", "PXD046507", "PXD068004", "10.7937/6rda-p940", "NCT05640999",
    "ABSOLUTE_tumor_purity", "Tumor_purity"],
 "跨队列对齐测试报告": ["align_concordance.json", "dirF_alignment.png"],
 "跨队列对齐测试报告_v2": ["bridge_3cohort.json", "b3_anchors.json",
    "tcga450k_genelevel_ucec.csv.gz", "dirF_bridge.png"],
}

print("=" * 78); print("【构建】"); print("=" * 78)
ok = 0
for d in DOCS:
    src = os.path.join(BASE, d + ".md")
    if not os.path.exists(src): print("MISSING", d); continue
    s = open(src, encoding="utf-8").read()
    for a, b in MAP: s = s.replace(a, b)
    dst_md = os.path.join(TMP, d + ".md")
    open(dst_md, "w", encoding="utf-8").write(s)
    out = os.path.join(BASE, d + ".pdf")
    r = subprocess.run(["pandoc", dst_md, "-o", out, "--pdf-engine=xelatex",
                        "-V", "geometry:top=2.2cm,bottom=2.2cm,left=2.0cm,right=2.0cm",
                        "-V", "fontsize=10pt", "-H", "/tmp/gyn_retyping/head.tex",
                        "--toc", "--toc-depth=2"], capture_output=True, text=True)
    miss = sorted(set(re.findall(r"no (.+?) \(U\+([0-9A-F]+)\)", r.stderr)))
    if r.returncode == 0: ok += 1
    print(f"  {d}: rc={r.returncode} 缺字形={len(miss)} {miss[:3]}")
print(f"\n构建成功 {ok}/{len(DOCS)}")

print("\n" + "=" * 78); print("【校验】缺失标识符 / 缺字 / 页数"); print("=" * 78)
allbad = 0
for d in DOCS:
    p = os.path.join(BASE, d + ".pdf")
    if not os.path.exists(p): print(f"  {d}: PDF 不存在"); continue
    doc = fitz.open(p); t = "".join(pg.get_text() for pg in doc)
    flat = re.sub(r"\s", "", t)
    bad = [k for k in TOKENS.get(d, []) if k.replace(" ", "") not in flat]
    tofu = t.count("\ufffd")
    allbad += len(bad)
    flag = "OK " if (not bad and tofu == 0) else "!! "
    print(f"  {flag}{d}: {doc.page_count}页 缺失={len(bad)} 缺字={tofu} {bad}")
print(f"\n总缺失: {allbad}")
