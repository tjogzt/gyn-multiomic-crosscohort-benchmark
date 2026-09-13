#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""终极校验：从 md 源自动抽取长无空格 token，逐个比对 PDF；+ 缺字 + 页数"""
import os, re, fitz
BASE = "/Users/taozhu/my researches/retyping"
DOCS = ["方向F_妇科肿瘤再分型_进展与选题论证", "四方团队审查裁决报告_方向F",
        "外部队列数据获取清单_G0", "方向F_预注册计划_v0.1",
        "外部队列获取执行报告_S1S5", "跨队列对齐测试报告",
        "跨队列对齐测试报告_v2", "方向F_预注册计划_v0.2"]

# 抽取连续 >=14 字符、不含空格的中文/ASCII 混杂长 token（易被排版截断者）
TOK = re.compile(r"[A-Za-z0-9_.:/\-\[\]<>+=]{14,}")

print("=" * 84)
print("从 md 源自动抽取长 token → 逐条比对 PDF（不依赖任何手工清单）")
print("=" * 84)
tot_missing = 0
for d in DOCS:
    md = os.path.join(BASE, d + ".md"); pdf = os.path.join(BASE, d + ".pdf")
    if not (os.path.exists(md) and os.path.exists(pdf)):
        print(f"  !! {d}: 文件缺失"); continue
    src = open(md, encoding="utf-8").read()
    # 去掉 markdown 链接目标与行内代码围栏，避免把路径片段当 token
    toks = sorted({t for t in TOK.findall(src)})
    doc = fitz.open(pdf)
    t = "".join(pg.get_text() for pg in doc)
    flat = re.sub(r"\s", "", t)
    miss = [k for k in toks if k not in flat]
    tofu = t.count("\ufffd")
    tot_missing += len(miss)
    flag = "OK " if (not miss and tofu == 0) else "!! "
    print(f"  {flag}{d}")
    print(f"      {doc.page_count}页 | 源 token {len(toks)} 个 | 缺失 {len(miss)} | 缺字 {tofu}")
    for m in miss[:8]:
        print(f"      ✗ {m}")
print(f"\n全部文档累计缺失: {tot_missing}")
