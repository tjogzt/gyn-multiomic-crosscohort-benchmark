#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""q10：正文引用与参考文献表的一致性（无漏引、无超引）"""
import re, os
R = "/Users/taozhu/my researches/retyping"
md = open(os.path.join(R, "F1_Discussion.md")).read()
body, refsec = md.split("## 参考文献", 1)

refs = set()
for m in re.finditer(r"^(\d+)\. ", refsec, re.M):
    refs.add(int(m.group(1)))

cited = set()
for m in re.finditer(r"\[([0-9,\s]+)\]", body):
    for p in m.group(1).split(","):
        p = p.strip()
        if p.isdigit(): cited.add(int(p))

print(f"参考文献表: {len(refs)} 条  ({min(refs)}–{max(refs)})")
print(f"正文引用号: {len(cited)} 个  → {sorted(cited)}")
print()
never = sorted(refs - cited)
undef = sorted(cited - refs)
print(f"列了但未引用: {never if never else '无 ✅'}")
print(f"引用了但未列出: {undef if undef else '无 ✅'}")
print()
if never:
    for n in never:
        line = re.search(rf"^{n}\. (.*)$", refsec, re.M)
        print(f"   [{n}] {line.group(1)[:96] if line else '?'}")
