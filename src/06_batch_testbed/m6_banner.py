#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""m6：给 v0.2/v0.3 加"已被 v1.0 取代"横幅"""
import os
BASE = "/Users/taozhu/my researches/retyping"
BANNER = "> ⚠️ **本文件已被《方向F_预注册计划_v1.0（已冻结）》取代** —— v1.0 合并了 F-32~F-58 并**冻结了 G2 阈值**。请以 v1.0 为准，本文件仅作版本留痕。"

for f in ["方向F_预注册计划_v0.1.md", "方向F_预注册计划_v0.2.md", "方向F_预注册计划_v0.3.md"]:
    p = os.path.join(BASE, f)
    s = open(p, encoding="utf-8").read()
    if "已被《方向F_预注册计划_v1.0" in s:
        print(f"{f}: 已有横幅，跳过")
        continue
    lines = s.split("\n")
    pos = None
    for i, l in enumerate(lines):
        if l.startswith("> **v") and "变更" in l:
            pos = i + 1
            break
    if pos is None:
        pos = 1
    lines.insert(pos, BANNER)
    open(p, "w", encoding="utf-8").write("\n".join(lines))
    print(f"{f}: 已加横幅（插入行 {pos+1}）")

for f in ["方向F_预注册计划_v0.1.md", "方向F_预注册计划_v0.2.md", "方向F_预注册计划_v0.3.md", "方向F_预注册计划_v1.0.md"]:
    p = os.path.join(BASE, f)
    s = open(p, encoding="utf-8").read()
    has = "v1.0（**已冻结**）" in s or "已被《方向F_预注册计划_v1.0" in s or "已被 v0.2 取代" in s
    print(f"  {f}: {'✓ 状态标注齐' if has else '✗ 无状态标注'}")
