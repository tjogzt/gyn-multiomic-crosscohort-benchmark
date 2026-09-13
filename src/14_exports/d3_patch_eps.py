#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""d3：给主图脚本加 EPS 输出 + 字体 Type42（按既定规则：写脚本文件而非内联 heredoc）"""
import re, os
T = "/tmp/gyn_retyping"
FILES = ["r3_fig12.py", "r7_fig3.py", "r11_fig45.py", "figstyle.py"]

for f in FILES:
    p = os.path.join(T, f)
    s = open(p, encoding="utf-8").read(); n0 = s
    # 1) 补 ps/pdf fonttype=42（字体以 TrueType 矢量嵌入，非 Type3 位图）
    if "ps.fonttype" not in s:
        s = s.replace('"figure.dpi": 300, "savefig.dpi": 300, "axes.unicode_minus": False,',
                      '"figure.dpi": 300, "savefig.dpi": 300, "axes.unicode_minus": False,\n    '
                      '"pdf.fonttype": 42, "ps.fonttype": 42,')
    # 2) 在每个 fig.savefig(...pdf...) 之后补一行 .eps（matplotlib 原生矢量）
    def add_eps(m):
        line = m.group(0)
        path_pdf = re.search(r'f"\{OUT\}/([A-Za-z0-9_]+)\.pdf"', line)
        if not path_pdf: return line
        stem = path_pdf.group(1)
        indent = line[:len(line) - len(line.lstrip())]
        return line + f'\n{indent}fig.savefig(f"{{OUT}}/{stem}.eps", bbox_inches="tight", facecolor="white")'
    s = re.sub(r'^[ \t]*fig\.savefig\(f"\{OUT\}/[A-Za-z0-9_]+\.pdf".*\)$', add_eps, s, flags=re.M)
    open(p, "w", encoding="utf-8").write(s)
    print(f"  {f}: {'已改' if s != n0 else '未改'}")
