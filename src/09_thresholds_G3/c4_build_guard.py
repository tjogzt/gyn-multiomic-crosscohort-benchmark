#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""c4：全量重建 + 失败即报错（杜绝静默旧 PDF）+ 删除线语法排查"""
import os, re, subprocess, sys, fitz

BASE = "/Users/taozhu/my researches/retyping"
TMP = "/tmp/gyn_retyping/pdfbuild"; os.makedirs(TMP, exist_ok=True)
MAP = [("\ufe0f", ""), ("\u26a0", "\u25b2"), ("\u274c", "\u00d7"), ("\u2705", "\u221a"),
       ("\u23f3", "\u2026"), ("\U0001f4cc", "\u2591"), ("\u25d0", "\u25cb"),
       ("\u2b50", "\u2605"), ("\U0001f6a9", "\u2691"), ("\u2b07", "\u2193"),
       ("\U0001f7e1", "\u25cf"), ("\U0001f534", "\u25cf")]

DOCS = [f[:-3] for f in sorted(os.listdir(BASE)) if f.endswith(".md")]

print("=" * 90); print("【0】删除线语法排查（~~…~~ + 中文 会让 soul 包崩溃）"); print("=" * 90)
bad_src = []
for d in DOCS:
    s = open(os.path.join(BASE, d + ".md"), encoding="utf-8").read()
    hits = re.findall(r"~~[^~]{0,80}~~", s)
    cjk = [h for h in hits if re.search(r"[\u4e00-\u9fff]", h)]
    if cjk:
        bad_src.append(d); print(f"  !!! {d}: {len(cjk)} 处含中文删除线 {cjk[:2]}")
    elif hits:
        print(f"  ok  {d}: {len(hits)} 处纯 ASCII 删除线（可接受）")
print(f"  含中文删除线的文档: {bad_src if bad_src else '无'}")

print("\n" + "=" * 90); print("【0.5】超长 token 预警（xelatex 可在 / 处断行，不可断的长片段才会被截断）"); print("=" * 90)
LONG = re.compile(r"[A-Za-z0-9_.:/\-\[\]<>+=]{73,}")
long_hits = []
for d in DOCS:
    s = open(os.path.join(BASE, d + ".md"), encoding="utf-8").read()
    blocks = set(re.findall(r"```[^\n]*\n(.*?)\n```", s, re.S))
    inblock = "\n".join(blocks)
    for tk in sorted(set(LONG.findall(s))):
        if tk in inblock:                    # 代码块内独占一行 → 安全
            continue
        # xelatex 可在 "/" 处断行；取最长不可断片段
        segs = [x for x in tk.split("/") if x]
        longest = max(segs, key=len) if segs else tk
        if len(longest) >= 73:               # 真正会被截断的情形
            long_hits.append((d, longest))
if long_hits:
    for d, seg in long_hits:
        print(f"  !! {d}: 不可断片段 {len(seg)} 字符 → {seg[:60]}…")
    print(f"  → {len(long_hits)} 处有被截断风险（须移入代码块）")
else:
    print("  无（≥73 字符的长 token 均位于代码块内，或含可断行的 / ）")

print("\n" + "=" * 90); print("【1】全量重建（rc≠0 即中止）"); print("=" * 90)
fails = []
for d in DOCS:
    src = os.path.join(BASE, d + ".md")
    s = open(src, encoding="utf-8").read()
    for a, b in MAP: s = s.replace(a, b)
    dst = os.path.join(TMP, d + ".md"); open(dst, "w", encoding="utf-8").write(s)
    out = os.path.join(BASE, d + ".pdf")
    r = subprocess.run(["pandoc", dst, "-o", out, "--pdf-engine=xelatex",
                        "-V", "geometry:top=2.2cm,bottom=2.2cm,left=2.0cm,right=2.0cm",
                        "-V", "fontsize=10pt", "-H", "/tmp/gyn_retyping/head.tex",
                        "--toc", "--toc-depth=2"], capture_output=True, text=True)
    if r.returncode != 0:
        fails.append(d)
        print(f"  !!! {d}: rc={r.returncode}")
        for ln in r.stderr.strip().split("\n")[-6:]: print("        ", ln[:150])
    else:
        print(f"  OK  {d}")

if fails:
    print(f"\n构建失败 {len(fails)} 份: {fails}")
    sys.exit(1)

print("\n" + "=" * 90); print("【2】校验（长 token / 缺字 / 页数 / 时效）"); print("=" * 90)
TOK = re.compile(r"[A-Za-z0-9_.:/\-\[\]<>+=]{14,}")
tot = 0
for d in DOCS:
    md = os.path.join(BASE, d + ".md"); pdf = os.path.join(BASE, d + ".pdf")
    toks = sorted({t for t in TOK.findall(open(md, encoding="utf-8").read())})
    doc = fitz.open(pdf); t = "".join(p.get_text() for p in doc); flat = re.sub(r"\s", "", t)
    miss = [k for k in toks if k not in flat]; tofu = t.count("\ufffd")
    fresh = os.path.getmtime(pdf) >= os.path.getmtime(md)
    tot += len(miss)
    flag = "OK " if (not miss and tofu == 0 and fresh) else "!! "
    print(f"  {flag}{d}: {doc.page_count}页 token={len(toks)} 缺失={len(miss)} 缺字={tofu} 时效={'新' if fresh else '旧!'}")
    for m in miss[:5]: print("        ✗", m)
print(f"\n累计缺失: {tot}")
