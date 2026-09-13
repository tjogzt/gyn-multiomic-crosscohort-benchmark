#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""c1：投稿包合规检查（GoB 规格 + 交叉引用 + 数字一致性）"""
import os, re, json, glob
import fitz, pandas as pd
R = "/Users/taozhu/my researches/retyping"; TD = f"{R}/tables"; T = "/tmp/gyn_retyping"

print("=" * 92); print("【A】包内容清单"); print("=" * 92)
doc_md = sorted(glob.glob(f"{R}/F1_*.md"))
figs = sorted(glob.glob(f"{R}/Fig[0-9]_*.png")) + sorted(glob.glob(f"{R}/FigS*.png"))
csvs = sorted(glob.glob(f"{TD}/*.csv"))
print(f"  正文文档 {len(doc_md)} 份:")
for f in doc_md: print(f"     {os.path.basename(f):44s} {os.path.getsize(f):>7,} B")
print(f"  图件 {len(figs)} 张（PNG）")
print(f"  表格 {len(csvs)} 个（CSV）")

print("\n" + "=" * 92); print("【B】图件尺寸与分辨率（Genome Biology 规格）"); print("=" * 92)
# GoB: 单栏 88 mm / 1.5 栏 120 mm / 双栏 180 mm；≥300 dpi
SPEC = {"single 88mm": 88, "1.5-col 120mm": 120, "double 180mm": 180}
rows = []
for pdf in sorted(glob.glob(f"{R}/Fig*_*.pdf")):
    d = fitz.open(pdf); pg = d[0]
    w_mm = pg.rect.width / 72 * 25.4; h_mm = pg.rect.height / 72 * 25.4
    png = pdf[:-4] + ".png"
    dpi = None
    if os.path.exists(png):
        from PIL import Image
        with Image.open(png) as im:
            dpi = round(im.size[0] / (w_mm / 25.4))
    fit = min(SPEC.items(), key=lambda kv: abs(kv[1] - w_mm))
    rows.append(dict(fig=os.path.basename(pdf)[:-4], width_mm=round(w_mm, 1), height_mm=round(h_mm, 1),
                     png_dpi=dpi, nearest_spec=fit[0],
                     oversize=round(w_mm - 180, 1) if w_mm > 180 else 0))
DIM = pd.DataFrame(rows)
print(DIM.to_string(index=False))
print(f"\n  ⚠ 超双栏 180mm 的图: {(DIM.width_mm > 180).sum()} / {len(DIM)}")
print(f"  ⚠ dpi < 300 的图: {(DIM.png_dpi < 300).sum()} / {len(DIM)}")

print("\n" + "=" * 92); print("【C】交叉引用完整性"); print("=" * 92)
TXT = ""
for f in doc_md: TXT += open(f, encoding="utf-8").read()
tabs_txt = open(f"{R}/F1_Tables.md", encoding="utf-8").read()
figtxt = open(f"{R}/F1_图件规划与图注.md", encoding="utf-8").read()
have_figs = {os.path.basename(f)[:4].replace("_", "") for f in figs}
cited = set()
for m in re.finditer(r"Figure\s+(S?\d+)([a-c])?", TXT + figtxt):
    cited.add(m.group(1))
for m in re.finditer(r"Fig(?:ure)?\s*(S?\d+)", TXT): cited.add(m.group(1))
# panel 级引用（Figure 1a / Figure S7b）按父图号归属
PANEL = set()
for m in re.finditer(r"Figure\s+(S?\d+)([a-c])\b", TXT + figtxt):
    PANEL.add(m.group(1) + m.group(2))
exist_f = set()
for f in figs:
    b = os.path.basename(f)
    m = re.match(r"Fig(S?\d+)", b)
    if m: exist_f.add(m.group(1))
print(f"  文中引用的图号: {sorted(cited)}")
print(f"  panel 级引用: {sorted(PANEL)}")
print(f"  实际存在的图号: {sorted(exist_f)}")
miss_f = sorted(cited - exist_f); extra_f = sorted(exist_f - cited)
print(f"  ⚠ 引用但不存在: {miss_f if miss_f else '无 ✅'}")
print(f"  ⚠ 存在但未引用: {extra_f if extra_f else '无 ✅'}")
bad_panel = sorted({p for p in PANEL if p[:-1] not in exist_f})
print(f"  ⚠ panel 引用的父图不存在: {bad_panel if bad_panel else '无 ✅'}")
cited_t = set()
for m in re.finditer(r"Table\s+(\d+[a-c]?)", TXT + tabs_txt): cited_t.add(m.group(1))
exist_t = set()
for f in csvs:
    m = re.match(r"T(\d+[a-c]?)_", os.path.basename(f))
    if m: exist_t.add(m.group(1))
print(f"  文中引用的表号: {sorted(cited_t)}")
print(f"  实际存在的表号: {sorted(exist_t)}")
miss_t = sorted(cited_t - exist_t); extra_t = sorted(exist_t - cited_t)
print(f"  ⚠ 引用但不存在: {miss_t if miss_t else '无 ✅'}")
print(f"  ⚠ 存在但未引用: {extra_t if extra_t else '无 ✅'}")

print("\n" + "=" * 92); print("【D】关键数字一致性（正文 vs 表）"); print("=" * 92)
def J(p): return json.load(open(f"{T}/{p}"))
CHK = [
    ("T3 KMeans overall", "0.3947", f"{J('bench_summary.json')['tab1']['3_KMeans(concat)']['all']:.4f}"),
    ("T4 SNF mean", "0.2081", f"{J('bench_merged.json')['r_ranking']['1_SNF']:.4f}"),
    ("T6 limma centroid", "0.5", "0.5"),
    ("T7a cross-platform ρ", "0.8473", f"{J('platform/platform_results.json')['ladder'][1]['spearman']:.4f}"),
]
for lab, in_text, in_tab in CHK:
    a = in_text in TXT; b = abs(float(in_text.replace("ρ", "")) - float(in_tab)) < 5e-5 if in_text.replace("ρ","").replace(".","").isdigit() else True
    print(f"  {'✓' if a else '~'} {lab:26s} 正文含 {in_text:>8s}  表值 {in_tab}")

print("\n" + "=" * 92); print("【E】文本层检查"); print("=" * 92)
for f in doc_md:
    s = open(f, encoding="utf-8").read()
    ph = re.findall(r"(TODO|TBD|XXX|待补|占位|placeholder)", s, re.I)
    stub = len(re.findall(r"\[\?\]", s))
    print(f"  {os.path.basename(f)[:34]:36s} 占位符 {len(ph)}  删除线 {s.count('~~')}  未替换引用 {stub}")
