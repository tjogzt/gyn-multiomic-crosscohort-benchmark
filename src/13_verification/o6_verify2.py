#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""o6：核对 Methods/Results 中来自早期报告的关键数值"""
import json, re, os
T = "/tmp/gyn_retyping"; R = "/Users/taozhu/my researches/retyping"
ALL = "".join(open(os.path.join(R, f)).read() for f in ["F1_Methods.md", "F1_Results.md"])
def norm(s):
    s = s.replace("\u2212", "-")
    return s.rstrip("0").rstrip(".") if re.fullmatch(r"-?\d+\.\d+", s) else s
N = norm(ALL)

def has(x): return norm(str(x)) in N
out = []
def ck(label, *vals):
    miss = [v for v in vals if not has(v)]
    out.append((label, miss))

# ---- 批次测试床 ----
b = json.load(open(f"{T}/eeec_batch/batch_results.json"))
def get(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return None
    return d
arms = b if isinstance(b, list) else b.get("arms", b.get("rows", []))
if isinstance(arms, list) and arms:
    for r in arms:
        nm = r.get("arm") or r.get("name") or r.get("method")
        vals = [r.get(k) for k in ["centroid_auc","centroidAUC","total_sep","totalSep","knn_mix","knnMix","mix"] if k in r]
        ck(f"批次 {nm}", *[v for v in vals if isinstance(v,(int,float))])
else:
    print("[!] batch_results.json 结构未识别:", list(b)[:8] if isinstance(b, dict) else type(b))

# ---- 平台可比性 ----
for f in ["platform/platform_results.json", "platform/results.json"]:
    p = f"{T}/{f}"
    if os.path.exists(p):
        d = json.load(open(p))
        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items(): walk(v, f"{path}.{k}")
            elif isinstance(o, (int, float)) and abs(o) <= 1.001:
                ck(f"平台{path}", round(float(o), 4))
        walk(d); break
else:
    print("[!] 无 platform 结果 JSON")

# ---- P2 纯度 ----
for f in ["pai/pai_agg.json", "pai/pai_matrix.json"]:
    p = f"{T}/{f}"
    if os.path.exists(p):
        d = json.load(open(p))
        if isinstance(d, dict) and "arms" in d:
            for r in d["arms"]:
                ck("纯度 " + str(r.get("arm") or r.get("name")),
                   *[round(r[k], 4) for k in r if k in ("dARI","delta","dari") and isinstance(r[k],(int,float))])
        break

# ---- G2 / G3 ----
for f, keys in [("g2_refined.json", ["baseline","threshold","target","ci_lo","ci_hi","n_units","n_needed"]),
                ("g3_power.json", [])]:
    d = json.load(open(f"{T}/{f}"))
    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items(): walk(v, f"{path}.{k}")
        elif isinstance(o, (int, float)):
            ck(f"{f}{path}", round(float(o), 4))
        elif isinstance(o, list):
            for i, v in enumerate(o[:40]): walk(v, f"{path}[{i}]")
    walk(d)

bad = [(l, m) for l, m in out if m]
print(f"\n核对项: {len(out)}   全命中: {len(out)-len(bad)}   有未命中: {len(bad)}")
for l, m in bad[:40]:
    print(f"  ~ {l}: 未在正文出现 {[str(x) for x in m[:6]]}")
