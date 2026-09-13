#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n2：纯度诊断——纯度占多少方差？是否构成队列混淆轴？"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

B = "/Volumes/tjogzt4T/data"
H = "/tmp/gyn_retyping/hcsv"
OUT = "/tmp/gyn_retyping/purity"
os.makedirs(OUT, exist_ok=True)

# ---------- 纯度 ----------
ind_meta = pd.read_excel(f"{B}/cptac/CPTAC_UCEC_independent_LinkedOmics/UCEC_confirmatory_meta_table_v3.0.xlsx")
dc = pd.read_csv(f"{B}/cptac/CPTAC_UCEC_Discovery_LinkedOmics/HS_CPTAC_UCEC_CLI.txt",
                 sep="\t", low_memory=False, encoding="utf-8", encoding_errors="replace")
# 键：ind 用 Case_id；dis 用 Proteomics_Participant_ID（均与层矩阵列名一致）
_i = pd.DataFrame({"id": ind_meta["Case_id"].astype(str),
                   "p": pd.to_numeric(ind_meta["ABSOLUTE_tumor_purity"], errors="coerce")}).dropna()
_i = _i[_i["p"] > 0].drop_duplicates("id").set_index("id")["p"]
_d = pd.DataFrame({"id": dc["Proteomics_Participant_ID"].astype(str),
                   "p": pd.to_numeric(dc["Purity_Cancer"], errors="coerce"),
                   "tn": dc["Proteomics_Tumor_Normal"].astype(str)})
_d = _d[(_d["tn"] == "Tumor") & _d["p"].notna()].drop_duplicates("id").set_index("id")["p"]
pur_ind, pur_dis = _i, _d
print(f"纯度可得: ind {len(pur_ind)} 例 (ABSOLUTE) | dis {len(pur_dis)} 例 (Purity_Cancer)")
print(f"  ind: 均值 {pur_ind.mean():.4f} 范围 [{pur_ind.min():.3f},{pur_ind.max():.3f}]")
print(f"  dis: 均值 {pur_dis.mean():.4f} 范围 [{pur_dis.min():.3f},{pur_dis.max():.3f}]")
from scipy.stats import mannwhitneyu
u, p = mannwhitneyu(pur_ind.values, pur_dis.values)
print(f"  两队列纯度分布差异: Mann-Whitney p = {p:.3g}  → {'差异显著' if p<0.05 else '无显著差异'}")

# ---------- 层矩阵 ----------
def load(coh, lay):
    p = f"{H}/{coh}__{lay}.csv.gz"
    if not os.path.exists(p): return None
    M = pd.read_csv(p, index_col=0)
    M.columns = [str(c) for c in M.columns]
    return M

DIAG = []
for coh, pur, tag in [("ind", pur_ind, "ABSOLUTE"), ("dis", pur_dis, "Tumor_purity")]:
    for lay in ["mRNA", "CNA", "protein", "meth", "miRNA"]:
        M = load(coh, lay)
        if M is None: continue
        common = [s for s in M.columns if s in pur.index]
        if len(common) < 25: continue
        X = M[common].T
        y = pur.reindex(common).values.astype(float)
        X = X.loc[:, X.notna().mean() > 0.9].fillna(X.median())
        Xs = (X - X.mean()) / X.std(ddof=1).replace(0, 1)
        # 逐基因与纯度的相关
        rs = np.array([pearsonr(Xs.iloc[:, j].values, y)[0] for j in range(Xs.shape[1])])
        rs = rs[np.isfinite(rs)]
        frac_hi = float(np.mean(np.abs(rs) > 0.3))
        # 前 10 个 PC 与纯度
        npc = min(10, Xs.shape[0]-1, Xs.shape[1])
        P = PCA(n_components=npc, random_state=49).fit(Xs.values)
        Z = P.transform(Xs.values)
        pc_r = [abs(pearsonr(Z[:, i], y)[0]) for i in range(npc)]
        # 纯度可解释的总方差（用 |r| 度量线性关联强度）
        r2_tot = float(np.mean(rs**2))
        DIAG.append({"cohort": coh, "layer": lay, "n": len(common), "n_gene": Xs.shape[1],
                     "mean_r2_purity": round(r2_tot, 4),
                     "frac_gene_absr_gt_0.3": round(frac_hi, 4),
                     "max_pc_r": round(float(max(pc_r)), 4),
                     "pc1_r": round(pc_r[0], 4), "pc2_r": round(pc_r[1], 4) if npc > 1 else None})
        print(f"  {coh}/{lay:8s} n={len(common):3d} 基因={Xs.shape[1]:5d} | "
              f"平均 r²(纯度)={r2_tot:.4f} | |r|>0.3 的基因 {frac_hi*100:5.1f}% | "
              f"PC1|r|={pc_r[0]:.3f} PC2|r|={pc_r[1]:.3f}" if npc > 1 else "")

D = pd.DataFrame(DIAG)
D.to_csv(f"{OUT}/purity_diag.csv", index=False)
print(f"\n平均 r²(纯度) 跨层: {D['mean_r2_purity'].mean():.4f} | 最高层: {D.loc[D['mean_r2_purity'].idxmax(),'layer']}")
print(f"最大 PC|r| 跨层中位: {D['max_pc_r'].median():.4f}")
json.dump(D.to_dict(orient="records"), open(f"{OUT}/purity_diag.json", "w"), ensure_ascii=False, indent=1)
print("\n已落盘 purity_diag.json")
