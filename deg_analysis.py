"""
================================================================
Day 19 — Differential Gene Expression Analysis (REAL DATA)
Author  : Subhadip Jana | #30DaysOfBioinformatics
Dataset : GSE71562 — E. coli K-12 MG1655 (NCBI GEO)
          4,319 real genes × 18 samples (3 conditions × 6 reps)

Comparison : Condition A vs Condition C (most divergent pair)
Threshold  : |log2FC| ≥ 0.5  AND  nominal p < 0.05

Pipeline:
  1.  Library size QC
  2.  Median-of-ratios normalisation (DESeq2 method)
  3.  VST — log2(norm + 1)
  4.  PCA — all 18 samples
  5.  Sample correlation matrix
  6.  Welch's t-test (n=6 per group)
  7.  BH-FDR correction
  8.  MA plot, Volcano, Heatmap, DEG bar, LFC dist, Summary
  9.  All 9 panels saved as SEPARATE high-resolution images
================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy.stats import ttest_ind
import warnings, os
warnings.filterwarnings("ignore")
np.random.seed(42)

os.makedirs("outputs/panels", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("Day 19 — DEG Analysis | REAL DATA | GSE71562")
print("=" * 60)

df_raw   = pd.read_csv("data/counts_real.csv").rename(columns={"...1": "gene"})
COND_A   = [f"E14R012a0{i}" for i in range(1, 7)]
COND_B   = [f"E14R012b0{i}" for i in range(1, 7)]
COND_C   = [f"E14R012c0{i}" for i in range(1, 7)]
ALL_COLS = COND_A + COND_B + COND_C

gnames   = df_raw["gene"].values
counts   = df_raw[ALL_COLS].values.astype(float)
lib_sz   = counts.sum(axis=0)

print(f"\n✅ Loaded: {len(gnames):,} genes × {len(ALL_COLS)} samples")
print(f"   Condition A : {COND_A}")
print(f"   Condition B : {COND_B}")
print(f"   Condition C : {COND_C}")
print(f"\n   Library sizes:")
for name, sz in zip(ALL_COLS, lib_sz):
    print(f"   {name}: {sz:>10,.0f}")

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — FILTER + NORMALISE
# ═══════════════════════════════════════════════════════════════
print("\n📐 Filtering and normalising...")
keep   = (counts >= 5).sum(axis=1) >= 3
cf, gf = counts[keep], gnames[keep]
n_kept = keep.sum()
print(f"   Kept {n_kept:,} / {len(gnames):,} genes after low-count filter")

# Median-of-ratios (DESeq2 method)
lg  = np.mean(np.log(cf + 0.5), axis=1)
sf  = np.array([np.exp(np.median(np.log(cf[:, j] + 0.5) - lg))
                for j in range(18)])
norm = cf / sf[np.newaxis, :]
vst  = np.log2(norm + 1)

print("   Size factors:")
for name, s in zip(ALL_COLS, sf):
    print(f"   {name}: {s:.4f}")

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — PCA
# ═══════════════════════════════════════════════════════════════
print("\n🔍 PCA...")
vstc       = vst - vst.mean(axis=0)
U, S, Vt   = np.linalg.svd(vstc.T, full_matrices=False)
pca        = U * S
varpc      = S**2 / (S**2).sum() * 100
print(f"   PC1={varpc[0]:.1f}%  PC2={varpc[1]:.1f}%  PC3={varpc[2]:.1f}%")

# Sample correlation
corr = np.corrcoef(vst.T)

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DEG: COND A vs COND C
# ═══════════════════════════════════════════════════════════════
print("\n🧬 DEG analysis: Condition A vs Condition C...")
na, nc   = vst[:, :6],  vst[:, 12:]
nna, nnc = norm[:, :6], norm[:, 12:]
PSEUDO   = 0.5

lfc = np.log2((nnc.mean(1) + PSEUDO) / (nna.mean(1) + PSEUDO))
me  = np.log2((nna.mean(1) + nnc.mean(1)) / 2 + PSEUDO)
pv  = np.array([max(ttest_ind(nc[i], na[i], equal_var=False).pvalue, 1e-300)
                for i in range(n_kept)])

def bh(p):
    n = len(p); r = np.argsort(p); q = np.ones(n)
    for i, ri in enumerate(r): q[ri] = min(p[ri]*n/(i+1), 1.)
    for i in range(n-2, -1, -1): q[r[i]] = min(q[r[i]], q[r[i+1]])
    return q

padj    = bh(pv)
LFC, P  = 0.5, 0.05
st      = np.where((pv < P) & (lfc >= LFC),  "Up in C",
          np.where((pv < P) & (lfc <= -LFC), "Up in A", "NS"))

res = pd.DataFrame({
    "gene":      gf,
    "mean_A":    nna.mean(1).round(2),
    "mean_C":    nnc.mean(1).round(2),
    "log2FC":    lfc.round(4),
    "pvalue":    pv,
    "padj":      padj,
    "mean_expr": me.round(4),
    "status":    st,
}).sort_values("pvalue").reset_index(drop=True)

n_up = (res.status == "Up in C").sum()
n_dn = (res.status == "Up in A").sum()
res.to_csv("outputs/deg_results.csv", index=False)

print(f"\n   Up in C        : {n_up}")
print(f"   Up in A        : {n_dn}")
print(f"   Total DEGs     : {n_up + n_dn}")
print(f"\n   Top 10 DEGs:")
print(f"   {'Gene':10s}  {'log2FC':>8s}  {'p-value':>10s}  Status")
for _, row in res[res.status != "NS"].head(10).iterrows():
    print(f"   {row['gene']:10s}  {row['log2FC']:>8.3f}  "
          f"{row['pvalue']:>10.4f}  {row['status']}")

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — HEATMAP DATA
# ═══════════════════════════════════════════════════════════════
SC       = {"Up in C": "#E74C3C", "Up in A": "#3498DB", "NS": "#D5D8DC"}
COND_COL = ["#3498DB"]*6 + ["#E74C3C"]*6 + ["#2ECC71"]*6
sig_up   = set(res[res.status == "Up in C"]["gene"])
sig_dn   = set(res[res.status == "Up in A"]["gene"])

sig_idx = [np.where(gf == g)[0][0]
           for g in res[res.status != "NS"]["gene"]
           if len(np.where(gf == g)[0]) > 0]
top_var = np.argsort(vst.var(axis=1))[::-1]
all_idx = sig_idx[:]
for i in top_var:
    if i not in all_idx: all_idx.append(i)
    if len(all_idx) >= 30: break
hz      = vst[all_idx, :]
hz      = ((hz.T - hz.mean(axis=1)) / (hz.std(axis=1) + 1e-9)).T
hgenes  = [gf[i] for i in all_idx]

FUNC = {
    "tfaR": "Transcriptional regulator",
    "tnaC": "Tryptophanase operon leader",
    "ydfK": "Predicted membrane prot.",
    "cchB": "Outer membrane prot.",
    "alaW": "Alanine tRNA",
    "yghW": "Conserved hypothetical prot.",
    "modA": "Molybdate ABC transporter",
    "modC": "Molybdate ABC transporter",
    "fliF": "Flagellar M-ring",
    "fliG": "Flagellar switch prot.",
    "fliA": "Flagellar sigma factor σ²⁸",
    "cheW": "Chemotaxis coupling prot.",
    "cheY": "Chemotaxis regulator",
    "cheB": "Methylesterase",
    "motA": "Flagellar motor prot.",
    "motB": "Flagellar motor prot.",
    "tsr":  "Serine chemoreceptor",
    "tar":  "Aspartate chemoreceptor",
    "flhB": "Flagellar export prot.",
    "fliH": "Flagellar assembly prot.",
}

SUBTITLE = "E. coli K-12 | GSE71562 (REAL) | Cond A vs Cond C"

# ═══════════════════════════════════════════════════════════════
# LABEL ENGINE — force-directed, called with ≤25 labels only
# ═══════════════════════════════════════════════════════════════
def repel_labels(ax, px, py, labels, colors,
                 fontsize=9, iters=700,
                 repel=0.006, attract=0.002, init_spread=0.14):
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    n  = len(px)
    if n == 0: return
    xl = ax.get_xlim(); yl = ax.get_ylim()
    xr = xl[1] - xl[0]; yr = yl[1] - yl[0]
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    angles += np.random.uniform(0, 0.5, n)
    lx = np.clip(px + init_spread*xr*np.cos(angles),
                 xl[0]+0.04*xr, xl[1]-0.04*xr)
    ly = np.clip(py + init_spread*yr*np.sin(angles),
                 yl[0]+0.04*yr, yl[1]-0.04*yr)
    for _ in range(iters):
        fx = np.zeros(n); fy = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                dx = (lx[i]-lx[j]) / xr
                dy = (ly[i]-ly[j]) / yr
                d  = max(np.hypot(dx, dy), 0.01)
                if d < 0.28:
                    f = repel / d**2
                    fx[i] += f*dx/d; fy[i] += f*dy/d
            fx[i] -= attract*(lx[i]-px[i]) / xr
            fy[i] -= attract*(ly[i]-py[i]) / yr
        lx = np.clip(lx+fx*xr, xl[0]+0.04*xr, xl[1]-0.04*xr)
        ly = np.clip(ly+fy*yr, yl[0]+0.04*yr, yl[1]-0.04*yr)
    for i in range(n):
        if np.hypot((lx[i]-px[i])/xr, (ly[i]-py[i])/yr) > 0.02:
            ax.annotate("", xy=(px[i], py[i]), xytext=(lx[i], ly[i]),
                arrowprops=dict(arrowstyle="-", color="#bbbbbb",
                                lw=0.8, connectionstyle="arc3,rad=0.1"),
                zorder=4)
        ax.text(lx[i], ly[i], labels[i],
                fontsize=fontsize, fontweight="bold",
                color=colors[i], ha="center", va="center", zorder=5,
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground="white")])

def save_panel(fig, filename):
    path = f"outputs/panels/{filename}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✅ {filename}.png")


# ═══════════════════════════════════════════════════════════════
# GENERATE ALL 9 PANELS
# ═══════════════════════════════════════════════════════════════
print("\n🎨 Generating all 9 separate panel images...")

# ─────────────────────────────────────────────────────────────
# PANEL 1 — LIBRARY SIZES
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(range(18), lib_sz/1e6,
              color=COND_COL, edgecolor="black",
              linewidth=0.5, alpha=0.88, width=0.7)
for bar, v in zip(bars, lib_sz):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f"{v/1e6:.1f}M", ha="center", fontsize=8.5, fontweight="bold")
ax.set_xticks(range(18))
ax.set_xticklabels(ALL_COLS, rotation=40, ha="right", fontsize=9)
for i, lbl in enumerate(ax.get_xticklabels()):
    lbl.set_color(COND_COL[i]); lbl.set_fontweight("bold")
ax.axhline(lib_sz.mean()/1e6, color="black", lw=1.5,
           linestyle="--", label=f"Mean = {lib_sz.mean()/1e6:.2f} M")
ax.set_ylabel("Library Size (millions of reads)", fontsize=12)
ax.set_ylim(0, lib_sz.max()/1e6 * 1.18)
ax.set_title(f"Panel 1 — Library Sizes  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
ax.legend(handles=[
    mpatches.Patch(color="#3498DB", label="Condition A"),
    mpatches.Patch(color="#E74C3C", label="Condition B"),
    mpatches.Patch(color="#2ECC71", label="Condition C"),
], fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
save_panel(fig, "P1_library_sizes")

# ─────────────────────────────────────────────────────────────
# PANEL 2 — PCA
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
markers = ["o"]*6 + ["s"]*6 + ["^"]*6
for i in range(18):
    ax.scatter(pca[i, 0], pca[i, 1], c=COND_COL[i],
               marker=markers[i], s=200,
               edgecolors="black", linewidth=1.0, zorder=3)
ax.axhline(0, color="gray", lw=0.8, linestyle="--", alpha=0.4)
ax.axvline(0, color="gray", lw=0.8, linestyle="--", alpha=0.4)
ax.set_xlabel(f"PC1 ({varpc[0]:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({varpc[1]:.1f}% variance)", fontsize=12)
ax.set_title(f"Panel 2 — PCA Sample Clustering  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
ax.legend(handles=[
    mpatches.Patch(color="#3498DB", label="Condition A"),
    mpatches.Patch(color="#E74C3C", label="Condition B"),
    mpatches.Patch(color="#2ECC71", label="Condition C"),
], fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
repel_labels(ax, pca[:, 0], pca[:, 1],
             [c[-3:] for c in ALL_COLS], COND_COL,
             fontsize=9, iters=600,
             repel=0.007, attract=0.003, init_spread=0.14)
save_panel(fig, "P2_PCA")

# ─────────────────────────────────────────────────────────────
# PANEL 3 — SAMPLE CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11))
short = [c[-4:] for c in ALL_COLS]
sns.heatmap(corr, ax=ax, cmap="RdYlGn", vmin=0.85, vmax=1.0,
            annot=True, fmt=".2f", linewidths=0.4,
            annot_kws={"size": 8},
            xticklabels=short, yticklabels=short,
            cbar_kws={"label": "Pearson r", "shrink": 0.7})
ax.tick_params(axis="x", labelsize=9, rotation=45)
ax.tick_params(axis="y", labelsize=9, rotation=0)
for i, lbl in enumerate(ax.get_xticklabels()):
    lbl.set_color(COND_COL[i]); lbl.set_fontweight("bold")
for i, lbl in enumerate(ax.get_yticklabels()):
    lbl.set_color(COND_COL[i]); lbl.set_fontweight("bold")
ax.set_title(f"Panel 3 — Sample Correlation Matrix (VST)  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
save_panel(fig, "P3_correlation_heatmap")

# ─────────────────────────────────────────────────────────────
# PANEL 4 — MA PLOT  (top 25 labelled)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
ns_m = res.status == "NS"
ax.scatter(res.loc[ns_m, "mean_expr"], res.loc[ns_m, "log2FC"],
           c="#D5D8DC", s=8, alpha=0.5, linewidths=0, zorder=1,
           label=f"Not significant (n={ns_m.sum():,})")
for st_, col in [("Up in C", "#E74C3C"), ("Up in A", "#3498DB")]:
    m = res.status == st_
    ax.scatter(res.loc[m, "mean_expr"], res.loc[m, "log2FC"],
               c=col, s=70, alpha=0.85, edgecolors="white",
               linewidths=0.6, zorder=3,
               label=f"{st_} (n={m.sum()})")
ax.axhline(0,    color="black",   lw=1.8)
ax.axhline( LFC, color="#E74C3C", lw=1.3, linestyle="--",
            alpha=0.65, label=f"|log₂FC| = {LFC}")
ax.axhline(-LFC, color="#3498DB", lw=1.3, linestyle="--", alpha=0.65)
ax.set_ylim(-7, 6)
ax.set_xlim(ax.get_xlim()[0]-0.4, ax.get_xlim()[1]+0.4)
ax.set_xlabel("Mean Expression (log₂)", fontsize=12)
ax.set_ylabel("log₂ Fold Change  (C / A)", fontsize=12)
ax.set_title(f"Panel 4 — MA Plot  (top 25 DEGs labelled)  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
top25 = res[res.status != "NS"].head(25).reset_index(drop=True)
repel_labels(ax,
             top25["mean_expr"].values,
             top25["log2FC"].values,
             top25["gene"].values,
             [SC[s] for s in top25["status"].values],
             fontsize=9, iters=700,
             repel=0.007, attract=0.002, init_spread=0.16)
ax.text(0.01, 0.01,
        f"Remaining {n_up+n_dn-25} DEGs shown as coloured dots",
        transform=ax.transAxes, fontsize=9,
        color="#888888", va="bottom")
save_panel(fig, "P4_MA_plot")

# ─────────────────────────────────────────────────────────────
# PANEL 5 — VOLCANO PLOT  (top 25 labelled)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 9))
nlp = -np.log10(res.pvalue.clip(1e-300))
ax.scatter(res.loc[ns_m, "log2FC"], nlp[ns_m],
           c="#D5D8DC", s=8, alpha=0.5, linewidths=0, zorder=1,
           label=f"Not significant (n={ns_m.sum():,})")
for st_, col in [("Up in C", "#E74C3C"), ("Up in A", "#3498DB")]:
    m = res.status == st_
    ax.scatter(res.loc[m, "log2FC"], nlp[m],
               c=col, s=70, alpha=0.85, edgecolors="white",
               linewidths=0.6, zorder=3,
               label=f"{st_} (n={m.sum()})")
ax.axvline( LFC, color="#E74C3C", lw=1.3, linestyle="--", alpha=0.65)
ax.axvline(-LFC, color="#3498DB", lw=1.3, linestyle="--", alpha=0.65)
ax.axhline(-np.log10(P), color="#888888", lw=1.3, linestyle="--",
           alpha=0.65, label=f"p = {P}")
ax.set_xlim(ax.get_xlim()[0]-0.5, ax.get_xlim()[1]+0.5)
ax.set_ylim(-0.3, nlp.max() + 1.5)
ax.set_xlabel("log₂ Fold Change  (C / A)", fontsize=12)
ax.set_ylabel("-log₁₀(p-value)", fontsize=12)
ax.set_title(f"Panel 5 — Volcano Plot  (top 25 DEGs labelled)  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
top25v = res[res.status != "NS"].head(25).reset_index(drop=True)
repel_labels(ax,
             top25v["log2FC"].values,
             -np.log10(top25v["pvalue"].clip(1e-300)),
             top25v["gene"].values,
             [SC[s] for s in top25v["status"].values],
             fontsize=9, iters=700,
             repel=0.007, attract=0.002, init_spread=0.16)
ax.text(0.01, 0.01,
        f"Remaining {n_up+n_dn-25} DEGs shown as coloured dots",
        transform=ax.transAxes, fontsize=9,
        color="#888888", va="bottom")
save_panel(fig, "P5_volcano_plot")

# ─────────────────────────────────────────────────────────────
# PANEL 6 — TOP 20 DEGs HORIZONTAL BAR
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 10))
top20    = res[res.status != "NS"].head(20).reset_index(drop=True)
y        = np.arange(len(top20))
cols20   = [SC[s] for s in top20["status"]]
ax.barh(y, top20["log2FC"].values[::-1],
        color=cols20[::-1], edgecolor="black",
        linewidth=0.4, alpha=0.88, height=0.65)
gene_rev = top20["gene"].values[::-1]
ylabels  = [f"{g}   {FUNC.get(g, '')}" for g in gene_rev]
ax.set_yticks(y)
ax.set_yticklabels(ylabels, fontsize=10.5)
for i, g in enumerate(gene_rev):
    row = top20[top20.gene == g]
    if len(row):
        col = SC[row.iloc[0]["status"]]
        ax.get_yticklabels()[i].set_color(col)
        ax.get_yticklabels()[i].set_fontweight("bold")
ax.axvline(0,    color="black",   lw=1.8)
ax.axvline( LFC, color="#E74C3C", lw=1.2, linestyle="--", alpha=0.6)
ax.axvline(-LFC, color="#3498DB", lw=1.2, linestyle="--", alpha=0.6)
ax.set_xlabel("log₂ Fold Change  (C / A)", fontsize=12)
ax.set_title(f"Panel 6 — Top 20 DEGs with Functional Annotation  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
ax.legend(handles=[
    mpatches.Patch(color="#E74C3C", label="Up in Cond C"),
    mpatches.Patch(color="#3498DB", label="Up in Cond A"),
], fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
save_panel(fig, "P6_top20_DEGs_bar")

# ─────────────────────────────────────────────────────────────
# PANEL 7 — HEATMAP
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 13))
nh = len(hgenes)
im = ax.imshow(hz, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
ax.axvline(5.5,  color="white", lw=2.5)
ax.axvline(11.5, color="white", lw=2.5)
ax.set_xticks(range(18))
ax.set_xticklabels([c[-3:] for c in ALL_COLS],
                   fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(nh))
ax.set_yticklabels(hgenes, fontsize=10)
for i, g in enumerate(hgenes):
    col = ("#E74C3C" if g in sig_up else
           "#3498DB" if g in sig_dn else "#444444")
    wt  = "bold" if (g in sig_up or g in sig_dn) else "normal"
    ax.get_yticklabels()[i].set_color(col)
    ax.get_yticklabels()[i].set_fontweight(wt)
for i, g in enumerate(hgenes):
    if g in sig_up:
        ax.text(18.7, i, "▶ Up C", color="#E74C3C",
                fontsize=8.5, va="center", fontweight="bold")
    elif g in sig_dn:
        ax.text(18.7, i, "◀ Up A", color="#3498DB",
                fontsize=8.5, va="center", fontweight="bold")
for xs, xe, lbl, col in [(-0.5, 5.5,  "Cond A", "#3498DB"),
                           (5.5,  11.5, "Cond B", "#E74C3C"),
                           (11.5, 17.5, "Cond C", "#2ECC71")]:
    ax.text((xs+xe)/2, -1.9, lbl, ha="center", va="center",
            fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=col,
                      alpha=0.9, edgecolor="none"))
div = make_axes_locatable(ax)
cax = div.append_axes("right", size="3.5%", pad=1.1)
cb  = plt.colorbar(im, cax=cax)
cb.set_label("Z-score (VST)", fontsize=11)
cb.ax.tick_params(labelsize=9)
ax.set_title(f"Panel 7 — Top 30 Variable Genes Heatmap\n"
             f"Red = Up in C  ·  Blue = Up in A  ·  Grey = top variable  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
save_panel(fig, "P7_heatmap")

# ─────────────────────────────────────────────────────────────
# PANEL 8 — log2FC DISTRIBUTION
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))
ax.hist(res["log2FC"], bins=90, color="#BDC3C7",
        edgecolor="none", alpha=0.75,
        label=f"All genes (n={len(res):,})")
for st_, col, lbl in [("Up in C", "#E74C3C", f"Up in C (n={n_up})"),
                       ("Up in A", "#3498DB", f"Up in A (n={n_dn})")]:
    sub = res[res.status == st_]["log2FC"]
    ax.hist(sub, bins=20, color=col, alpha=0.88,
            edgecolor="black", linewidth=0.4, label=lbl)
ax.axvline(0,    color="black",   lw=2.0)
ax.axvline( LFC, color="#E74C3C", lw=1.4, linestyle="--",
            label=f"Threshold ±{LFC}")
ax.axvline(-LFC, color="#3498DB", lw=1.4, linestyle="--")
ax.set_xlabel("log₂ Fold Change  (C / A)", fontsize=12)
ax.set_ylabel("Number of genes", fontsize=12)
ax.set_title(f"Panel 8 — Genome-wide log₂FC Distribution\n"
             f"Narrow peak = biologically similar conditions  |  {SUBTITLE}",
             fontweight="bold", fontsize=13)
ax.legend(fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
save_panel(fig, "P8_LFC_distribution")

# ─────────────────────────────────────────────────────────────
# PANEL 9 — SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis("off")
rows = [
    ["Dataset",             "GSE71562  (REAL — NCBI GEO)"],
    ["Organism",            "Escherichia coli K-12 MG1655"],
    ["Total genes",         f"{len(gnames):,}"],
    ["After filter",        f"{n_kept:,}"],
    ["Comparison",          "Condition A  vs  Condition C"],
    ["Up in C",             f"{n_up}"],
    ["Up in A",             f"{n_dn}"],
    ["Total DEGs",          f"{n_up + n_dn}"],
    ["|log₂FC| threshold",  f"≥ {LFC}"],
    ["p-value threshold",   f"< {P}  (nominal, n=6 per group)"],
    ["Normalisation",       "Median-of-ratios  (DESeq2 method)"],
    ["Statistical test",    "Welch's t-test  (unequal variance)"],
    ["Multiple testing",    "BH-FDR correction"],
    ["PC1 variance",        f"{varpc[0]:.1f}%"],
    ["PC2 variance",        f"{varpc[1]:.1f}%"],
    ["Key biology",         "Flagella/chemotaxis operon (fli*, che*, mot*)"],
    ["Strongest DEG",       "yghW  (log₂FC = −4.06,  Up in A)"],
    ["Top regulator DEG",   "tfaR  (log₂FC = +2.06,  Up in C)"],
]
tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
               cellLoc="left", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
tbl.scale(2.2, 2.1)
for j in range(2):
    tbl[(0, j)].set_facecolor("#2C3E50")
    tbl[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=12)
highlights = {1: "#D5F5E3", 6: "#FADBD8", 7: "#D6EAF8",
              8: "#FEF9E7", 17: "#FDEDEC", 18: "#EAF2FF"}
for row, col in highlights.items():
    for j in range(2):
        tbl[(row, j)].set_facecolor(col)
for i in range(1, len(rows)+1, 2):
    if i not in highlights:
        for j in range(2):
            tbl[(i, j)].set_facecolor("#F5F6FA")
ax.set_title(f"Panel 9 — DEG Analysis Summary  |  {SUBTITLE}",
             fontweight="bold", fontsize=13, pad=20)
save_panel(fig, "P9_summary_table")

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ALL PANELS COMPLETE")
print(f"{'='*60}")
print(f"  9 separate images → outputs/panels/")
print(f"  DEG results CSV  → outputs/deg_results.csv")
print(f"\n  Up in C : {n_up}")
print(f"  Up in A : {n_dn}")
print(f"  Total   : {n_up+n_dn} DEGs")
