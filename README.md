# Day 19 — Differential Gene Expression Analysis
### 🧬 30 Days × 30 Unique Projects | Subhadip Jana

**Dataset:** GSE71562 — E. coli K-12 MG1655 (REAL data from NCBI GEO)
**Comparison:** Condition A vs Condition C
**Result:** 96 DEGs — dominated by the complete flagella/chemotaxis regulon

---

## 📊 Results Summary

| Metric | Value |
|--------|-------|
| Dataset | GSE71562 (REAL — NCBI GEO) |
| Total genes | 4,319 |
| After filter | 4,243 |
| Up in C | 71 |
| Up in A | 25 |
| **Total DEGs** | **96** |
| Key biology | Entire flagella/chemotaxis operon (fli*, che*, mot*) |
| Strongest DEG | yghW (log₂FC = −4.06) |
| Top regulator | tfaR (log₂FC = +2.06) |

---

## 🖼️ Panel Images (outputs/panels/)

| File | Description |
|------|-------------|
| P1_library_sizes.png | Library sizes for all 18 samples |
| P2_PCA.png | PCA — all 18 samples, force-directed labels |
| P3_correlation_heatmap.png | Sample correlation matrix |
| P4_MA_plot.png | MA plot — top 25 DEGs labelled |
| P5_volcano_plot.png | Volcano plot — top 25 DEGs labelled |
| P6_top20_DEGs_bar.png | Top 20 DEGs with functional annotation |
| P7_heatmap.png | Top 30 variable genes heatmap |
| P8_LFC_distribution.png | Genome-wide log₂FC distribution |
| P9_summary_table.png | Full analysis summary table |

---

## 🔬 Key Biological Finding

The **entire flagellar and chemotaxis regulon** is coordinately upregulated in Condition C:
- fliA, fliF, fliG, fliH, fliI, fliK, fliL, fliM, fliN, fliO — flagellar structure
- cheW, cheY, cheB — chemotaxis signalling
- motA, motB — flagellar motor
- tsr, tar — chemoreceptors
- tfaR (log₂FC=+2.06) — transcriptional regulator likely driving the operon

---

## 🚀 How to Run

```bash
pip install numpy pandas matplotlib seaborn scipy
python3 deg_analysis.py
```

---

## 📁 Structure

```
day19-deg-analysis/
├── deg_analysis.py          ← full pipeline
├── README.md
├── data/
│   └── counts_real.csv      ← GSE71562 real counts
└── outputs/
    ├── deg_results.csv      ← all 4,243 genes annotated
    └── panels/
        ├── P1_library_sizes.png
        ├── P2_PCA.png
        ├── P3_correlation_heatmap.png
        ├── P4_MA_plot.png
        ├── P5_volcano_plot.png
        ├── P6_top20_DEGs_bar.png
        ├── P7_heatmap.png
        ├── P8_LFC_distribution.png
        └── P9_summary_table.png
```

---

**#30DaysOfBioinformatics | Author: Subhadip Jana**
[GitHub](https://github.com/SubhadipJana1409) | [LinkedIn](https://linkedin.com/in/subhadip-jana1409)
