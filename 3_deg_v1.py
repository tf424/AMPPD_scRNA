#!/usr/bin/env python

import numpy as np
import scanpy as sc
import decoupler as dc
from scipy.io import mmread
import scipy.sparse as sp
import pandas as pd
import anndata
import os
import matplotlib.pyplot as plt
from adjustText import adjust_text
from adjustText import adjust_text
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats

# Output directory
out_dir = "/rds/general/user/tf424/ephemeral/97/3_DEG_all"
os.makedirs(out_dir, exist_ok=True)

# Load h5ad file
adata = sc.read("/rds/general/user/tf424/ephemeral/97/1.1_pmc_87_remapped.h5ad")

def plot_volcano_pvalue(results_df, lfc_thresh=0.5, pval_thresh=0.05, top_n=5, output_path=None):

    df = results_df.copy()
    df["pvalue"] = df["pvalue"].replace(0, np.nextafter(0, 1))
    df["-log10(pval)"] = -np.log10(df["pvalue"])
    df["color"] = "gray"
    df.loc[(df["pvalue"] < pval_thresh) & (df["log2FoldChange"] > lfc_thresh), "color"] = "red"
    df.loc[(df["pvalue"] < pval_thresh) & (df["log2FoldChange"] < -lfc_thresh), "color"] = "blue"

    plt.figure(figsize=(8, 6))
    plt.scatter(df["log2FoldChange"], df["-log10(pval)"], c=df["color"], s=10, alpha=0.7)
    plt.axhline(-np.log10(pval_thresh), ls="--", color="black")
    plt.axvline(lfc_thresh, ls="--", color="black")
    plt.axvline(-lfc_thresh, ls="--", color="black")
    plt.xlabel("log2FoldChange")
    plt.ylabel("-log10(pvalue)")
    plt.title("Volcano Plot")

    # Top up and down genes
    top_up_pd = df[(df["pvalue"] < pval_thresh) & (df["log2FoldChange"] > lfc_thresh)].sort_values("pvalue").head(top_n)
    top_up_hc = df[(df["pvalue"] < pval_thresh) & (df["log2FoldChange"] < -lfc_thresh)].sort_values("pvalue").head(top_n)

    # Text labels with adjustText
    texts = []
    for _, row in pd.concat([top_up_pd, top_up_hc]).iterrows():
        texts.append(
            plt.text(
                row["log2FoldChange"], row["-log10(pval)"],
                row["gene_symbol"], fontsize=7, ha='center', va='bottom'
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", lw=0.5),
        expand_points=(1.5, 1.5),
        expand_text=(1.2, 1.2),
        force_text=0.5,
        force_points=0.5,
        only_move={'points': 'y', 'text': 'xy'},
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



# Pseudobulk by participant and cell type
pdata = dc.pp.pseudobulk(
    adata,
    sample_col="participant_id",
    groups_col="cell_type",
    layer=None,
    mode="sum"
)

# === Loop over each cell type ===
celltype_summary = []

for cell_type in pdata.obs["cell_type"].unique():
    print(f"\n🔬 Processing: {cell_type}")
    sub = pdata[pdata.obs["cell_type"] == cell_type].copy()

    # Count cells and participants before filtering
    n_participants = sub.obs["participant_id"].nunique()
    n_cells = adata.obs[adata.obs["cell_type"] == cell_type].shape[0]
    print(f"🔢 {cell_type}: {n_participants} participants, {n_cells} cells before filtering")

    # Check integer counts
    if not np.all(np.equal(sub.X.data, np.floor(sub.X.data))):
        print(f"⚠️ Skipping {cell_type}: count matrix contains non-integer values.")
        celltype_summary.append((cell_type, n_participants, n_cells, 0, 0, "non-integer"))
        continue

    # Sample counts per group
    print("📊 Sample count per diagnosis:")
    print(sub.obs["diagnosis_latest"].value_counts())

    # Filtering
    dc.pp.filter_by_expr(sub, group="diagnosis_latest", min_count=10, min_total_count=15, large_n=10, min_prop=0.7)
    dc.pp.filter_by_prop(sub, min_prop=0.1, min_smpls=2)

    # Report shape after filtering
    post_n_samples = sub.shape[0]
    post_n_genes = sub.shape[1]
    print(f"🧪 {cell_type} after filtering: {post_n_samples} samples × {post_n_genes} genes")

    if post_n_samples < 4 or post_n_genes < 200:
        print(f"⚠️ Skipping {cell_type}: too few samples or genes.")
        celltype_summary.append((cell_type, n_participants, n_cells, post_n_samples, post_n_genes, "too_few_samples_or_genes"))
        continue

    # Run DESeq2
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        adata=sub,
        design_factors=["diagnosis", "sex"],
        refit_cooks=True,
        inference=inference,
    )
    dds.deseq2()

    # DEG stats
    stat_res = DeseqStats(
        dds,
        contrast=["diagnosis", "PD", "HC"],
        inference=inference
    )
    stat_res.summary()

    results_df = stat_res.results_df
    results_df["gene_symbol"] = sub.var.loc[results_df.index, "gene_symbol"].values
    cols = ["gene_symbol", "log2FoldChange", "pvalue", "padj"]
    results_df[cols].round(4).to_csv(f"{out_dir}/{cell_type}_deseq2_results.csv", index_label="EnsemblID")

    # Volcano plot
    plot_volcano_pvalue(
        results_df,
        lfc_thresh=0.5,
        pval_thresh=0.05,
        top_n=5,
        output_path=f"{out_dir}/{cell_type}_volcano.pdf"
    )

    print(f"✅ Finished: {cell_type}")
    celltype_summary.append((cell_type, n_participants, n_cells, post_n_samples, post_n_genes, "completed"))

# === Save summary across all cell types ===
summary_df = pd.DataFrame(celltype_summary, columns=["cell_type", "n_participants", "n_cells", "samples_post_filter", "genes_post_filter", "status"])
summary_df.to_csv(f"{out_dir}/celltype_summary.csv", index=False)
print("\n📄 Saved cell type summary to celltype_summary.csv")