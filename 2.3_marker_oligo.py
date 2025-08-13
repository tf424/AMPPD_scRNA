import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import sys
import os


# === Configuration ===
input_path = "/rds/general/user/tf424/ephemeral/02_pmc_oligo_87_leiden.h5ad"
output_dir = "/rds/general/user/tf424/ephemeral/97/2_markers_oligo_pc50_0.14"
os.makedirs(output_dir, exist_ok=True)

adata = sc.read_h5ad(input_path)

# === Define Leiden resolution column ===
res = 0.14
resolution_key = f"leiden_res_{res:.2f}"
adata.obs["oligo_subtype"] = adata.obs[resolution_key].astype(str)

# === Marker gene detection ===
print("Running marker gene detection...")
sc.tl.rank_genes_groups(adata, groupby="oligo_subtype", method="wilcoxon", n_genes=100)

# === Export full table ===
markers_df = sc.get.rank_genes_groups_df(adata, group=None)
markers_df.to_csv(f"{output_dir}/marker_genes_{resolution_key}.csv", index=False)

# === Rank summary plot ===
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False, show=False)
plt.savefig(f"{output_dir}/marker_rank_summary_{resolution_key}.svg", bbox_inches="tight")
plt.close()

# === Top markers (5 per group) ===
top_markers = (
    markers_df.groupby("group").head(5)
    .sort_values(by=["group", "logfoldchanges"], ascending=[True, False])
)
top_genes = list(dict.fromkeys(top_markers["names"]))  # Remove duplicates while preserving order

# === Paper-derived markers (optional comparison/visualization) ===
paper_genes = [
    "MOL": ["PLP1", "MOG", "MBP"],
    "Early MOL" : ["CNP", "MAG", "PLP1", "MAL"],
    "Late MOL" : ["OPALIN", "SLC44A1", "GJB1", "HSPA2"],
    "Stress": ["IFI27", "IFI6", "IFIT1", "IFIT3", "STAT1"],
    "OPC-like": ["PDGFRA", "CSPG4", "SOX6", "VCAN"],
]
paper_genes = [g for g in paper_genes if g in adata.var_names]

# === UMAP plot ===
sc.pl.umap(adata, color=top_genes, ncols=5, show=False)
plt.savefig(f"{output_dir}/umap_top_markers_combined.svg", bbox_inches="tight")
plt.close()

# === Dotplot ===
sc.pl.dotplot(
    adata,
    var_names=top_genes,
    groupby="oligo_subtype",
    standard_scale="var",
    show=False
)
plt.savefig(f"{output_dir}/dotplot_markers_{resolution_key}.svg", bbox_inches="tight")
plt.close()

# === Paper gene expression dotplot ===
sc.pl.dotplot(
    adata,
    var_names=paper_genes,
    groupby="oligo_subtype",
    standard_scale="var",
    show=False
)
plt.savefig(f"{output_dir}/dotplot_paper_genes_{resolution_key}.svg", bbox_inches="tight")
plt.close()

# === Violin plot ===
sc.pl.violin(
    adata,
    keys=top_genes,
    groupby="oligo_subtype",
    rotation=90,
    stripplot=False,
    multi_panel=True,
    show=False
)
plt.savefig(f"{output_dir}/violin_markers_{resolution_key}.svg", bbox_inches="tight")
plt.close()

print("All done. Marker detection and plots completed.")
