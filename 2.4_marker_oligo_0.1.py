import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import sys
import os


# === Configuration ===
input_path = "/rds/general/user/tf424/ephemeral/02_pmc_oligo_87_leiden.h5ad"
output_dir = "/rds/general/user/tf424/ephemeral/97/02_markers_oligo_pc50"
os.makedirs(output_dir, exist_ok=True)

# === Load AnnData ===
print("🔄 Loading AnnData...")
adata = sc.read(input_path)

# === Confirm required metadata column exists ===
required_column = "barcode"

if required_column not in adata.obs.columns:
    print(f"❌ Required column '{required_column}' is missing in adata.obs. Exiting.")
    sys.exit(1)
else:
    print(f"✅ '{required_column}' column found in adata.obs.")

# === Check for raw counts without converting to dense ===
X = adata.raw.X 
if sp.issparse(X):
    data = X.data 
else:
    data = X.flatten()

# Check: all values are non-negative integers
if not np.allclose(data, np.round(data), atol=1e-6) or np.min(data) < 0:
    print("❌ .raw.X does not appear to contain raw counts (non-integer or negative values detected). Exiting.")
    sys.exit(1)
else:
    print("✅ .raw.X appears to be raw count matrix.")
    print("Raw shape:", adata.raw.X.shape)


# === Define Leiden resolution column ===
res = 0.10
resolution_key = f"leiden_res_{res:.2f}"
adata.obs["oligo_subtype"] = adata.obs[resolution_key].astype(str)

# === Marker gene detection ===
print("📈 Running marker gene detection...")
sc.tl.rank_genes_groups(adata, groupby="oligo_subtype", method="wilcoxon", n_genes=100)

# === Export full table ===
markers_df = sc.get.rank_genes_groups_df(adata, group=None)
markers_df.to_csv(f"{output_dir}/marker_genes_{resolution_key}.csv", index=False)

# === Rank summary plot ===
sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False, show=False)
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
    "OPALIN", "KLK6", "LINGO1", "HSP90AA1", "MOG",
    "PLP1", "ASPA", "CNP", "PDGFRA", "OLIG2", "SOX6"
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

# Define marker genes for each subtype
marker_dict = {
    "MOL": ["PLP1", "MOG", "MBP", "OPALIN", "SLC44A1"],
    "MFOL": ["MAL", "MOBP", "KLK6", "OPALIN"],
    "IFN-oligos": ["IFI27", "IFI6", "IFIT1", "IFIT3", "STAT1"],
    "OPC-like": ["PDGFRA", "CSPG4", "SOX6", "VCAN"],
    "COP": ["TCF7L2", "CASR", "EGR1"]
}
# Flatten list of all markers
all_markers = set(sum(marker_dict.values(), []))

# Check presence
available_markers = [gene for gene in all_markers if gene in adata.raw.var_names]
missing_markers = [gene for gene in all_markers if gene not in adata.raw.var_names]

print("✅ Found in data:", available_markers)
print("❌ Missing:", missing_markers)


sc.pl.dotplot(
    adata,
    var_names=marker_dict,
    groupby="leiden",  # or your cluster column
    use_raw=True,
    standard_scale="var",
    show=False
)
plt.savefig("marker_dotplot_subtypes.pdf", bbox_inches="tight")

print("🎉 All done. Marker detection and plots completed.")
