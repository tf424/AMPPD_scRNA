#!/usr/bin/env python

import os
import sys
import scipy.sparse as sp
import scanpy as sc
import pandas as pd
import anndata
import numpy as np
import harmonypy as hm
import seaborn as sns
import matplotlib.pyplot as plt

# === Helper: Track participant numbers ===
def check_participant_counts(adata, label):
    print(f"\n[{label}] Unique participants: {adata.obs['participant_id'].nunique()}")
    print(adata.obs['participant_id'].value_counts())

# === Configuration ===
input_path = "/rds/general/user/tf424/ephemeral/1_pmc_oligo_87_pca.h5ad"
output_dir = "/rds/general/user/tf424/ephemeral/97/2_harmony_oligo"
os.makedirs(output_dir, exist_ok=True)

# === Load AnnData ===
print("Loading AnnData...")
adata = sc.read_h5ad(input_path)
print("adata.X type:", type(adata.X))
print("Layers available:", adata.layers.keys())

# === Confirm required metadata column exists ===
required_column = "barcode"

if required_column not in adata.obs.columns:
    print(f"❌ Required column '{required_column}' is missing in adata.obs. Exiting.")
    sys.exit(1)
else:
    print(f"✅ '{required_column}' column found in adata.obs.")

# === Check for raw counts without converting to dense ===
def check_raw_counts(adata, expected_shape=(220600, 40496)):
    """
    Verifies that adata.raw.X is a valid raw count matrix.
    
    Checks:
    - .raw exists
    - shape matches expected
    - all values are non-negative integers (tolerant of float storage)

    Exits if any check fails.
    """
    if adata.raw is None:
        print("❌ adata.raw is None. Exiting.")
        sys.exit(1)

    X = adata.raw.X

    # Shape check
    if X.shape != expected_shape:
        print(f"❌ .raw.X shape mismatch. Expected {expected_shape}, got {X.shape}")
        sys.exit(1)

    # Extract non-zero values efficiently
    data = X.data if sp.issparse(X) else X.flatten()

    # Check for non-negative integers
    if not np.allclose(data, np.round(data), atol=1e-6) or np.min(data) < 0:
        print("❌ .raw.X does not appear to contain raw counts (non-integer or negative values detected). Exiting.")
        sys.exit(1)

    # If all checks pass
    print(f"✅ .raw.X passed integrity check. Shape: {X.shape}")

check_raw_counts(adata)

# drop all mitochondrial genes from the feature set
mt_mask = adata.var_names.str.startswith('MT-')
adata = adata[:, ~mt_mask].copy()

# === PCA & Harmony ===
adata.obsm["X_pca_50"] = adata.obsm["X_pca"].copy()

sc.tl.pca(adata, svd_solver='arpack', n_comps=20)
adata.obsm["X_pca_20"] = adata.obsm["X_pca"].copy()

check_raw_counts(adata)
# Run Harmony
print("Running Harmony...")
ho = hm.run_harmony(adata.obsm["X_pca_20"], adata.obs, vars_use=["participant_id", "sex"])
adata.obsm["X_pca_harmony_20"] = ho.Z_corr.T

sc.pp.neighbors(adata, use_rep="X_pca_harmony_20")
sc.tl.umap(adata)

adata.write("/rds/general/user/tf424/ephemeral/02_pmc_oligo_87_harmony_20.h5ad")

check_raw_counts(adata)
res = 0.24
key = "leiden_res_0.24"
sc.tl.leiden(adata, resolution=res, key_added=key, flavor="igraph")
adata.write("/rds/general/user/tf424/ephemeral/02_pca_20_res_0.24.h5ad")
check_raw_counts(adata)

# === Configuration ===
out_dir = "/rds/general/user/tf424/ephemeral/0_results/deg_oligo_pc20_res0.24"
os.makedirs(output_dir, exist_ok=True)

# === DEG ===
import decoupler as dc
from adjustText import adjust_text
from adjustText import adjust_text
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats

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


# Extract full raw data as a separate AnnData object
adata_full = adata.raw.to_adata()

# Now you can safely store raw counts as a layer
adata_full.layers["counts"] = adata_full.X.copy()

# Check for negatives in raw counts layer
if (adata_full.layers["counts"] < 0).sum() > 0:
    raise ValueError("❌ Negative values found in adata.layers['counts'].")

# Pseudobulk by participant and Leiden cluster using raw counts
pdata = dc.pp.pseudobulk(
    adata_full,
    sample_col="participant_id",
    groups_col="leiden_res_0.24",
    layer="counts", 
    mode="sum"
)

# === Add metacell count checks ===

# Metacells per participant
meta_per_participant = pdata.obs["participant_id"].value_counts()
meta_per_participant.to_csv(f"{out_dir}/metacell_count_per_participant.csv")
print("📊 Saved metacell counts per participant.")

# Metacells per Leiden cluster
meta_per_cluster = pdata.obs["leiden_res_0.24"].value_counts()
meta_per_cluster.to_csv(f"{out_dir}/metacell_count_per_cluster.csv")
print("📊 Saved metacell counts per Leiden cluster.")

# Cross-tab: participant × cluster
meta_table = pd.crosstab(
    pdata.obs["participant_id"],
    pdata.obs["leiden_res_0.24"]
)
meta_table.to_csv(f"{out_dir}/metacell_table_participant_by_cluster.csv")
print("📊 Saved metacell table (participants × clusters).")

# === Loop over each Leiden cluster ===
cluster_summary = []

for cluster in pdata.obs["leiden_res_0.24"].unique():
    print(f"\n🔬 Processing cluster: {cluster}")
    sub = pdata[pdata.obs["leiden_res_0.24"] == cluster].copy()

    # Count cells and participants before filtering
    n_participants = sub.obs["participant_id"].nunique()
    n_cells = adata.obs[adata.obs["leiden_res_0.24"] == cluster].shape[0]
    print(f"🔢 Cluster {cluster}: {n_participants} participants, {n_cells} cells before filtering")

    # Extract count matrix from layer
    X = sub.layers["counts"]
    data = X.data if sp.issparse(X) else X.flatten()
    
    if not np.allclose(data, np.round(data), atol=1e-6):
        print(f"⚠️ Skipping cluster {cluster}: count matrix contains non-integer values.")
        cluster_summary.append((cluster, n_participants, n_cells, 0, 0, "non-integer"))
        continue

    print("📊 Sample count per diagnosis:")
    print(sub.obs["diagnosis_latest"].value_counts())

    dc.pp.filter_by_expr(sub, group="diagnosis_latest", min_count=10, min_total_count=15, large_n=10, min_prop=0.7)
    dc.pp.filter_by_prop(sub, min_prop=0.1, min_smpls=2)

    post_n_samples = sub.shape[0]
    post_n_genes = sub.shape[1]
    print(f"🧪 Cluster {cluster} after filtering: {post_n_samples} samples × {post_n_genes} genes")

    if post_n_samples < 4 or post_n_genes < 200:
        print(f"⚠️ Skipping cluster {cluster}: too few samples or genes.")
        cluster_summary.append((cluster, n_participants, n_cells, post_n_samples, post_n_genes, "too_few_samples_or_genes"))
        continue

    # Run DESeq2
    inference = DefaultInference(n_cpus=8)
    dds = DeseqDataSet(
        adata=sub,
        design_factors=["diagnosis", "sex"],
        refit_cooks=True,
        inference=inference,
        layer="counts"
    )
    dds.deseq2()

    stat_res = DeseqStats(
        dds,
        contrast=["diagnosis", "PD", "HC"],
        inference=inference
    )
    stat_res.summary()

    results_df = stat_res.results_df
    results_df["gene_symbol"] = sub.var.loc[results_df.index, "gene_symbol"].values
    cols = ["gene_symbol", "log2FoldChange", "pvalue", "padj"]
    results_df[cols].round(4).to_csv(f"{out_dir}/cluster_{cluster}_deseq2_results.csv", index_label="EnsemblID")

    plot_volcano_pvalue(
        results_df,
        lfc_thresh=0.5,
        pval_thresh=0.05,
        top_n=5,
        output_path=f"{out_dir}/cluster_{cluster}_volcano.pdf"
    )

    print(f"✅ Finished cluster: {cluster}")
    cluster_summary.append((cluster, n_participants, n_cells, post_n_samples, post_n_genes, "completed"))

# === Save summary across clusters ===
summary_df = pd.DataFrame(
    cluster_summary,
    columns=["leiden_cluster", "n_participants", "n_cells", "samples_post_filter", "genes_post_filter", "status"]
)
summary_df.to_csv(f"{out_dir}/leiden_cluster_summary.csv", index=False)
print("\n📄 Saved Leiden cluster summary to leiden_cluster_summary.csv")

print("✅ Complete.")
