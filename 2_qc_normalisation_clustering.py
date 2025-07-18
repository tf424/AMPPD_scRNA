#!/usr/bin/env python

import scanpy as sc
import pandas as pd
import harmonypy as hm
import anndata
import matplotlib.pyplot as plt
import seaborn as sns
import scrublet as scr
import numpy as np
from sklearn.metrics import silhouette_score
import os

# === Configuration ===
input_path = "/rds/general/user/tf424/home/subset_pmc_20.h5ad"
output_dir = "/rds/general/user/tf424/home/0.Output"
os.makedirs(output_dir, exist_ok=True)

# === Add QC metrics BEFORE filtering ===
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], percent_top=None, log1p=False, inplace=True)

# Save number of cells BEFORE filtering
n_cells_before = adata.n_obs

# === Violin plot BEFORE filtering ===
sc.pl.violin(
    adata,
    keys=["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"],
    jitter=0.4,
    multi_panel=True,
    show=False
)
plt.savefig(f"{output_dir}/vln_qc_before_filtering.pdf", bbox_inches='tight', dpi=300)

# === Apply filtering thresholds ===
adata = adata[
    (adata.obs["pct_counts_mt"] < 10) &
    (adata.obs["pct_counts_ribo"] < 50)
].copy()
check_participant_counts(adata, "After mito/ribo filtering")

# Save intermediate number of cells after mito/ribo filtering
n_cells_after_qc = adata.n_obs

# Apply standard filters
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)
print(adata.obs["n_genes_by_counts"].describe())
check_participant_counts(adata, "After standard filters")

# Final cell count
n_cells_final = adata.n_obs

# === Print summary ===
print(f"Number of cells before filtering: {n_cells_before}")
print(f"After mito/ribo QC: {n_cells_after_qc} (removed {n_cells_before - n_cells_after_qc})")
print(f"After standard filters: {n_cells_final} (removed {n_cells_after_qc - n_cells_final})")

# === Recalculate QC metrics after filtering ===
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo"], percent_top=None, log1p=False, inplace=True)

# === Violin plot AFTER filtering ===
sc.pl.violin(
    adata,
    keys=["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo"],
    jitter=0.4,
    multi_panel=True,
    show=False
)
plt.savefig(f"{output_dir}/vln_qc_after_filtering.pdf", bbox_inches='tight', dpi=300)

# === Save ===
adata.write(f"{output_dir}/adata_pmc_filtered.h5ad")

# === Doublet detection using Scrublet ===
counts_matrix = adata.X if not isinstance(adata.X, np.matrix) else np.array(adata.X)
scrub = scr.Scrublet(counts_matrix)
doublet_scores, predicted_doublets = scrub.scrub_doublets()

# Add to obs
adata.obs["doublet_score"] = doublet_scores
adata.obs["predicted_doublet"] = predicted_doublets

# Remove predicted doublets
adata = adata[~adata.obs["predicted_doublet"]].copy()
check_participant_counts(adata, "After doublet removal")

# === Visualize Scrublet results ===
plt.figure(figsize=(6, 4))
sns.histplot(adata.obs["doublet_score"], bins=50, kde=True)
plt.xlabel("Doublet Score")
plt.ylabel("Cell Count")
plt.title("Scrublet Doublet Score Distribution")
plt.tight_layout()
plt.savefig(f"{output_dir}/doublet_score_histogram.pdf", dpi=300)

# Save after QC and doublet removal
adata.write(f"{output_dir}/10_pmc_qc.h5ad")

# === Normalize and log-transform ===
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# === HVG Selection ===
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# === Subset to HVGs and scale ===
adata = adata[:, adata.var.highly_variable]
sc.pp.scale(adata, max_value=10)

# === PCA ===
sc.tl.pca(adata, svd_solver='arpack')

# === Elbow plot ===
plt.figure()
plt.plot(
    range(1, len(adata.uns["pca"]["variance_ratio"]) + 1),
    adata.uns["pca"]["variance_ratio"],
    marker="o"
)
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained Ratio")
plt.title("PCA Elbow Plot")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/pca_elbow_plot.pdf", dpi=300)

# === UMAP without Harmony (baseline) ===
sc.pp.neighbors(adata, use_rep="X_pca")
sc.tl.umap(adata)
adata.obsm["X_umap_noharmony"] = adata.obsm["X_umap"].copy()

# Save UMAP (no Harmony)
sc.pl.umap(adata, color="participant_id", title="No Harmony", show=False)
plt.savefig(f"{output_dir}/umap_noharmony_participant.pdf", bbox_inches='tight', dpi=300)

# === UMAP colored by doublet score ===
sc.pl.umap(
    adata, color="doublet_score", cmap="viridis", title="Doublet Score", show=False
)
plt.savefig(f"{output_dir}/umap_doublet_score.pdf", dpi=300, bbox_inches='tight')

# === UMAP with predicted doublets ===
adata.obs["doublet_label"] = adata.obs["predicted_doublet"].map({True: "Doublet", False: "Singlet"})
sc.pl.umap(
    adata, color="doublet_label", palette=["lightgrey", "red"], title="Predicted Doublets", show=False
)
plt.savefig(f"{output_dir}/umap_predicted_doublets.pdf", dpi=300, bbox_inches='tight')

# === Harmony Batch Correction ===
ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, ["participant_id"])
adata.obsm["X_pca_harmony"] = ho.Z_corr.T

# === UMAP with Harmony ===
sc.pp.neighbors(adata, use_rep="X_pca_harmony")
sc.tl.umap(adata)
adata.obsm["X_umap_harmony"] = adata.obsm["X_umap"].copy()

# Save UMAP (with Harmony)
sc.pl.umap(adata, color="participant_id", title="With Harmony", show=False)
plt.savefig(f"{output_dir}/umap_harmony_participant.pdf", bbox_inches='tight', dpi=300)

# === Compare UMAPs Side-by-Side ===
sc.pl.embedding(adata, basis="X_umap_noharmony", color="participant_id", title="No Harmony", show=False)
plt.savefig(f"{output_dir}/umap_side_noharmony.pdf", dpi=300, bbox_inches='tight')

sc.pl.embedding(adata, basis="X_umap_harmony", color="participant_id", title="With Harmony", show=False)
plt.savefig(f"{output_dir}/umap_side_harmony.pdf", dpi=300, bbox_inches='tight')

# === Silhouette Score Comparison ===
score_noharmony = silhouette_score(adata.obsm["X_pca"], adata.obs["participant_id"])
score_harmony = silhouette_score(adata.obsm["X_pca_harmony"], adata.obs["participant_id"])

print(f"Silhouette Score (No Harmony): {score_noharmony:.4f}")
print(f"Silhouette Score (With Harmony): {score_harmony:.4f}")

# Save scores to file
with open(f"{output_dir}/silhouette_scores.txt", "w") as f:
    f.write(f"Silhouette Score (No Harmony): {score_noharmony:.4f}\n")
    f.write(f"Silhouette Score (With Harmony): {score_harmony:.4f}\n")

# === Leiden clustering ===
for res in [0.02, 0.04, 0.05, 0.06, 0.08]:
    sc.tl.leiden(
        adata,
        resolution=res,
        key_added=f"leiden_res_{res:.2f}",
        flavor="igraph"
    )

sc.pl.umap(
    adata,
    color=["leiden_res_0.04", "leiden_res_0.06", "leiden_res_0.08"],
    legend_loc="on data",
    show=False
)
plt.savefig(f"{output_dir}/umap_res.pdf", bbox_inches='tight', dpi=300)

# === Plot cell type annotation ===
sc.pl.umap(
    adata,
    color='cell_type',
    legend_loc='right margin',
    title='',
    show=False
)
plt.savefig(f"{output_dir}/umap_celltype.pdf", bbox_inches='tight', dpi=300)

# === Final save ===
adata.write(f"{output_dir}/adata_pmc_umap.h5ad")
print("Done.")
