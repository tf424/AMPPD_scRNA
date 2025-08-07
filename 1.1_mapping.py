#!/usr/bin/env python

import scanpy as sc
import pandas as pd
import anndata
import mygene
import numpy as np

# === Part 1. Mapping ===
# Load the h5ad file
adata = sc.read_h5ad("/rds/general/user/tf424/ephemeral/1_pmc_87_raw_filtered.h5ad")

print(adata.obs.columns)

# Define column names manually
columns = ["original_index", "attribute", "feature_type", "gene_id", "gene_name"]
df = pd.read_csv("/rds/general/user/tf424/home/gtf_annotations.csv", header=0, names=columns)
print(df.head())

#  Strip version from AnnData gene names
adata.var["ensembl_id"] = adata.var_names.str.replace(r"\..*", "", regex=True)

#  Strip version from GTF dataframe gene_id
df["ensembl_id"] = df["gene_id"].str.replace(r"\..*", "", regex=True)

# Create full mapping
ensg_to_symbol = pd.Series(df["gene_name"].values, index=df["ensembl_id"]).to_dict()

# Map gene symbols onto AnnData
adata.var["gene_symbol"] = adata.var["ensembl_id"].map(ensg_to_symbol).fillna("")

# Total genes
total = adata.var.shape[0]

# Mapped gene symbols (non-empty)
mapped = (adata.var["gene_symbol"] != "").sum()

# Unmapped gene symbols
unmapped = total - mapped

print(f"✅ Mapped: {mapped} / {total} genes ({mapped / total:.2%})")
print(f"❌ Unmapped: {unmapped}")

adata.write("/rds/general/user/tf424/ephemeral/97/1_pmc_87_mapped.h5ad")

print("adata.X type:", type(adata.X))
print("Layers available:", adata.layers.keys())


# === Part 2. Further Mapping ===

# Load the GTF-mapped h5ad file
adata = sc.read_h5ad("/rds/general/user/tf424/ephemeral/97/1_pmc_87_mapped.h5ad")

# Identify unmapped genes (gene_symbol == ensembl_id → means no real symbol was found)
unmapped_mask = adata.var["gene_symbol"] == adata.var["ensembl_id"]
unmapped_ids = adata.var.loc[unmapped_mask, "ensembl_id"].unique().tolist()

print(f"🔍 Attempting MyGene mapping for {len(unmapped_ids)} genes...")

# Query MyGene
mg = mygene.MyGeneInfo()
results = mg.querymany(unmapped_ids, scopes="ensembl.gene", fields="symbol", species="human")

# Build mapping dictionary from MyGene results
mg_map = {r["query"]: r.get("symbol", "") for r in results if not r.get("notfound", False)}

# Ensure gene_symbol column is plain string, not Categorical
adata.var["gene_symbol"] = adata.var["gene_symbol"].astype(str)

adata.var.loc[unmapped_mask, "gene_symbol"] = (
    adata.var.loc[unmapped_mask, "ensembl_id"]
    .map(mg_map)
    .fillna(adata.var.loc[unmapped_mask, "ensembl_id"])
)

adata.var["gene_symbol"] = adata.var["gene_symbol"].astype(str)
adata.var_names = adata.var["gene_symbol"]
adata.var_names_make_unique()

total = adata.var.shape[0]
mapped = (adata.var["gene_symbol"] != adata.var["ensembl_id"]).sum()
unmapped = total - mapped

print(f"✅ Total genes: {total}")
print(f"🧬 Now mapped: {mapped} ({mapped / total:.2%})")
print(f"❌ Still unmapped (gene_symbol == ensembl_id): {unmapped}")

unmapped_genes = adata.var.loc[adata.var["gene_symbol"] == adata.var["ensembl_id"], "ensembl_id"]
print(unmapped_genes.tolist()[:10])  # show first 10

unmapped_genes.to_csv("/rds/general/user/tf424/ephemeral/97/unmapped_genes_all.csv", index=False)

print(adata.var_names[:10])       
print(adata.shape)                

# Fix the index name before saving
adata.var.index.name = None
adata.write("/rds/general/user/tf424/ephemeral/97/1.1_pmc_87_remapped.h5ad")
print("✅ Saved successfully.")

# Check how many gene names are duplicated
adata = sc.read_h5ad("/rds/general/user/tf424/ephemeral/97/1.1_pmc_87_remapped.h5ad")
duplicated = adata.var_names.duplicated()
print(f"Number of duplicated gene names: {duplicated.sum()}")

adata = adata[adata.obs['cell_type'] == 'Oligo'].copy()
adata.write("/rds/general/user/tf424/ephemeral/97/1.1_pmc_oligo_87_remapped.h5ad")