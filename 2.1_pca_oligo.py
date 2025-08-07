#!/usr/bin/env python

import scanpy as sc
import numpy as np
import pandas as pd
import sys
import os

# === Paths ===
input_path = "/rds/general/user/tf424/ephemeral/97/1.1_pmc_oligo_87_remapped.h5ad"
output_path = "/rds/general/user/tf424/ephemeral/1_pmc_oligo_87_pca.h5ad"

# === Helper ===
def check_participant_counts(adata, label):
    print(f"\n[{label}] Unique participants: {adata.obs['participant_id'].nunique()}")
    print(adata.obs['participant_id'].value_counts())

# === Load AnnData ===
print("🔄 Loading AnnData...")
adata = sc.read_h5ad(input_path)
check_participant_counts(adata, "Loaded")

# === Store barcodes safely ===
if 'barcode' not in adata.obs.columns:
    print("🧬 Adding barcode column from obs_names")
    adata.obs['barcode'] = adata.obs_names.copy()

# === Exit if barcode is still missing
if 'barcode' not in adata.obs.columns:
    print("❌ ERROR: 'barcode' column not found after parsing. Exiting.")
    sys.exit(1)

# === Backup raw counts ===
adata.raw = adata.copy()

# === Inspect matrix
print("adata.X type:", type(adata.X))
print("Layers available:", adata.layers.keys())

# === Normalize and log-transform
print("🧪 Normalizing...")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

# === HVG selection
print("🎯 Selecting HVGs...")
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# === Subset to HVGs and retain barcode
adata = adata[:, adata.var.highly_variable].copy()
adata.obs['barcode'] = adata.obs['barcode'].copy()  # ✅ restore after subset

# === Scale and PCA
sc.pp.scale(adata, max_value=10)
print("🧬 Running PCA...")
sc.tl.pca(adata, svd_solver='arpack')

# === Save output
print(f"💾 Saving to {output_path}")
adata.write(output_path)

print("✅ Done.")