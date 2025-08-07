#!/usr/bin/env python

import os
import pandas as pd
import scanpy as sc
import anndata
import numpy as np
from scipy.sparse import issparse

# Load the h5ad file
data_path = "/rds/general/user/tf424/ephemeral/97/amppd_merged_snRNAseq_annotated_original.h5ad"
adata = anndata.read(data_path)
adata.obs["barcode"] = adata.obs_names

# Load demographics CSV
file_1 = "/rds/general/user/tf424/projects/ppmi_verily/live/AMP-PD/participants/amp_pd_case_control.csv"
file_2 = "/rds/general/user/tf424/projects/ppmi_verily/live/AMP-PD/participants/amp_pd_participants.csv"
lbd = pd.read_csv("/rds/general/user/tf424/projects/ppmi_verily/live/AMP-PD/clinical/LBD_Cohort_Path_Data.csv")
demographics = pd.read_csv("/rds/general/user/tf424/home/clinical/Demographics.csv")

# Extract brain region
adata.obs['brain_region'] = adata.obs['sample_id'].str.extract(r'-(GPI|PMC|PVC|DMNX|PFC)-')

# Add gender
demographics = demographics.set_index("participant_id")
adata.obs["sex"] = adata.obs["participant_id"].map(demographics["sex"])
adata.obs['sex'] = adata.obs['sex'].astype('category')

# Add Braak stage score
filtered_lbd = lbd[lbd["participant_id"].isin(adata.obs["participant_id"].unique())]
filtered_lbd = filtered_lbd.set_index("participant_id")
adata.obs["path_braak_lb"] = adata.obs["participant_id"].map(filtered_lbd["path_braak_lb"]) 

# Add diagnosis
df_1 = pd.read_csv(file_1)
df_2 = pd.read_csv(file_2)
merged_data = pd.merge(df_1, df_2, on='participant_id', how='inner')
participants_data = merged_data
merged_obs = pd.merge(adata.obs, participants_data, on='participant_id', how='inner')
merged_obs = merged_obs.reset_index(drop=True)
adata.obs = merged_obs

# Filter to just the two diagnosis groups
keep = adata.obs['diagnosis_latest'].isin([
    "Parkinson's Disease",
    "No PD Nor Other Neurological Disorder"
])
adata = adata[keep].copy()

# Map to a “diagnosis” column
mapping = {
    "Parkinson's Disease": "PD",
    "No PD Nor Other Neurological Disorder": "HC"
}
adata.obs['diagnosis'] = adata.obs['diagnosis_latest'].map(mapping)

print(adata.obs['diagnosis'].value_counts())
print("Unique participants per diagnosis:")
print(adata.obs.groupby('diagnosis')['participant_id'].nunique())

# Drop guid column if present
if "guid" in adata.obs.columns:
    adata.obs = adata.obs.drop(columns=["guid"])

# Convert specified columns to categorical
cols_to_convert = [
    "participant_id",
    "diagnosis_at_baseline",
    "diagnosis_latest",
    "case_control_other_at_baseline",
    "case_control_other_latest",
    "study",
    "study_participant_id"
]

for col in cols_to_convert:
    if col in adata.obs.columns:
        if not pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].astype("category")
            print(f"Converted: {col}")
        else:
            print(f"Already categorical: {col}")
    else:
        print(f"Column not found: {col}")

# Subset to PMC
pmc_adata = adata[adata.obs['brain_region'] == 'PMC'].copy()
pmc_adata = pmc_adata.to_memory()

# Keep only participants with enough cells
min_cells_required = 500
participant_counts = pmc_adata.obs["participant_id"].value_counts()
valid_participants = participant_counts[participant_counts >= min_cells_required].index.tolist()
pmc_adata = pmc_adata[pmc_adata.obs["participant_id"].isin(valid_participants)].copy()

# Save
pmc_adata.write("/rds/general/user/tf424/ephemeral/1_pmc_87_raw_filtered.h5ad")
print("✅ Done. PMC filtered and saved.")
