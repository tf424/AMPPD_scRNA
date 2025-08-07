#!/usr/bin/env python

import os
import pandas as pd
import gseapy as gp

# === Config ===
input_dir = "/rds/general/user/tf424/ephemeral/97/3_DEG_oligo_0.14"
out_dir = os.path.join(input_dir, "gseapy_enrichment_split")
os.makedirs(out_dir, exist_ok=True)

# === Helper to check valid terms ===
def has_valid_terms(df, cutoff=0.05):
    return df is not None and not df.empty and (df["Adjusted P-value"] < cutoff).any()

# === List DE result files ===
csv_files = [f for f in os.listdir(input_dir) if f.endswith("_deseq2_results.csv")]

for file in csv_files:
    celltype = file.replace("_deseq2_results.csv", "")
    print(f"\n🔎 Processing: {celltype}")

    # Load DE results
    df = pd.read_csv(os.path.join(input_dir, file))

    # Upregulated
    up = df[(df["padj"] < 0.05) & (df["log2FoldChange"] > 0.5)]
    up_genes = up["gene_symbol"].dropna().tolist()

    # Downregulated
    down = df[(df["padj"] < 0.05) & (df["log2FoldChange"] < -0.5)]
    down_genes = down["gene_symbol"].dropna().tolist()

    for direction, gene_list in [("up", up_genes), ("down", down_genes)]:
        if len(gene_list) < 5:
            print(f"⚠️ Skipping {celltype} {direction} — too few significant genes.")
            continue

        # Output folders
        go_out = os.path.join(out_dir, f"{celltype}_GO_BP_{direction}")
        kegg_out = os.path.join(out_dir, f"{celltype}_KEGG_{direction}")
        os.makedirs(go_out, exist_ok=True)
        os.makedirs(kegg_out, exist_ok=True)

        # === GO: Biological Process ===
        try:
            enr_go = gp.enrichr(gene_list=gene_list,
                                gene_sets='GO_Biological_Process_2025',
                                organism='Human',
                                outdir=go_out,
                                cutoff=0.05,
                                no_plot=False)

            if has_valid_terms(enr_go.res2d):
                gp.barplot(enr_go.res2d,
                           title=f"{celltype} - GO BP ({direction})",
                           ofname=os.path.join(go_out, f"{celltype}_GO_BP_{direction}_barplot.png"))
                gp.dotplot(enr_go.res2d,
                           title=f"{celltype} - GO BP ({direction})",
                           ofname=os.path.join(go_out, f"{celltype}_GO_BP_{direction}_dotplot.png"))
            else:
                print(f"❌ No enriched GO terms for {celltype} ({direction}).")

        except Exception as e:
            print(f"❌ GO enrichment failed for {celltype} ({direction}): {e}")
