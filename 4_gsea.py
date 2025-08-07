#!/usr/bin/env python

import os
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt

# === Config ===
input_dir = "/rds/general/user/tf424/ephemeral/97/3_DEG_oligo_0.14"
output_dir = os.path.join(input_dir, "gsea_results")
os.makedirs(output_dir, exist_ok=True)

# === Loop through cluster DEG files ===
for file in os.listdir(input_dir):
    if not file.endswith("_deseq2_results.csv"):
        continue

    cluster = file.replace("_deseq2_results.csv", "")
    in_path = os.path.join(input_dir, file)
    print(f"🔬 Processing: {cluster}")

    try:
        # Load and rank genes
        df = pd.read_csv(in_path)
        ranked = df[["gene_symbol", "log2FoldChange"]].dropna().drop_duplicates("gene_symbol")
        ranked = ranked.set_index("gene_symbol").sort_values("log2FoldChange", ascending=False)
        ranked = ranked.loc[~ranked.index.duplicated(keep="first")]

        ranked_df = ranked.reset_index()
        ranked_df.columns = ["Gene", "Rank"]

        # Run GSEA
        gsea = gp.prerank(
            rnk=ranked_df,
            gene_sets="GO_Biological_Process_2025",
            organism="Human",
            permutation_num=500,
            permutation_type="phenotype",
            method="s2n",
            threads=16,
            outdir=None,
            seed=123
        )

        # Save results
        result_csv = os.path.join(output_dir, f"{cluster}_gsea.csv")
        gsea.res2d.to_csv(result_csv, index=False)
        print(f"✅ GSEA result saved: {result_csv}")

        # Plot top 5 UP terms
        top_up = gsea.res2d[gsea.res2d["NES"] > 0].sort_values("NES", ascending=False)
        if not top_up.empty:
            terms = top_up["Term"].head(5)
            axes = gsea.plot(terms=terms)
            fig = axes[0].figure if isinstance(axes, list) else axes.figure
            fig.savefig(os.path.join(output_dir, f"{cluster}_up_top5_terms.pdf"),
                        bbox_inches="tight", dpi=300)
            print(f"📈 Saved up-regulated plot: {cluster}_up_top5_terms.pdf")

        # Plot top 5 DOWN terms
        top_down = gsea.res2d[gsea.res2d["NES"] < 0].sort_values("NES", ascending=True)
        if not top_down.empty:
            terms = top_down["Term"].head(5)
            axes = gsea.plot(terms=terms)
            fig = axes[0].figure if isinstance(axes, list) else axes.figure
            fig.savefig(os.path.join(output_dir, f"{cluster}_down_top5_terms.pdf"),
                        bbox_inches="tight", dpi=300)
            print(f"📉 Saved down-regulated plot: {cluster}_down_top5_terms.pdf")

    except Exception as e:
        print(f"❌ Error processing {cluster}: {e}")




