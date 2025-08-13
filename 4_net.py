#!/usr/bin/env python

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import textwrap
import gseapy as gp

# =========================
# Config
# =========================
base_dir   = "/rds/general/user/tf424/ephemeral/97/3_DEG_oligo_0.14"
gsea_dir   = os.path.join(base_dir, "gsea_results")
output_dir = os.path.join(base_dir, "network_results_refined_sig")
os.makedirs(output_dir, exist_ok=True)

alpha_main   = 0.05   # FDR threshold for publication
alpha_relax  = 0.25   # fallback if nothing survives
top_n_each   = 5      # top up/down terms to plot
emap_cutoff  = 0.15   # Jaccard cutoff for edges (try 0.10–0.20)
vmin, vmax   = -2, 3  # NES color scale
cmap         = plt.cm.RdYlBu_r

# =========================
# Utilities
# =========================
def wrap_label(term: str, width: int = 25) -> str:
    return '\n'.join(textwrap.wrap(str(term), width=width))

def save_components_and_edges(cluster_name, nodes, edges, outdir):
    # Edges table with overlap sizes
    edges_out = edges.copy()
    edges_out['n_overlap_genes'] = edges_out['overlap_genes'].str.split(';').apply(
        lambda x: 0 if (x is np.nan or x is None) else len(x)
    )
    edges_out.to_csv(os.path.join(outdir, f"{cluster_name}_edges.csv"), index=False)

    # Graph for component stats
    G = nx.from_pandas_edgelist(
        edges, 'src_idx', 'targ_idx',
        edge_attr=['jaccard_coef', 'overlap_coef', 'overlap_genes']
    )
    for idx in nodes.index:
        if idx not in G:
            G.add_node(idx)

    comp_rows = []
    for comp_id, comp in enumerate(nx.connected_components(G), start=1):
        subG = G.subgraph(comp).copy()
        n = subG.number_of_nodes()
        m = subG.number_of_edges()
        possible = n * (n - 1) / 2
        density = (m / possible) if possible else 0.0
        mean_jaccard = np.mean([d['jaccard_coef'] for *_, d in subG.edges(data=True)]) if m else 0.0
        comp_rows.append({
            'cluster': cluster_name,
            'component_id': comp_id,
            'size': n,
            'edges': m,
            'edge_density': round(density, 3),
            'mean_jaccard': round(mean_jaccard, 3),
            'terms': "; ".join(nodes.loc[list(comp), 'Term'])
        })
    pd.DataFrame(comp_rows).to_csv(os.path.join(outdir, f"{cluster_name}_components.csv"), index=False)
    return G  # return graph for plotting

# =========================
# Main loop
# =========================
for file in os.listdir(gsea_dir):
    if not file.endswith("_gsea.csv"):
        continue

    cluster_name = file.replace("_gsea.csv", "")
    gsea_path = os.path.join(gsea_dir, file)

    # ---- Load results
    gsea = pd.read_csv(gsea_path, index_col=0)
    if 'NES' not in gsea.columns:
        print(f"{cluster_name}: Missing NES column, skipping.")
        continue

    # Keep only rows with valid NES
    gsea['NES'] = pd.to_numeric(gsea['NES'], errors='coerce')
    gsea = gsea.dropna(subset=['NES'])
    if gsea.empty:
        print(f"{cluster_name}: No valid NES, skipping.")
        continue

    # ---- Clean term names (drop "(GO:XXXXXXX)")
    if 'Term' not in gsea.columns:
        print(f"{cluster_name}: Missing Term column, skipping.")
        continue

    gsea['Term'] = (
        gsea['Term']
        .astype(str)
        .str.replace(r"\s*\(GO:\d+\)", "", regex=True)
    )
    # ensure term index unique
    gsea['Term'] = gsea['Term'].where(~gsea['Term'].duplicated(),
                                      gsea['Term'] + '_' + gsea.groupby('Term').cumcount().astype(str))
    gsea = gsea.set_index('Term')

    # ---- Apply significance filter (FDR q < 0.05), relax to 0.25 if needed
    fdr_col = None
    for cand in ['FDR q-val', 'FDR', 'FDR_q', 'padj', 'qvalue']:
        if cand in gsea.columns:
            fdr_col = cand
            break
    if fdr_col is None and 'NOM p-val' not in gsea.columns:
        print(f"{cluster_name}: No FDR/NOM p available; plotting NES-only (exploratory).")

    if fdr_col is not None:
        sig = gsea[gsea[fdr_col] < alpha_main].copy()
        if sig.empty:
            sig = gsea[gsea[fdr_col] < alpha_relax].copy()
    else:
        # fallback: use nominal p if present, otherwise NES-only (not ideal for publication)
        if 'NOM p-val' in gsea.columns:
            sig = gsea[gsea['NOM p-val'] < 0.05].copy()
        else:
            sig = gsea.copy()

    if sig.empty:
        print(f"{cluster_name}: No significant terms even at relaxed threshold; skipping.")
        continue

    # ---- Choose top N per direction
    pos_df = sig[sig['NES'] > 0].sort_values('NES', ascending=False)
    neg_df = sig[sig['NES'] < 0].sort_values('NES', ascending=True)

    up_terms   = pos_df.index[:top_n_each].tolist()
    down_terms = neg_df.index[:top_n_each].tolist()
    all_terms  = up_terms + down_terms

    if len(all_terms) == 0:
        print(f"{cluster_name}: No terms to plot after filtering, skipping.")
        continue

    print(f"{cluster_name} — plotting {len(up_terms)} up and {len(down_terms)} down terms "
          f"(FDR<={alpha_main} or relaxed {alpha_relax} if needed).")

    # ---- Enrichment map (edges based on leading-edge overlaps)
    try:
        subset = gsea.loc[all_terms].copy()
        subset['Term'] = subset.index
        nodes, edges = gp.enrichment_map(subset, column='NES', cutoff=emap_cutoff)
    except Exception as e:
        print(f"{cluster_name}: enrichment_map failed: {e}")
        continue

    # ---- Quantify connectivity and build graph
    G = save_components_and_edges(cluster_name, nodes, edges, output_dir)

    # ---- Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    pos_layout = nx.kamada_kawai_layout(G)

    # Nodes
    nx.draw_networkx_nodes(
        G, pos=pos_layout,
        node_color=list(nodes['NES']),
        node_size=list(nodes['Hits_ratio'] * 1000),
        cmap=cmap, vmin=vmin, vmax=vmax
    )

    # Labels (wrapped)
    nodes_plot = nodes.copy()
    nodes_plot['Term'] = nodes_plot['Term'].apply(wrap_label)
    nx.draw_networkx_labels(
        G, pos=pos_layout,
        labels=nodes_plot['Term'].to_dict(),
        font_size=9, font_weight='bold'
    )

    # Edges (width ∝ Jaccard overlap)
    edge_widths = [d['jaccard_coef'] * 6 for *_, d in G.edges(data=True)]
    nx.draw_networkx_edges(
        G, pos=pos_layout,
        width=edge_widths,
        edge_color='#CDDBD4', alpha=0.9
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.1)
    cbar.set_label('NES')

    # Size legend
    size_handles = [
        Line2D([0], [0], marker='o', color='gray', markeredgecolor='k',
               markersize=np.sqrt(r * 2000), linestyle='None', label=f'{int(r * 100)}% hits')
        for r in [0.1, 0.3, 0.5]
    ]
    leg = fig.legend(
        handles=size_handles,
        title='Node size ∝ Hits_ratio',
        loc='upper right', bbox_to_anchor=(0.98, 0.98),
        frameon=True, markerscale=0.5, labelspacing=1.2, borderpad=1.0, handletextpad=0.5
    )
    leg.get_title().set_fontsize(10)

    plt.title("")  # clean title area
    plt.axis('off')
    plt.tight_layout()

    out_pdf = os.path.join(output_dir, f"{cluster_name}_network.pdf")
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved network: {out_pdf}")








