#!/usr/bin/env python3
"""Map variant/gene features onto PPI graph nodes."""
import argparse, logging, pickle
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--graph',    required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--output',   default='featured_graph.pkl')
    return p.parse_args()

def main():
    args = parse_args()

    with open(args.graph, 'rb') as f:
        graph_data = pickle.load(f)
    G = graph_data['graph']

    feat_df = pd.read_csv(args.features, sep='\t')
    logger.info(f"Features: {len(feat_df)} rows × {feat_df.shape[1]} cols")

    # Aggregate to gene level: max value per feature (worst-case variant per gene)
    id_cols = {'variant_id', 'gene'}
    numeric_cols = [c for c in feat_df.columns
                    if c not in id_cols and pd.api.types.is_numeric_dtype(feat_df[c])]

    if 'gene' in feat_df.columns:
        gene_feat = feat_df.groupby('gene')[numeric_cols].max().reset_index()
    else:
        # No gene column — use variant_id as node key
        gene_feat = feat_df.copy()
        gene_feat['gene'] = feat_df.get('variant_id', feat_df.index.astype(str))

    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    n_nodes = len(node_list)
    n_feats = len(numeric_cols)

    X = np.zeros((n_nodes, n_feats), dtype=np.float32)
    matched = 0
    for _, row in gene_feat.iterrows():
        gene = str(row.get('gene', ''))
        if gene in node_to_idx:
            X[node_to_idx[gene]] = row[numeric_cols].values.astype(np.float32)
            matched += 1

    logger.info(f"Matched {matched}/{len(gene_feat)} genes to graph nodes "
                f"({matched/max(len(gene_feat),1)*100:.1f}%)")

    with open(args.output, 'wb') as f:
        pickle.dump({
            'graph':          G,
            'node_list':      node_list,
            'node_features':  X,
            'feature_names':  numeric_cols,
            'k_hops':         graph_data.get('k_hops', 2),
        }, f)
    logger.info(f"Featured graph saved → {args.output}")

if __name__ == '__main__':
    main()
