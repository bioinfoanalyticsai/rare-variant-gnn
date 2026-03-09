#!/usr/bin/env python3
"""
========================================================================================
    bin/build_protein_graph.py
    Build a protein-protein interaction graph from STRING network data.

    Input:  STRING PPI TSV (protein1, protein2, combined_score)
    Output: NetworkX graph → pickled with node metadata
========================================================================================
"""
import argparse, logging, pickle
import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ppi',      required=True,  help='STRING PPI TSV')
    p.add_argument('--k-hops',   type=int, default=2)
    p.add_argument('--min-score', type=float, default=400, help='Min STRING combined score (0-1000)')
    p.add_argument('--output',   default='graph_topology.pkl')
    p.add_argument('--stats',    default='graph_stats.txt')
    return p.parse_args()


def build_graph(ppi_df: pd.DataFrame, min_score: float) -> nx.Graph:
    G = nx.Graph()
    filtered = ppi_df[ppi_df['combined_score'] >= min_score]
    for _, row in filtered.iterrows():
        G.add_edge(row['protein1'], row['protein2'],
                   weight=float(row['combined_score']) / 1000.0)
    return G


def add_node_attributes(G: nx.Graph) -> nx.Graph:
    """Add basic node properties as attributes."""
    for node in G.nodes():
        G.nodes[node]['degree']       = G.degree(node)
        G.nodes[node]['betweenness']  = 0.0  # Computed lazily
        G.nodes[node]['node_id']      = node
    return G


def main():
    args = parse_args()

    logger.info(f"Loading PPI network from {args.ppi}")
    ppi_df = pd.read_csv(args.ppi, sep='\t')

    # Handle various STRING formats
    if 'protein1' not in ppi_df.columns:
        ppi_df.columns = ['protein1', 'protein2', 'combined_score']

    logger.info(f"Raw PPI edges: {len(ppi_df)}")
    G = build_graph(ppi_df, min_score=args.min_score)
    G = add_node_attributes(G)

    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Save
    with open(args.output, 'wb') as f:
        pickle.dump({'graph': G, 'k_hops': args.k_hops}, f)

    # Stats
    degrees = [d for _, d in G.degree()]
    with open(args.stats, 'w') as f:
        f.write(f"nodes\t{G.number_of_nodes()}\n")
        f.write(f"edges\t{G.number_of_edges()}\n")
        f.write(f"avg_degree\t{np.mean(degrees):.2f}\n")
        f.write(f"max_degree\t{max(degrees)}\n")
        f.write(f"connected_components\t{nx.number_connected_components(G)}\n")

    logger.info(f"Graph saved to {args.output}")


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────────────────────────


#!/usr/bin/env python3
"""
========================================================================================
    bin/assign_node_features.py
    Map variant/gene features onto graph nodes.
    Handles:
        - Gene-to-node mapping
        - Multiple variants per gene (aggregation: max pathogenicity-associated)
        - Missing node features (zero-padding)
========================================================================================
"""
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


def aggregate_variant_features(features_df: pd.DataFrame, gene_col: str = 'gene') -> pd.DataFrame:
    """For genes with multiple variants, aggregate by taking the max of each feature."""
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    if gene_col in numeric_cols:
        numeric_cols.remove(gene_col)
    agg = features_df.groupby(gene_col)[numeric_cols].max().reset_index()
    return agg


def main():
    args = parse_args()

    with open(args.graph, 'rb') as f:
        graph_data = pickle.load(f)
    G = graph_data['graph']

    features_df = pd.read_csv(args.features, sep='\t')
    logger.info(f"Features: {len(features_df)} variants × {features_df.shape[1]} columns")

    # Aggregate to gene level
    gene_features = aggregate_variant_features(features_df)

    # Map gene → node index
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    feature_cols = [c for c in gene_features.columns if c != 'gene']
    n_nodes = len(node_list)
    n_feats = len(feature_cols)

    # Initialize feature matrix with zeros (missing genes get zero features)
    X = np.zeros((n_nodes, n_feats), dtype=np.float32)

    matched = 0
    for _, row in gene_features.iterrows():
        gene = str(row['gene'])
        if gene in node_to_idx:
            X[node_to_idx[gene]] = row[feature_cols].values.astype(np.float32)
            matched += 1

    logger.info(f"Matched {matched}/{len(gene_features)} genes to graph nodes")

    # Save featured graph
    with open(args.output, 'wb') as f:
        pickle.dump({
            'graph':         G,
            'node_list':     node_list,
            'node_features': X,
            'feature_names': feature_cols,
            'k_hops':        graph_data.get('k_hops', 2)
        }, f)
    logger.info(f"Featured graph saved to {args.output}")


if __name__ == '__main__':
    main()
