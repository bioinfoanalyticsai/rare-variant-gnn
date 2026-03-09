#!/usr/bin/env python3
"""Build a protein-protein interaction graph from STRING network data."""
import argparse, logging, pickle
import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ppi',       required=True)
    p.add_argument('--k-hops',    type=int,   default=2)
    p.add_argument('--min-score', type=float, default=400)
    p.add_argument('--output',    default='graph_topology.pkl')
    p.add_argument('--stats',     default='graph_stats.txt')
    return p.parse_args()

def main():
    args = parse_args()
    logger.info(f"Loading PPI from {args.ppi}")
    ppi_df = pd.read_csv(args.ppi, sep='\t')

    if 'protein1' not in ppi_df.columns:
        ppi_df.columns = ['protein1', 'protein2', 'combined_score']

    logger.info(f"Raw edges: {len(ppi_df)}")
    filtered = ppi_df[ppi_df['combined_score'] >= args.min_score]
    logger.info(f"After min_score={args.min_score}: {len(filtered)} edges")

    G = nx.Graph()
    for _, row in filtered.iterrows():
        G.add_edge(str(row['protein1']), str(row['protein2']),
                   weight=float(row['combined_score']) / 1000.0)

    # Add all proteins from the full list as isolated nodes too
    for _, row in ppi_df.iterrows():
        G.add_node(str(row['protein1']))
        G.add_node(str(row['protein2']))

    for node in G.nodes():
        G.nodes[node]['degree']  = G.degree(node)
        G.nodes[node]['node_id'] = node

    logger.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    with open(args.output, 'wb') as f:
        pickle.dump({'graph': G, 'k_hops': args.k_hops}, f)

    degrees = [d for _, d in G.degree()]
    with open(args.stats, 'w') as f:
        f.write(f"nodes\t{G.number_of_nodes()}\n")
        f.write(f"edges\t{G.number_of_edges()}\n")
        f.write(f"avg_degree\t{np.mean(degrees):.2f}\n")
        f.write(f"max_degree\t{max(degrees)}\n")
        f.write(f"connected_components\t{nx.number_connected_components(G)}\n")

    logger.info(f"Graph saved → {args.output}")

if __name__ == '__main__':
    main()
