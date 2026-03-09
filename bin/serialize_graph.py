#!/usr/bin/env python3
"""
========================================================================================
    bin/serialize_graph.py
    Convert a featured NetworkX graph into a PyTorch Geometric Data object.

    Handles:
        - Edge index construction (bidirectional)
        - Edge weight normalization
        - Feature normalization (StandardScaler per feature)
        - Node ID metadata embedding
        - Train/val/test mask placeholder
========================================================================================
"""
import argparse, json, logging, pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, remove_self_loops
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True, help='featured_graph.pkl')
    p.add_argument('--output', default='pyg_graph.pt')
    p.add_argument('--stats',  default='graph_stats.json')
    p.add_argument('--normalize', action='store_true', default=True)
    return p.parse_args()


def networkx_to_pyg(G, node_list, X, feature_names) -> Data:
    """Convert NetworkX graph + feature matrix to PyG Data."""
    import networkx as nx

    n = len(node_list)
    node_to_idx = {n_: i for i, n_ in enumerate(node_list)}

    # Build edge index
    edges = [(node_to_idx[u], node_to_idx[v])
             for u, v in G.edges()
             if u in node_to_idx and v in node_to_idx]

    if not edges:
        logger.warning("No edges found! Creating a fully disconnected graph.")
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_weight = torch.zeros(0, dtype=torch.float)
    else:
        src, dst = zip(*edges)
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)  # undirected

        # Extract edge weights
        weights = []
        for u, v in edges:
            n_u = node_list[u]
            n_v = node_list[v]
            w = G[n_u][n_v].get('weight', 1.0)
            weights.append(w)
        edge_weight = torch.tensor(weights + weights, dtype=torch.float)

    # Node features
    x_tensor = torch.tensor(X, dtype=torch.float32)

    # Build PyG Data object
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=n
    )

    # Attach metadata
    data.node_ids     = node_list
    data.feature_names = feature_names

    return data


def normalize_features(data: Data) -> Data:
    """Standardize node features (zero-mean, unit-variance)."""
    scaler = StandardScaler()
    x_np   = data.x.numpy()
    # Only normalize non-zero rows
    nonzero_mask = (x_np != 0).any(axis=1)
    if nonzero_mask.sum() > 0:
        x_np[nonzero_mask] = scaler.fit_transform(x_np[nonzero_mask])
    data.x = torch.tensor(x_np, dtype=torch.float32)
    return data


def main():
    args = parse_args()

    logger.info(f"Loading featured graph from {args.input}")
    with open(args.input, 'rb') as f:
        graph_data = pickle.load(f)

    G            = graph_data['graph']
    node_list    = graph_data['node_list']
    X            = graph_data['node_features']
    feature_names = graph_data['feature_names']

    logger.info(f"Nodes: {len(node_list)} | Features: {X.shape[1]}")

    # Convert to PyG
    data = networkx_to_pyg(G, node_list, X, feature_names)

    # Normalize features
    if args.normalize:
        data = normalize_features(data)
        logger.info("Node features normalized (StandardScaler)")

    # Remove self-loops and re-add (clean)
    data.edge_index, data.edge_weight = remove_self_loops(data.edge_index, data.edge_weight)
    data.edge_index, data.edge_weight = add_self_loops(
        data.edge_index, data.edge_weight, num_nodes=data.num_nodes
    )

    logger.info(f"PyG Data: {data}")

    # Save
    torch.save(data, args.output)
    logger.info(f"PyG graph saved to {args.output}")

    # Stats
    stats = {
        'num_nodes':     int(data.num_nodes),
        'num_edges':     int(data.num_edges),
        'num_features':  int(data.num_node_features),
        'feature_names': feature_names,
        'has_self_loops': True,
        'is_undirected':  True
    }
    with open(args.stats, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Graph stats: {stats}")


if __name__ == '__main__':
    main()
