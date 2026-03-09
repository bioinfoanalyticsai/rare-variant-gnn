#!/usr/bin/env python3
"""
========================================================================================
    bin/predict_gnn.py
    Run inference with a trained GNN model.
    Outputs per-variant pathogenicity probabilities and ranked scores.
========================================================================================
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gnn_model import build_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='GNN inference for variant pathogenicity')
    p.add_argument('--graph',       required=True)
    p.add_argument('--model',       required=True)
    p.add_argument('--predictions', default='predictions.tsv')
    p.add_argument('--probs',       default='probabilities.tsv')
    p.add_argument('--scores',      default='variant_scores.tsv')
    return p.parse_args()


def load_model(ckpt_path: str, graph_data: Data) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_args = ckpt.get('args', {})
    model = build_model(
        in_channels     = graph_data.num_node_features,
        hidden_channels = model_args.get('hidden', 128),
        num_layers      = model_args.get('layers', 3),
        heads           = model_args.get('heads', 4),
        dropout         = 0.0  # No dropout at inference
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading graph and model...")
    graph_data = torch.load(args.graph, map_location=device)
    model = load_model(args.model, graph_data).to(device)
    graph_data = graph_data.to(device)

    logger.info(f"Running inference on {graph_data.num_nodes} nodes...")

    with torch.no_grad():
        logits, probs = model(graph_data.x, graph_data.edge_index)

    preds = (probs >= 0.5).long().cpu().numpy()
    probs_np = probs.cpu().numpy()

    # Build output DataFrames
    # Node metadata
    node_ids = getattr(graph_data, 'node_ids', np.arange(graph_data.num_nodes))
    variant_ids = getattr(graph_data, 'variant_ids', None)

    pred_df = pd.DataFrame({
        'node_idx':   np.arange(graph_data.num_nodes),
        'node_id':    node_ids if isinstance(node_ids, list) else node_ids,
        'prediction': preds,
        'predicted_class': ['PATHOGENIC' if p == 1 else 'BENIGN' for p in preds],
    })

    prob_df = pd.DataFrame({
        'node_idx':             np.arange(graph_data.num_nodes),
        'pathogenicity_prob':   probs_np,
        'benign_prob':          1 - probs_np
    })

    # Variant-level scores sorted by pathogenicity
    score_df = pred_df.copy()
    score_df['pathogenicity_score'] = probs_np
    score_df = score_df.sort_values('pathogenicity_score', ascending=False)
    score_df['rank'] = range(1, len(score_df) + 1)

    pred_df.to_csv(args.predictions, sep='\t', index=False)
    prob_df.to_csv(args.probs, sep='\t', index=False)
    score_df.to_csv(args.scores, sep='\t', index=False)

    n_path = (preds == 1).sum()
    logger.info(f"Inference complete: {n_path}/{len(preds)} variants predicted pathogenic")
    logger.info(f"Scores saved to {args.scores}")


if __name__ == '__main__':
    main()
