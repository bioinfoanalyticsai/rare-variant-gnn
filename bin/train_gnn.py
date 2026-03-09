#!/usr/bin/env python3
"""
========================================================================================
    bin/train_gnn.py
    Train the Graph Attention Network on variant pathogenicity data.

    Features:
        - Full training loop with train/val/test evaluation each epoch
        - Early stopping on validation AUC
        - Learning rate scheduling (cosine annealing with warm restarts)
        - Gradient clipping
        - Mixed precision training (AMP)
        - Checkpoint saving (best val AUC + every N epochs)
        - Comprehensive logging (JSON + console)
========================================================================================
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gnn_model import build_model, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """Compute classification metrics given true labels and predicted probabilities."""
    preds = (probs >= 0.5).astype(int)
    metrics = {}
    try:
        metrics['auroc'] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics['auroc'] = 0.0
    try:
        metrics['auprc'] = float(average_precision_score(labels, probs))
    except ValueError:
        metrics['auprc'] = 0.0

    metrics['f1']        = float(f1_score(labels, preds, zero_division=0))
    metrics['precision'] = float(precision_score(labels, preds, zero_division=0))
    metrics['recall']    = float(recall_score(labels, preds, zero_division=0))
    metrics['accuracy']  = float((labels == preds).mean())
    return metrics


# ─── Training Loop ────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, graph_data, labels_df, args):
        self.model      = model
        self.graph_data = graph_data
        self.args       = args
        self.device     = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.model = self.model.to(self.device)
        self.graph_data = graph_data.to(self.device)

        # Build masks from split labels
        self.train_mask, self.val_mask, self.test_mask = self._build_masks(labels_df)
        self.labels = self._build_label_tensor(labels_df)

        # Optimizer & scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=args.epochs // 4, T_mult=2
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=torch.cuda.is_available())

        # Tracking
        self.best_val_auroc = 0.0
        self.best_epoch     = 0
        self.patience_cnt   = 0
        self.history        = {'train': [], 'val': [], 'test': []}

    def _build_masks(self, labels_df):
        import pandas as pd
        n_nodes = self.graph_data.num_nodes
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask   = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask  = torch.zeros(n_nodes, dtype=torch.bool)

        for _, row in labels_df.iterrows():
            idx = int(row['node_idx'])
            if idx >= n_nodes:
                continue
            if row['split'] == 'train':
                train_mask[idx] = True
            elif row['split'] == 'val':
                val_mask[idx] = True
            elif row['split'] == 'test':
                test_mask[idx] = True

        return (
            train_mask.to(self.device),
            val_mask.to(self.device),
            test_mask.to(self.device)
        )

    def _build_label_tensor(self, labels_df):
        import pandas as pd
        n_nodes = self.graph_data.num_nodes
        labels = torch.zeros(n_nodes, dtype=torch.float32)
        for _, row in labels_df.iterrows():
            idx = int(row['node_idx'])
            if idx < n_nodes:
                labels[idx] = float(row['label'])
        return labels.to(self.device)

    def _train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            logits, probs = self.model(
                self.graph_data.x,
                self.graph_data.edge_index
            )
            loss = self.model.loss(logits, self.labels, self.train_mask)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        train_labels = self.labels[self.train_mask].cpu().numpy()
        train_probs  = probs[self.train_mask].detach().cpu().numpy()
        metrics = compute_metrics(train_labels, train_probs)
        metrics['loss'] = float(loss.item())
        return metrics

    @torch.no_grad()
    def _eval_epoch(self, mask, split_name: str):
        self.model.eval()
        logits, probs = self.model(
            self.graph_data.x,
            self.graph_data.edge_index
        )
        loss = self.model.loss(logits, self.labels, mask)

        split_labels = self.labels[mask].cpu().numpy()
        split_probs  = probs[mask].detach().cpu().numpy()
        metrics = compute_metrics(split_labels, split_probs)
        metrics['loss'] = float(loss.item())
        return metrics

    def train(self):
        logger.info(f"Training on {self.device} | "
                    f"Train: {self.train_mask.sum()} | "
                    f"Val: {self.val_mask.sum()} | "
                    f"Test: {self.test_mask.sum()}")

        for epoch in range(1, self.args.epochs + 1):
            t0 = time.time()

            train_m = self._train_epoch()
            val_m   = self._eval_epoch(self.val_mask, 'val')
            test_m  = self._eval_epoch(self.test_mask, 'test')

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train'].append(train_m)
            self.history['val'].append(val_m)
            self.history['test'].append(test_m)

            logger.info(
                f"Epoch {epoch:04d}/{self.args.epochs} | "
                f"LR: {lr:.2e} | "
                f"Train AUROC: {train_m['auroc']:.4f} | "
                f"Val AUROC: {val_m['auroc']:.4f} | "
                f"Test AUROC: {test_m['auroc']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpoint saving
            os.makedirs(self.args.model_dir, exist_ok=True)
            if epoch % self.args.save_every == 0:
                ckpt_path = os.path.join(self.args.model_dir, f'checkpoint_epoch_{epoch:04d}.pt')
                self._save_checkpoint(ckpt_path, epoch, val_m)

            # Best model saving
            if val_m['auroc'] > self.best_val_auroc:
                self.best_val_auroc = val_m['auroc']
                self.best_epoch = epoch
                self.patience_cnt = 0
                self._save_checkpoint(self.args.best_model, epoch, val_m)
                logger.info(f"  ✓ New best model saved (val AUROC = {val_m['auroc']:.4f})")
            else:
                self.patience_cnt += 1

            # Early stopping
            if self.args.patience > 0 and self.patience_cnt >= self.args.patience:
                logger.info(f"Early stopping at epoch {epoch} "
                            f"(best epoch: {self.best_epoch})")
                break

        # Final test evaluation with best model
        logger.info("Loading best model for final test evaluation...")
        ckpt = torch.load(self.args.best_model, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        final_test = self._eval_epoch(self.test_mask, 'test')
        logger.info(f"\n{'='*60}")
        logger.info(f"FINAL TEST RESULTS (best epoch {self.best_epoch}):")
        logger.info(f"  AUROC:     {final_test['auroc']:.4f}")
        logger.info(f"  AUPRC:     {final_test['auprc']:.4f}")
        logger.info(f"  F1:        {final_test['f1']:.4f}")
        logger.info(f"  Precision: {final_test['precision']:.4f}")
        logger.info(f"  Recall:    {final_test['recall']:.4f}")
        logger.info(f"{'='*60}\n")

        return self.history, final_test

    def _save_checkpoint(self, path: str, epoch: int, val_metrics: dict):
        torch.save({
            'epoch':            epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics':      val_metrics,
            'args':             vars(self.args)
        }, path)


# ─── Data Loading ─────────────────────────────────────────────────────────────
def load_graph(graph_path: str) -> Data:
    """Load serialized PyG Data object."""
    graph = torch.load(graph_path, map_location='cpu')
    logger.info(f"Graph loaded: {graph.num_nodes} nodes, "
                f"{graph.num_edges} edges, "
                f"{graph.num_node_features} node features")
    return graph


def load_labels(labels_path: str):
    """Load split labels TSV."""
    import pandas as pd
    df = pd.read_csv(labels_path, sep='\t')
    logger.info(f"Labels loaded: {len(df)} entries | "
                f"Pathogenic: {(df['label']==1).sum()} | "
                f"Benign: {(df['label']==0).sum()}")
    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train GNN for rare variant pathogenicity prediction'
    )
    parser.add_argument('--graph',        required=True, help='PyG graph .pt file')
    parser.add_argument('--labels',       required=True, help='Split labels TSV')
    parser.add_argument('--epochs',       type=int,   default=100)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--hidden',       type=int,   default=128)
    parser.add_argument('--layers',       type=int,   default=3)
    parser.add_argument('--heads',        type=int,   default=4)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--batch-size',   type=int,   default=64)
    parser.add_argument('--patience',     type=int,   default=20, help='Early stopping patience (0=disabled)')
    parser.add_argument('--save-every',   type=int,   default=10)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--model-dir',    default='checkpoints/')
    parser.add_argument('--best-model',   default='best_model.pt')
    parser.add_argument('--logs',         default='training_logs.json')
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("="*60)
    logger.info("  Rare Variant GNN Training")
    logger.info("="*60)
    logger.info(f"  Epochs:  {args.epochs}")
    logger.info(f"  LR:      {args.lr}")
    logger.info(f"  Hidden:  {args.hidden}")
    logger.info(f"  Layers:  {args.layers}")
    logger.info(f"  Heads:   {args.heads}")
    logger.info(f"  Device:  {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info("="*60)

    # Load data
    graph_data = load_graph(args.graph)
    labels_df  = load_labels(args.labels)

    # Build model
    in_channels = graph_data.num_node_features
    model = build_model(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        heads=args.heads,
        dropout=args.dropout
    )
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Train
    trainer = Trainer(model, graph_data, labels_df, args)
    history, final_test = trainer.train()

    # Save logs
    logs = {
        'args':        vars(args),
        'history':     history,
        'final_test':  final_test,
        'best_epoch':  trainer.best_epoch,
        'best_val_auroc': trainer.best_val_auroc
    }
    with open(args.logs, 'w') as f:
        json.dump(logs, f, indent=2)
    logger.info(f"Training logs saved to {args.logs}")
    logger.info("Training complete ✓")


if __name__ == '__main__':
    main()
