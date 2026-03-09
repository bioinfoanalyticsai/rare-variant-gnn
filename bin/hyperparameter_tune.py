#!/usr/bin/env python3
"""
========================================================================================
    bin/hyperparameter_tune.py
    Optuna-based hyperparameter optimization for the GNN model.

    Search space:
        - hidden_channels: [64, 128, 256]
        - num_layers:       [2, 3, 4, 5]
        - attention_heads:  [2, 4, 8]
        - dropout:          [0.1, 0.5]
        - learning_rate:    [1e-4, 1e-2] (log scale)
        - weight_decay:     [1e-6, 1e-3] (log scale)
        - jk_mode:          ['last', 'cat', 'max']
========================================================================================
"""
import argparse, json, logging, pickle, sys
from pathlib import Path
import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gnn_model import build_model
from train_gnn import Trainer, load_graph, load_labels

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--graph',   required=True)
    p.add_argument('--labels',  required=True)
    p.add_argument('--trials',  type=int, default=50)
    p.add_argument('--timeout', type=int, default=3600, help='Time limit in seconds')
    p.add_argument('--output',  default='best_hparams.json')
    p.add_argument('--study',   default='hpo_study.pkl')
    p.add_argument('--seed',    type=int, default=42)
    return p.parse_args()


def objective(trial, graph_data, labels_df):
    """Optuna objective: returns validation AUROC."""
    import types

    # ── Sample hyperparameters ──────────────────────────────────────────────
    hidden    = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    layers    = trial.suggest_int('num_layers', 2, 5)
    heads     = trial.suggest_categorical('heads', [2, 4, 8])
    dropout   = trial.suggest_float('dropout', 0.1, 0.5)
    lr        = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    wd        = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    jk_mode   = trial.suggest_categorical('jk_mode', ['last', 'cat', 'max'])

    # ── Build and train model ───────────────────────────────────────────────
    model = build_model(
        in_channels=graph_data.num_node_features,
        hidden_channels=hidden,
        num_layers=layers,
        heads=heads,
        dropout=dropout,
        jk_mode=jk_mode
    )

    # Minimal args object
    args = types.SimpleNamespace(
        lr=lr,
        weight_decay=wd,
        epochs=30,               # Short runs for HPO
        patience=10,
        save_every=9999,         # Don't save during HPO
        seed=42,
        model_dir='/tmp/hpo_checkpoints/',
        best_model='/tmp/hpo_best.pt'
    )

    trainer = Trainer(model, graph_data, labels_df, args)
    history, _ = trainer.train()

    # Return best validation AUROC
    best_val_auroc = max([e['auroc'] for e in history['val']])

    # Report intermediate values for pruning
    for step, metrics in enumerate(history['val']):
        trial.report(metrics['auroc'], step)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_auroc


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info(f"Loading data for HPO ({args.trials} trials)...")
    graph_data = load_graph(args.graph)
    labels_df  = load_labels(args.labels)

    # Create study with median pruner
    sampler = TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='rare_variant_gnn_hpo'
    )

    logger.info(f"Starting HPO: {args.trials} trials, timeout={args.timeout}s")
    study.optimize(
        lambda trial: objective(trial, graph_data, labels_df),
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True
    )

    # Results
    best_trial = study.best_trial
    logger.info(f"\n{'='*50}")
    logger.info(f"Best trial #{best_trial.number}")
    logger.info(f"Best val AUROC: {best_trial.value:.4f}")
    logger.info(f"Best hyperparameters:")
    for k, v in best_trial.params.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"{'='*50}")

    # Save
    best_params = {
        'best_auroc': float(best_trial.value),
        'best_trial': int(best_trial.number),
        'params':     best_trial.params,
        'n_trials':   len(study.trials)
    }
    with open(args.output, 'w') as f:
        json.dump(best_params, f, indent=2)

    with open(args.study, 'wb') as f:
        pickle.dump(study, f)

    logger.info(f"Best hyperparameters saved to {args.output}")


if __name__ == '__main__':
    main()
