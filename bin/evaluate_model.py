#!/usr/bin/env python3
"""
========================================================================================
    bin/evaluate_model.py
    Comprehensive model evaluation: metrics, ROC/PR curves, calibration, feature importance.
    Generates an HTML report with embedded plots.
========================================================================================
"""
import argparse, json, logging, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    brier_score_loss, f1_score
)
from sklearn.calibration import calibration_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True, help='predictions TSV from predict_gnn.py')
    p.add_argument('--labels',      required=True, help='split_labels.tsv from preprocessing')
    p.add_argument('--metrics',     default='metrics.json')
    p.add_argument('--plots-dir',   default='plots/')
    p.add_argument('--report',      default='eval_report.html')
    return p.parse_args()


def find_optimal_threshold(labels, probs):
    """Find F1-optimal decision threshold."""
    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores = [f1_score(labels, (probs >= t).astype(int), zero_division=0)
                 for t in thresholds]
    best_t = thresholds[np.argmax(f1_scores)]
    return best_t, float(max(f1_scores))


def compute_all_metrics(labels, probs, split: str = 'test') -> dict:
    preds_05 = (probs >= 0.5).astype(int)
    opt_t, opt_f1 = find_optimal_threshold(labels, probs)
    preds_opt = (probs >= opt_t).astype(int)

    metrics = {
        'split':          split,
        'n_total':        int(len(labels)),
        'n_positive':     int(labels.sum()),
        'n_negative':     int((1 - labels).sum()),
        'prevalence':     float(labels.mean()),
        'auroc':          float(roc_auc_score(labels, probs)),
        'auprc':          float(average_precision_score(labels, probs)),
        'brier_score':    float(brier_score_loss(labels, probs)),
        'optimal_threshold': float(opt_t),
        'f1_at_05':       float(f1_score(labels, preds_05, zero_division=0)),
        'f1_optimal':     float(opt_f1),
    }

    # At 0.5 threshold
    cm = confusion_matrix(labels, preds_05)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])
    metrics.update({
        'sensitivity': float(tp / (tp + fn + 1e-9)),
        'specificity': float(tn / (tn + fp + 1e-9)),
        'ppv':         float(tp / (tp + fp + 1e-9)),
        'npv':         float(tn / (tn + fn + 1e-9)),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    })
    return metrics


def plot_roc_pr(labels, probs, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('GNN Rare Variant Pathogenicity — Model Performance', fontsize=14, fontweight='bold')

    # ── ROC Curve ──────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, 'b-', lw=2, label=f'GNN (AUROC = {auroc:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='blue')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curve', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1.02])
    axes[0].grid(alpha=0.3)

    # ── PR Curve ───────────────────────────────────────────────────────────
    precision, recall, _ = precision_recall_curve(labels, probs)
    auprc = average_precision_score(labels, probs)
    prevalence = labels.mean()
    axes[1].plot(recall, precision, 'r-', lw=2, label=f'GNN (AUPRC = {auprc:.4f})')
    axes[1].axhline(y=prevalence, color='k', ls='--', lw=1, alpha=0.5,
                    label=f'Baseline ({prevalence:.3f})')
    axes[1].fill_between(recall, precision, alpha=0.1, color='red')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curve', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1.02])
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'roc_pr_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC/PR plot saved: {path}")
    return path


def plot_score_distribution(labels, probs, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Pathogenicity Score Distribution', fontsize=14, fontweight='bold')

    # Histogram
    benign_probs = probs[labels == 0]
    patho_probs  = probs[labels == 1]
    axes[0].hist(benign_probs,  bins=50, alpha=0.6, color='steelblue', label=f'Benign (n={len(benign_probs)})', density=True)
    axes[0].hist(patho_probs,   bins=50, alpha=0.6, color='tomato',    label=f'Pathogenic (n={len(patho_probs)})', density=True)
    axes[0].axvline(0.5, color='black', ls='--', lw=2, label='Threshold=0.5')
    axes[0].set_xlabel('Pathogenicity Score', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Score Distribution by Class', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Calibration
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10)
    axes[1].plot(mean_pred, frac_pos, 's-', color='blue', lw=2, ms=6, label='GNN')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Perfect calibration')
    axes[1].set_xlabel('Mean Predicted Probability', fontsize=12)
    axes[1].set_ylabel('Fraction of Positives', fontsize=12)
    axes[1].set_title('Calibration Curve', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(outdir, 'score_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_confusion_matrix(labels, probs, outdir):
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Benign', 'Pathogenic'], fontsize=12)
    ax.set_yticklabels(['Benign', 'Pathogenic'], fontsize=12)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('True', fontsize=13)
    ax.set_title('Confusion Matrix (threshold=0.5)', fontsize=13)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, fontweight='bold',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(outdir, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def generate_html_report(metrics: dict, plot_paths: list, outpath: str):
    html = f"""<!DOCTYPE html>
<html>
<head>
<title>GNN Variant Pathogenicity — Evaluation Report</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; margin: 40px; background: #f8f9fa; color: #333; }}
  h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
  h2 {{ color: #34495e; margin-top: 30px; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
  .metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; }}
  .metric-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
  .metric-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
  .good {{ color: #27ae60; }}
  .warn {{ color: #e67e22; }}
  table {{ background: white; border-collapse: collapse; width: 100%; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
  tr:hover {{ background: #f5f9ff; }}
  img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0; }}
  .plot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
</style>
</head>
<body>
<h1>🧬 GNN Rare Variant Pathogenicity — Evaluation Report</h1>
<p>Generated from test set evaluation of the Graph Attention Network model.</p>

<h2>📊 Key Performance Metrics</h2>
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-value {'good' if metrics['auroc'] > 0.85 else 'warn'}">{metrics['auroc']:.4f}</div>
    <div class="metric-label">AUROC</div>
  </div>
  <div class="metric-card">
    <div class="metric-value {'good' if metrics['auprc'] > 0.7 else 'warn'}">{metrics['auprc']:.4f}</div>
    <div class="metric-label">AUPRC</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{metrics['sensitivity']:.4f}</div>
    <div class="metric-label">Sensitivity (Recall)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{metrics['specificity']:.4f}</div>
    <div class="metric-label">Specificity</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{metrics['ppv']:.4f}</div>
    <div class="metric-label">PPV (Precision)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{metrics['npv']:.4f}</div>
    <div class="metric-label">NPV</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{metrics['f1_at_05']:.4f}</div>
    <div class="metric-label">F1 (threshold=0.5)</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{metrics['brier_score']:.4f}</div>
    <div class="metric-label">Brier Score</div>
  </div>
</div>

<h2>📋 Detailed Metrics</h2>
<table>
<tr><th>Metric</th><th>Value</th></tr>
{"".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items())}
</table>

<h2>📈 Performance Plots</h2>
<div class="plot-grid">
{"".join(f'<img src="{p}" alt="plot" />' for p in plot_paths)}
</div>

<p style="color:#999;font-size:0.8em;margin-top:40px;">Generated by Rare Variant GNN Pipeline</p>
</body>
</html>"""

    with open(outpath, 'w') as f:
        f.write(html)
    logger.info(f"HTML report saved to {outpath}")



def main():
    args = parse_args()
    os.makedirs(args.plots_dir, exist_ok=True)

    # ── Load files ────────────────────────────────────────────────────────────
    preds_df  = pd.read_csv(args.predictions, sep='\t')
    labels_df = pd.read_csv(args.labels,      sep='\t')

    logger.info(f"Predictions columns: {list(preds_df.columns)}")
    logger.info(f"Labels columns:      {list(labels_df.columns)}")

    # ── Determine probability column ─────────────────────────────────────────
    # Try predictions.tsv first, then probabilities.tsv if present
    if 'pathogenicity_prob' in preds_df.columns:
        prob_col = 'pathogenicity_prob'
    elif 'prediction' in preds_df.columns:
        prob_col = 'prediction'
    else:
        raise ValueError(f"No probability column found in predictions. "
                         f"Columns: {list(preds_df.columns)}")

    # ── Drop any 'label' column from predictions to avoid conflict ────────────
    if 'label' in preds_df.columns:
        preds_df = preds_df.drop(columns=['label'])

    # ── Merge predictions with ground-truth labels ────────────────────────────
    # Try node_idx first, fall back to variant_id
    label_cols = ['label', 'split']
    if 'node_idx' in preds_df.columns and 'node_idx' in labels_df.columns:
        merge_on = 'node_idx'
    elif 'variant_id' in preds_df.columns and 'variant_id' in labels_df.columns:
        merge_on = 'variant_id'
    elif 'node_id' in preds_df.columns and 'node_idx' in labels_df.columns:
        preds_df = preds_df.rename(columns={'node_id': 'node_idx'})
        merge_on = 'node_idx'
    else:
        # Last resort: merge by position (same row order assumed)
        logger.warning("No common key column found — merging by position")
        n = min(len(preds_df), len(labels_df))
        merged = preds_df.iloc[:n].copy().reset_index(drop=True)
        merged['label'] = labels_df['label'].iloc[:n].values
        merged['split'] = labels_df.get('split', pd.Series(['test'] * n)).iloc[:n].values
        merge_on = None

    if merge_on is not None:
        keep_label_cols = [c for c in ['node_idx', 'variant_id', 'label', 'split']
                           if c in labels_df.columns]
        merged = preds_df.merge(labels_df[keep_label_cols], on=merge_on, how='inner')

    logger.info(f"Merged rows: {len(merged)}")

    if 'label' not in merged.columns:
        raise ValueError(f"'label' column missing after merge. "
                         f"Available: {list(merged.columns)}")

    # Drop VUS (label == -1) — only evaluate on 0/1 labeled variants
    merged = merged[merged['label'].isin([0, 1])].copy()
    logger.info(f"Labeled variants (0/1): {len(merged)}")

    # ── Select evaluation split ───────────────────────────────────────────────
    if 'split' in merged.columns:
        test_df = merged[merged['split'] == 'test']
        if len(test_df) == 0:
            logger.warning("No 'test' split rows — evaluating on all labeled data")
            test_df = merged
    else:
        test_df = merged

    if len(test_df) == 0:
        logger.warning("No labeled test data available — writing empty metrics")
        metrics = {'auroc': None, 'auprc': None, 'n_test': 0, 'note': 'no_labeled_test_data'}
        with open(args.metrics, 'w') as f:
            json.dump(metrics, f, indent=2)
        generate_html_report(metrics, [], args.report)
        return

    labels = test_df['label'].values.astype(int)
    probs  = test_df[prob_col].values.astype(float)

    # Clamp probs to valid range
    probs = np.clip(probs, 1e-7, 1 - 1e-7)

    n_path = labels.sum()
    n_ben  = (labels == 0).sum()
    logger.info(f"Evaluating {len(test_df)} test variants: {n_path} pathogenic, {n_ben} benign")

    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present in test set — skipping AUC metrics")
        metrics = {
            'n_test': int(len(labels)),
            'n_pathogenic': int(n_path),
            'n_benign': int(n_ben),
            'note': 'only_one_class_in_test_set'
        }
    else:
        metrics = compute_all_metrics(labels, probs, split='test')
        logger.info(f"Test AUROC: {metrics.get('auroc', 'N/A'):.4f} | "
                    f"AUPRC: {metrics.get('auprc', 'N/A'):.4f}")

    with open(args.metrics, 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_paths = []
    try:
        plot_paths.append(plot_roc_pr(labels, probs, args.plots_dir))
    except Exception as e:
        logger.warning(f"ROC/PR plot failed: {e}")
    try:
        plot_paths.append(plot_score_distribution(labels, probs, args.plots_dir))
    except Exception as e:
        logger.warning(f"Score distribution plot failed: {e}")
    try:
        plot_paths.append(plot_confusion_matrix(labels, probs, args.plots_dir))
    except Exception as e:
        logger.warning(f"Confusion matrix plot failed: {e}")

    generate_html_report(metrics, [p for p in plot_paths if p], args.report)
    logger.info("Evaluation complete ✓")


if __name__ == '__main__':
    main()
