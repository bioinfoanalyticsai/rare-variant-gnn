#!/usr/bin/env python3
"""Gene-stratified train/val/test split to prevent data leakage."""
import argparse, logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--train',  type=float, default=0.7)
    p.add_argument('--val',    type=float, default=0.15)
    p.add_argument('--test',   type=float, default=0.15)
    p.add_argument('--seed',   type=int,   default=42)
    p.add_argument('--output', default='split_labels.tsv')
    p.add_argument('--stats',  default='split_stats.txt')
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    df = pd.read_csv(args.input, sep='\t')

    labeled = df[df['label'].isin([0, 1])].copy().reset_index(drop=True)
    logger.info(f"Labeled variants: {len(labeled)} "
                f"({(labeled['label']==1).sum()} pathogenic, {(labeled['label']==0).sum()} benign)")

    if len(labeled) < 6:
        logger.warning("Too few labeled variants — assigning all to train")
        labeled['split'] = 'train'
    elif 'gene' in labeled.columns and labeled['gene'].nunique() >= 3:
        # Gene-level stratified split
        gene_labels = labeled.groupby('gene')['label'].max().reset_index()
        try:
            tr_genes, tmp = train_test_split(
                gene_labels['gene'].values,
                test_size=(args.val + args.test),
                random_state=args.seed,
                stratify=gene_labels['label'].values
            )
            val_genes, te_genes = train_test_split(
                tmp, test_size=args.test / (args.val + args.test),
                random_state=args.seed
            )
            labeled.loc[labeled['gene'].isin(tr_genes),  'split'] = 'train'
            labeled.loc[labeled['gene'].isin(val_genes), 'split'] = 'val'
            labeled.loc[labeled['gene'].isin(te_genes),  'split'] = 'test'
            labeled['split'] = labeled['split'].fillna('train')
        except ValueError:
            labeled['split'] = 'train'
    else:
        # Variant-level stratified split
        try:
            tr_idx, tmp_idx = train_test_split(
                labeled.index, test_size=(args.val + args.test),
                random_state=args.seed, stratify=labeled['label']
            )
            val_idx, te_idx = train_test_split(
                tmp_idx, test_size=args.test / (args.val + args.test),
                random_state=args.seed, stratify=labeled.loc[tmp_idx, 'label']
            )
            labeled.loc[tr_idx,  'split'] = 'train'
            labeled.loc[val_idx, 'split'] = 'val'
            labeled.loc[te_idx,  'split'] = 'test'
        except ValueError:
            labeled['split'] = 'train'

    labeled['node_idx'] = range(len(labeled))

    out_cols = ['variant_id', 'node_idx', 'label', 'split']
    if 'gene' in labeled.columns:
        out_cols.append('gene')
    labeled[out_cols].to_csv(args.output, sep='\t', index=False)

    stats_lines = []
    for split in ['train', 'val', 'test']:
        s = labeled[labeled['split'] == split]
        n_p = (s['label'] == 1).sum()
        n_b = (s['label'] == 0).sum()
        logger.info(f"  {split:5s}: {len(s):4d} variants ({n_p} pathogenic, {n_b} benign)")
        stats_lines += [f"{split}_total\t{len(s)}", f"{split}_pathogenic\t{n_p}", f"{split}_benign\t{n_b}"]

    with open(args.stats, 'w') as f:
        f.write('\n'.join(stats_lines) + '\n')

if __name__ == '__main__':
    main()
