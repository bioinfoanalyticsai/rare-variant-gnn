#!/usr/bin/env python3
"""Merge annotated variants with known ClinVar/HGMD pathogenicity labels."""
import argparse, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATHOGENIC = {'pathogenic', 'likely_pathogenic', 'pathogenic/likely_pathogenic',
              'likely pathogenic', 'pathogenic, likely pathogenic'}
BENIGN     = {'benign', 'likely_benign', 'benign/likely_benign',
              'likely benign', 'benign, likely benign'}

def label_clinsig(val):
    s = str(val).lower().strip()
    if any(t in s for t in PATHOGENIC):
        return 1
    if any(t in s for t in BENIGN):
        return 0
    return -1

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', required=True)
    p.add_argument('--labels',   required=True)
    p.add_argument('--output',   default='merged_variants.tsv')
    p.add_argument('--stats',    default='label_stats.txt')
    return p.parse_args()

def main():
    args = parse_args()
    variants_df = pd.read_csv(args.variants, sep='\t')
    labels_df   = pd.read_csv(args.labels,   sep='\t')
    logger.info(f"Variants: {len(variants_df)} | Labels: {len(labels_df)}")

    # Determine label column
    if 'clinical_significance' in labels_df.columns:
        labels_df['label'] = labels_df['clinical_significance'].apply(label_clinsig)
    elif 'CLNSIG' in labels_df.columns:
        labels_df['label'] = labels_df['CLNSIG'].apply(label_clinsig)
    elif 'label' not in labels_df.columns:
        labels_df['label'] = -1

    label_map = labels_df.set_index('variant_id')['label'].to_dict()
    variants_df['label'] = variants_df['variant_id'].map(label_map).fillna(-1).astype(int)

    # Also pull through annotation columns from labels if not already present
    extra_cols = [c for c in labels_df.columns
                  if c not in variants_df.columns and c not in ('variant_id', 'label')]
    if extra_cols:
        variants_df = variants_df.merge(
            labels_df[['variant_id'] + extra_cols],
            on='variant_id', how='left'
        )

    variants_df.to_csv(args.output, sep='\t', index=False)
    n_p = (variants_df['label'] == 1).sum()
    n_b = (variants_df['label'] == 0).sum()
    n_v = (variants_df['label'] == -1).sum()
    with open(args.stats, 'w') as f:
        f.write(f"total\t{len(variants_df)}\n")
        f.write(f"pathogenic\t{n_p}\n")
        f.write(f"benign\t{n_b}\n")
        f.write(f"vus_unknown\t{n_v}\n")
    logger.info(f"Merged: {n_p} pathogenic, {n_b} benign, {n_v} VUS → {args.output}")

if __name__ == '__main__':
    main()
