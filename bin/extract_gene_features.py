#!/usr/bin/env python3
"""Merge gene-level constraint and expression features onto variants."""
import argparse, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variants',    required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--output',      default='gene_features.tsv')
    return p.parse_args()

def main():
    args = parse_args()
    var_df  = pd.read_csv(args.variants,    sep='\t')
    gene_df = pd.read_csv(args.annotations, sep='\t')
    logger.info(f"Variants: {len(var_df)} | Gene annotations: {len(gene_df)}")

    if 'gene' not in var_df.columns:
        var_df['gene'] = 'UNKNOWN'

    gene_cols = [c for c in gene_df.columns if c != 'gene']
    merged = var_df[['variant_id', 'gene']].merge(gene_df, on='gene', how='left')

    # Fill missing gene annotations with population medians
    for col in gene_cols:
        if merged[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            merged[col] = merged[col].fillna(merged[col].median())
        else:
            merged[col] = merged[col].fillna(0)

    out_cols = ['variant_id', 'gene'] + gene_cols
    merged[out_cols].to_csv(args.output, sep='\t', index=False)
    logger.info(f"Gene features: {merged[out_cols].shape} → {args.output}")

if __name__ == '__main__':
    main()
