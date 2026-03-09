#!/usr/bin/env python3
"""Merge all feature matrices into a unified node feature matrix."""
import argparse, logging
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seq-features',    required=True)
    p.add_argument('--cons-features',   required=True)
    p.add_argument('--struct-features', required=True)
    p.add_argument('--gene-features',   required=True)
    p.add_argument('--output',          default='node_features.tsv')
    p.add_argument('--names',           default='feature_names.txt')
    return p.parse_args()

def main():
    args = parse_args()
    ID_COLS = {'variant_id', 'gene', 'chrom', 'pos', 'ref', 'alt', 'label', 'split'}

    dfs = []
    for fpath in [args.seq_features, args.cons_features,
                  args.struct_features, args.gene_features]:
        try:
            df = pd.read_csv(fpath, sep='\t')
            dfs.append(df)
            logger.info(f"Loaded {fpath}: {df.shape}")
        except FileNotFoundError:
            logger.warning(f"Missing: {fpath}, skipping")

    if not dfs:
        raise ValueError("No feature files found!")

    merged = dfs[0]
    for df in dfs[1:]:
        # Drop duplicate non-ID columns before merge to avoid _x/_y suffixes
        shared = [c for c in df.columns if c in merged.columns and c not in ('variant_id',)]
        df = df.drop(columns=shared, errors='ignore')
        merged = merged.merge(df, on='variant_id', how='outer')

    feature_cols = [c for c in merged.columns
                    if c not in ID_COLS
                    and pd.api.types.is_numeric_dtype(merged[c])]

    logger.info(f"Total numeric features: {len(feature_cols)}")

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(merged[feature_cols].values.astype(float))

    result = pd.DataFrame(X, columns=feature_cols)
    result.insert(0, 'variant_id', merged['variant_id'].values)
    if 'gene' in merged.columns:
        result.insert(1, 'gene', merged['gene'].values)

    result.to_csv(args.output, sep='\t', index=False)
    with open(args.names, 'w') as f:
        f.write('\n'.join(feature_cols))
    logger.info(f"Combined features: {result.shape} → {args.output}")

if __name__ == '__main__':
    main()
