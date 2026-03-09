#!/usr/bin/env python3
"""Merge per-position conservation scores with variants."""
import argparse, logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variants',     required=True)
    p.add_argument('--conservation', required=True)
    p.add_argument('--output',       default='cons_features.tsv')
    return p.parse_args()

def main():
    args = parse_args()
    var_df  = pd.read_csv(args.variants,     sep='\t')
    cons_df = pd.read_csv(args.conservation, sep='\t')
    logger.info(f"Variants: {len(var_df)} | Conservation positions: {len(cons_df)}")

    # Normalise column names
    cons_df = cons_df.rename(columns={'chr':'chrom','chromosome':'chrom','position':'pos','start':'pos'})

    # Convert chrom to str for consistent merge
    var_df['chrom']  = var_df['chrom'].astype(str)
    cons_df['chrom'] = cons_df['chrom'].astype(str)

    merged = var_df[['variant_id','chrom','pos']].merge(
        cons_df, on=['chrom','pos'], how='left'
    )

    out = pd.DataFrame({'variant_id': merged['variant_id']})
    out['phylop100']    = merged.get('phyloP100way_vertebrate',    pd.Series(np.zeros(len(merged)))).fillna(0).values
    out['phastcons100'] = merged.get('phastCons100way_vertebrate', pd.Series(np.zeros(len(merged)))).fillna(0).values
    out['gerp_rs']      = merged.get('GERP++_RS',                  pd.Series(np.zeros(len(merged)))).fillna(0).values
    out['gerp_nr']      = merged.get('GERP++_NR',                  pd.Series(np.zeros(len(merged)))).fillna(0).values
    out['siphy']        = merged.get('SiPhy_29way_logOdds',        pd.Series(np.zeros(len(merged)))).fillna(0).values

    score_cols = ['phylop100', 'phastcons100', 'gerp_rs']
    out['mean_conservation'] = out[score_cols].mean(axis=1)
    out['max_conservation']  = out[score_cols].max(axis=1)

    out.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Conservation features: {out.shape} → {args.output}")

if __name__ == '__main__':
    main()
