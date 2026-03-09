#!/usr/bin/env python3
"""Filter variants to retain only rare variants below AF threshold."""
import argparse, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--af-max', type=float, default=0.001)
    p.add_argument('--output', default='filtered_variants.tsv')
    p.add_argument('--stats',  default='filter_stats.txt')
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep='\t')
    n_orig = len(df)
    logger.info(f"Input: {n_orig} variants")

    df = df[df['AF'].fillna(0) < args.af_max]
    n_rare = len(df)

    if 'filter' in df.columns:
        df = df[df['filter'].isin(['PASS', '.', 'pass', '', 'PASS\n'])]

    df = df[~df['alt'].astype(str).str.contains('<|>', na=False)]
    n_final = len(df)

    df.to_csv(args.output, sep='\t', index=False)
    with open(args.stats, 'w') as f:
        f.write(f"input\t{n_orig}\n")
        f.write(f"after_af_filter\t{n_rare}\n")
        f.write(f"after_quality_filter\t{n_final}\n")
        f.write(f"removed\t{n_orig - n_final}\n")
    logger.info(f"After filtering: {n_final} rare variants (AF < {args.af_max})")

if __name__ == '__main__':
    main()
