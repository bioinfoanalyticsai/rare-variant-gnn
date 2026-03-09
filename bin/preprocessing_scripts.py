#!/usr/bin/env python3
"""
========================================================================================
    bin/parse_vcf.py
    Parse a VCF/VCF.gz file into a structured TSV.
    Extracts: chrom, pos, ref, alt, variant_id, AF, AC, AN, QUAL, FILTER
========================================================================================
"""
import argparse, gzip, logging, re
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vcf',           required=True)
    p.add_argument('--genome-build',  default='GRCh38')
    p.add_argument('--output',        default='parsed_variants.tsv')
    p.add_argument('--stats',         default='vcf_stats.txt')
    return p.parse_args()


def open_vcf(path: str):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'rt')


def parse_info(info_str: str) -> dict:
    d = {}
    for field in info_str.split(';'):
        if '=' in field:
            k, v = field.split('=', 1)
            d[k] = v
        else:
            d[field] = True
    return d


def main():
    args = parse_args()
    records = []
    n_total = 0
    n_multi = 0

    with open_vcf(args.vcf) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            n_total += 1
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue

            chrom, pos, vid, ref, alt_field, qual, flt, info_str = parts[:8]

            # Handle multi-allelic sites
            for alt in alt_field.split(','):
                alt = alt.strip()
                if alt in ('.', '*', '<NON_REF>'):
                    continue

                info = parse_info(info_str)
                af_raw = info.get('AF', info.get('MAF', '0.0'))
                if isinstance(af_raw, str) and ',' in af_raw:
                    af_raw = af_raw.split(',')[0]
                try:
                    af = float(af_raw)
                except (ValueError, TypeError):
                    af = 0.0

                variant_id = (vid if vid != '.' else
                              f"{chrom}:{pos}:{ref}:{alt}")

                records.append({
                    'variant_id': variant_id,
                    'chrom':      chrom,
                    'pos':        int(pos),
                    'ref':        ref,
                    'alt':        alt,
                    'qual':       qual,
                    'filter':     flt,
                    'AF':         af,
                    'AC':         int(info.get('AC', 0)),
                    'AN':         int(info.get('AN', 0)),
                    'variant_type': ('SNP' if len(ref) == len(alt) == 1
                                    else 'INDEL'),
                })

    df = pd.DataFrame(records)
    df.to_csv(args.output, sep='\t', index=False)

    with open(args.stats, 'w') as f:
        f.write(f"total_records\t{n_total}\n")
        f.write(f"parsed_variants\t{len(df)}\n")
        f.write(f"snps\t{(df['variant_type']=='SNP').sum()}\n")
        f.write(f"indels\t{(df['variant_type']=='INDEL').sum()}\n")

    logger.info(f"Parsed {len(df)} variants from {n_total} records → {args.output}")


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────────────────────────


#!/usr/bin/env python3
"""
========================================================================================
    bin/filter_variants.py
    Filter variants to retain only rare variants (AF < threshold).
    Also removes common polymorphisms and low-quality calls.
========================================================================================
"""
import argparse, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--af-max', type=float, default=0.001, help='Max allele frequency')
    p.add_argument('--output', default='filtered_variants.tsv')
    p.add_argument('--stats',  default='filter_stats.txt')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep='\t')
    n_orig = len(df)
    logger.info(f"Input: {n_orig} variants")

    # Rare variant filter
    df = df[df['AF'].fillna(0) < args.af_max]
    n_rare = len(df)

    # Quality filter (keep PASS variants if FILTER column exists)
    if 'filter' in df.columns:
        df = df[df['filter'].isin(['PASS', '.', 'pass', ''])]

    # Remove structural variants and complex alleles
    df = df[~df['alt'].str.contains('<|>', na=False)]

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


# ─────────────────────────────────────────────────────────────────────────────


#!/usr/bin/env python3
"""
========================================================================================
    bin/merge_labels.py
    Merge annotated variants with known pathogenicity labels (ClinVar/HGMD).

    Label encoding:
        1 = pathogenic / likely pathogenic
        0 = benign / likely benign
       -1 = VUS / unknown (excluded from supervised training)
========================================================================================
"""
import argparse, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PATHOGENIC_TERMS = {
    'pathogenic', 'likely_pathogenic', 'pathogenic/likely_pathogenic',
    'likely pathogenic', 'pathogenic, likely pathogenic'
}
BENIGN_TERMS = {
    'benign', 'likely_benign', 'benign/likely_benign',
    'likely benign', 'benign, likely benign'
}


def label_clinsig(clinsig: str) -> int:
    s = str(clinsig).lower().strip()
    if any(t in s for t in PATHOGENIC_TERMS):
        return 1
    if any(t in s for t in BENIGN_TERMS):
        return 0
    return -1  # VUS or unknown


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

    # Standardize label column
    if 'clinical_significance' in labels_df.columns:
        labels_df['label'] = labels_df['clinical_significance'].apply(label_clinsig)
    elif 'CLNSIG' in labels_df.columns:
        labels_df['label'] = labels_df['CLNSIG'].apply(label_clinsig)
    else:
        labels_df['label'] = labels_df.get('label', -1)

    # Merge on variant_id (primary) or chrom+pos+ref+alt
    merged = variants_df.merge(
        labels_df[['variant_id', 'label']],
        on='variant_id',
        how='left'
    )
    merged['label'] = merged['label'].fillna(-1).astype(int)

    merged.to_csv(args.output, sep='\t', index=False)

    n_path = (merged['label'] == 1).sum()
    n_benign = (merged['label'] == 0).sum()
    n_vus = (merged['label'] == -1).sum()

    with open(args.stats, 'w') as f:
        f.write(f"total\t{len(merged)}\n")
        f.write(f"pathogenic\t{n_path}\n")
        f.write(f"benign\t{n_benign}\n")
        f.write(f"vus_unknown\t{n_vus}\n")

    logger.info(f"Labels: {n_path} pathogenic, {n_benign} benign, {n_vus} VUS")


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────────────────────────


#!/usr/bin/env python3
"""
========================================================================================
    bin/split_dataset.py
    Create stratified train/val/test splits with gene-level holdout.

    Important: Splits are done at GENE level (not variant level) to prevent
    data leakage from the same gene appearing in train and test.
========================================================================================
"""
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

    # Only use labeled variants (exclude VUS)
    labeled = df[df['label'].isin([0, 1])].copy()
    logger.info(f"Labeled variants for splitting: {len(labeled)}")

    # Gene-level split to avoid leakage
    if 'gene' in labeled.columns:
        genes = labeled['gene'].unique()
        gene_labels = labeled.groupby('gene')['label'].max().reset_index()

        # Stratified gene split
        train_genes, temp_genes = train_test_split(
            gene_labels['gene'].values,
            test_size=(args.val + args.test),
            random_state=args.seed,
            stratify=gene_labels['label'].values
        )
        val_genes, test_genes = train_test_split(
            temp_genes,
            test_size=args.test / (args.val + args.test),
            random_state=args.seed
        )

        labeled.loc[labeled['gene'].isin(train_genes), 'split'] = 'train'
        labeled.loc[labeled['gene'].isin(val_genes),   'split'] = 'val'
        labeled.loc[labeled['gene'].isin(test_genes),  'split'] = 'test'
    else:
        # Variant-level stratified split
        train_idx, temp_idx = train_test_split(
            labeled.index,
            test_size=(args.val + args.test),
            random_state=args.seed,
            stratify=labeled['label']
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=args.test / (args.val + args.test),
            random_state=args.seed,
            stratify=labeled.loc[temp_idx, 'label']
        )
        labeled.loc[train_idx, 'split'] = 'train'
        labeled.loc[val_idx,   'split'] = 'val'
        labeled.loc[test_idx,  'split'] = 'test'

    # Assign node_idx (row index acts as node ID in graph)
    labeled['node_idx'] = range(len(labeled))

    output_cols = ['variant_id', 'node_idx', 'label', 'split']
    if 'gene' in labeled.columns:
        output_cols.append('gene')

    labeled[output_cols].to_csv(args.output, sep='\t', index=False)

    for split in ['train', 'val', 'test']:
        s = labeled[labeled['split'] == split]
        logger.info(f"  {split}: {len(s)} variants "
                    f"({(s['label']==1).sum()} pathogenic, {(s['label']==0).sum()} benign)")

    with open(args.stats, 'w') as f:
        for split in ['train', 'val', 'test']:
            s = labeled[labeled['split'] == split]
            f.write(f"{split}_total\t{len(s)}\n")
            f.write(f"{split}_pathogenic\t{(s['label']==1).sum()}\n")
            f.write(f"{split}_benign\t{(s['label']==0).sum()}\n")


if __name__ == '__main__':
    main()
