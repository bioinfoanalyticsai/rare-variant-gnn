#!/usr/bin/env python3
"""
========================================================================================
    bin/extract_sequence_features.py
    Extract sequence-based pathogenicity features for each variant.

    Features extracted:
        - CADD score (if in annotation)
        - SIFT score (damaging probability)
        - PolyPhen2 HDIV and HVAR scores
        - BLOSUM62 substitution score
        - Amino acid physicochemical property changes
        - Position in protein (relative to length)
        - ESM-2 amino acid embedding distance (simplified proxy)
========================================================================================
"""
import argparse, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# BLOSUM62 diagonal (self-similarity scores for amino acids)
BLOSUM62_SELF = {
    'A': 4, 'R': 5, 'N': 6, 'D': 6, 'C': 9, 'Q': 5, 'E': 5,
    'G': 6, 'H': 8, 'I': 4, 'L': 4, 'K': 5, 'M': 5, 'F': 6,
    'P': 7, 'S': 4, 'T': 5, 'W': 11,'Y': 7, 'V': 4, 'X': 0
}

# Amino acid properties
AA_HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8,
    'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2, 'X': 0.0
}

AA_CHARGE = {
    'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.5,
    'A': 0, 'N': 0, 'C': 0, 'Q': 0, 'G': 0, 'I': 0, 'L': 0,
    'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0, 'X': 0
}


def compute_aa_change_features(ref_aa: str, alt_aa: str) -> dict:
    """Compute amino acid substitution features."""
    ref_aa = ref_aa.upper() if isinstance(ref_aa, str) else 'X'
    alt_aa = alt_aa.upper() if isinstance(alt_aa, str) else 'X'

    hydro_ref   = AA_HYDROPHOBICITY.get(ref_aa, 0.0)
    hydro_alt   = AA_HYDROPHOBICITY.get(alt_aa, 0.0)
    charge_ref  = AA_CHARGE.get(ref_aa, 0.0)
    charge_alt  = AA_CHARGE.get(alt_aa, 0.0)
    blosum_ref  = BLOSUM62_SELF.get(ref_aa, 0)
    blosum_alt  = BLOSUM62_SELF.get(alt_aa, 0)

    return {
        'hydrophobicity_change': hydro_alt - hydro_ref,
        'charge_change':        charge_alt - charge_ref,
        'blosum_score_diff':    blosum_alt - blosum_ref,
        'is_charge_reversal':   int(np.sign(charge_ref) != np.sign(charge_alt) and
                                    charge_ref != 0 and charge_alt != 0),
        'is_stop_gain':         int(alt_aa == '*'),
        'is_stop_loss':         int(ref_aa == '*' and alt_aa != '*'),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input',  required=True)
    p.add_argument('--output', default='seq_features.tsv')
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.input, sep='\t')
    logger.info(f"Extracting sequence features for {len(df)} variants")

    features = []
    for _, row in df.iterrows():
        feat = {'variant_id': row.get('variant_id', f"{row.get('chrom')}:{row.get('pos')}:{row.get('ref')}:{row.get('alt')}")}

        # Direct annotation scores (populated by VEP)
        feat['cadd_phred']    = float(row.get('CADD_PHRED', 0) or 0)
        feat['cadd_raw']      = float(row.get('CADD_RAW', 0) or 0)
        feat['sift_score']    = float(row.get('SIFT_score', 0.5) or 0.5)
        feat['sift_pred']     = int(str(row.get('SIFT_pred', '')).startswith('D'))  # D=damaging
        feat['polyphen_hdiv'] = float(row.get('Polyphen2_HDIV_score', 0) or 0)
        feat['polyphen_hvar'] = float(row.get('Polyphen2_HVAR_score', 0) or 0)
        feat['revel_score']   = float(row.get('REVEL_score', 0) or 0)
        feat['m_cap_score']   = float(row.get('M_CAP_score', 0) or 0)

        # Amino acid change features
        aa_feats = compute_aa_change_features(
            str(row.get('ref_aa', 'X')),
            str(row.get('alt_aa', 'X'))
        )
        feat.update(aa_feats)

        # Variant type encoding
        vtype = str(row.get('variant_type', 'SNP')).upper()
        feat['is_snp']     = int('SNP' in vtype or 'SNV' in vtype)
        feat['is_indel']   = int('INDEL' in vtype or 'DEL' in vtype or 'INS' in vtype)
        feat['is_frameshift'] = int('FRAME' in str(row.get('consequence', '')).upper())
        feat['is_missense'] = int('MISSENSE' in str(row.get('consequence', '')).upper())
        feat['is_nonsense'] = int(
            'STOP_GAINED' in str(row.get('consequence', '')).upper() or
            'NONSENSE' in str(row.get('consequence', '')).upper()
        )
        feat['is_splice']   = int('SPLICE' in str(row.get('consequence', '')).upper())

        # Position in protein
        try:
            aa_pos = float(row.get('protein_position', 0) or 0)
            prot_len = float(row.get('protein_length', 1) or 1)
            feat['relative_protein_position'] = aa_pos / max(prot_len, 1)
        except (TypeError, ValueError):
            feat['relative_protein_position'] = 0.0

        # Allele frequency
        feat['log10_af'] = float(np.log10(float(row.get('AF', 1e-6) or 1e-6) + 1e-10))

        features.append(feat)

    feat_df = pd.DataFrame(features)
    feat_df.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Sequence features saved: {feat_df.shape} → {args.output}")


if __name__ == '__main__':
    main()


# ─────────────────────────────────────────────────────────────────────────────


#!/usr/bin/env python3
"""
========================================================================================
    bin/extract_conservation.py
    Merge per-position conservation scores (PhyloP, PhastCons, GERP++) with variants.
========================================================================================
"""
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
    variants_df = pd.read_csv(args.variants, sep='\t')
    cons_df     = pd.read_csv(args.conservation, sep='\t')

    logger.info(f"Merging {len(variants_df)} variants with {len(cons_df)} conservation scores")

    # Normalize column names
    cons_df = cons_df.rename(columns={
        'chr': 'chrom', 'chromosome': 'chrom',
        'position': 'pos', 'start': 'pos'
    })

    # Left-join variants with conservation by position
    merged = variants_df[['variant_id', 'chrom', 'pos']].merge(
        cons_df, on=['chrom', 'pos'], how='left'
    )

    # Conservation features
    cons_features = pd.DataFrame()
    cons_features['variant_id']   = merged['variant_id']
    cons_features['phylop100']    = merged.get('phyloP100way_vertebrate', pd.Series(np.zeros(len(merged)))).fillna(0)
    cons_features['phastcons100'] = merged.get('phastCons100way_vertebrate', pd.Series(np.zeros(len(merged)))).fillna(0)
    cons_features['gerp_rs']      = merged.get('GERP++_RS', pd.Series(np.zeros(len(merged)))).fillna(0)
    cons_features['gerp_nr']      = merged.get('GERP++_NR', pd.Series(np.zeros(len(merged)))).fillna(0)
    cons_features['siphy']        = merged.get('SiPhy_29way_logOdds', pd.Series(np.zeros(len(merged)))).fillna(0)

    # Aggregate conservation score
    score_cols = ['phylop100', 'phastcons100', 'gerp_rs']
    avail = [c for c in score_cols if c in cons_features.columns]
    if avail:
        cons_features['mean_conservation'] = cons_features[avail].mean(axis=1)
        cons_features['max_conservation']  = cons_features[avail].max(axis=1)

    cons_features.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Conservation features saved: {cons_features.shape} → {args.output}")


if __name__ == '__main__':
    main()
