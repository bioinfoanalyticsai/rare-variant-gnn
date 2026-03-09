#!/usr/bin/env python3
"""Extract sequence-based pathogenicity features per variant."""
import argparse, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

AA_HYDRO = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
            'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
            'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2,'X':0.0,'*':0.0}
AA_CHARGE = {'R':1,'K':1,'D':-1,'E':-1,'H':0.5,'A':0,'N':0,'C':0,'Q':0,'G':0,
             'I':0,'L':0,'M':0,'F':0,'P':0,'S':0,'T':0,'W':0,'Y':0,'V':0,'X':0,'*':0}
BLOSUM_SELF = {'A':4,'R':5,'N':6,'D':6,'C':9,'Q':5,'E':5,'G':6,'H':8,'I':4,'L':4,
               'K':5,'M':5,'F':6,'P':7,'S':4,'T':5,'W':11,'Y':7,'V':4,'X':0,'*':0}

def aa_feats(ref, alt):
    r = str(ref).upper() if isinstance(ref, str) else 'X'
    a = str(alt).upper() if isinstance(alt, str) else 'X'
    r = r if r in AA_HYDRO else 'X'
    a = a if a in AA_HYDRO else 'X'
    return {
        'hydrophobicity_change': AA_HYDRO[a] - AA_HYDRO[r],
        'charge_change':         AA_CHARGE[a] - AA_CHARGE[r],
        'blosum_score_diff':     BLOSUM_SELF[a] - BLOSUM_SELF[r],
        'is_charge_reversal':    int(np.sign(AA_CHARGE[r]) != np.sign(AA_CHARGE[a])
                                     and AA_CHARGE[r] != 0 and AA_CHARGE[a] != 0),
        'is_stop_gain':          int(a == '*'),
        'is_stop_loss':          int(r == '*' and a != '*'),
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

    feats = []
    for _, row in df.iterrows():
        vid = row.get('variant_id', f"{row.get('chrom','.')}:{row.get('pos','.')}:{row.get('ref','.')}:{row.get('alt','.')}")
        f = {'variant_id': vid, 'gene': row.get('gene', 'UNKNOWN')}
        f['cadd_phred']    = float(row.get('CADD_PHRED',   0) or 0)
        f['cadd_raw']      = float(row.get('CADD_RAW',     0) or 0)
        f['sift_score']    = float(row.get('SIFT_score', 0.5) or 0.5)
        f['sift_pred_dam'] = int(str(row.get('SIFT_pred', '')).startswith('D'))
        f['polyphen_hdiv'] = float(row.get('Polyphen2_HDIV_score', 0) or 0)
        f['polyphen_hvar'] = float(row.get('Polyphen2_HVAR_score', 0) or 0)
        f['revel_score']   = float(row.get('REVEL_score',  0) or 0)
        f['m_cap_score']   = float(row.get('M_CAP_score',  0) or 0)
        f.update(aa_feats(row.get('ref_aa', 'X'), row.get('alt_aa', 'X')))
        csq = str(row.get('consequence', '')).upper()
        f['is_missense']   = int('MISSENSE'  in csq)
        f['is_nonsense']   = int('STOP_GAIN' in csq or 'NONSENSE' in csq)
        f['is_frameshift'] = int('FRAME'     in csq)
        f['is_splice']     = int('SPLICE'    in csq)
        f['is_synonymous'] = int('SYNONYMOUS' in csq)
        try:
            pp = float(row.get('protein_position', 0) or 0)
            pl = float(row.get('protein_length',   1) or 1)
            f['relative_protein_position'] = pp / max(pl, 1)
        except (TypeError, ValueError):
            f['relative_protein_position'] = 0.0
        af = float(row.get('AF', 1e-6) or 1e-6)
        f['log10_af'] = float(np.log10(af + 1e-10))
        feats.append(f)

    out = pd.DataFrame(feats)
    out.to_csv(args.output, sep='\t', index=False)
    logger.info(f"Sequence features: {out.shape} → {args.output}")

if __name__ == '__main__':
    main()
