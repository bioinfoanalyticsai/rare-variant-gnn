#!/usr/bin/env python3
"""Parse a VCF/VCF.gz file into a structured TSV."""
import argparse, gzip, logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--vcf',          required=True)
    p.add_argument('--genome-build', default='GRCh38')
    p.add_argument('--output',       default='parsed_variants.tsv')
    p.add_argument('--stats',        default='vcf_stats.txt')
    return p.parse_args()

def open_vcf(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'rt')

def parse_info(info_str):
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
    with open_vcf(args.vcf) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            n_total += 1
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            chrom, pos, vid, ref, alt_field, qual, flt, info_str = parts[:8]
            for alt in alt_field.split(','):
                alt = alt.strip()
                if alt in ('.', '*', '<NON_REF>') or alt.startswith('<'):
                    continue
                info = parse_info(info_str)
                af_raw = info.get('AF', info.get('MAF', '0.0'))
                if isinstance(af_raw, str) and ',' in af_raw:
                    af_raw = af_raw.split(',')[0]
                try:
                    af = float(af_raw)
                except (ValueError, TypeError):
                    af = 0.0
                variant_id = vid if vid != '.' else f"{chrom}:{pos}:{ref}:{alt}"
                gene = info.get('GENE', info.get('gene', 'UNKNOWN'))
                csq  = info.get('CSQ', info.get('consequence', ''))
                records.append({
                    'variant_id':   variant_id,
                    'chrom':        chrom,
                    'pos':          int(pos),
                    'ref':          ref,
                    'alt':          alt,
                    'qual':         qual,
                    'filter':       flt,
                    'AF':           af,
                    'AC':           int(info.get('AC', 0)),
                    'AN':           int(info.get('AN', 0)),
                    'gene':         gene,
                    'consequence':  csq,
                    'variant_type': 'SNP' if len(ref) == len(alt) == 1 else 'INDEL',
                })
    df = pd.DataFrame(records)
    df.to_csv(args.output, sep='\t', index=False)
    with open(args.stats, 'w') as f:
        f.write(f"total_records\t{n_total}\n")
        f.write(f"parsed_variants\t{len(df)}\n")
        if len(df):
            f.write(f"snps\t{(df['variant_type']=='SNP').sum()}\n")
            f.write(f"indels\t{(df['variant_type']=='INDEL').sum()}\n")
    logger.info(f"Parsed {len(df)} variants → {args.output}")

if __name__ == '__main__':
    main()
