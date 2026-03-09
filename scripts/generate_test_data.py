#!/usr/bin/env python3
"""
    scripts/generate_test_data.py
    Generate synthetic test data for pipeline validation.
    No required arguments — run with: python scripts/generate_test_data.py
"""
import argparse, os, random
import numpy as np
import pandas as pd

GENES = [
    'BRCA1','BRCA2','TP53','PTEN','MLH1','MSH2','APC','VHL',
    'RET','MEN1','TSC1','TSC2','NF1','NF2','SMAD4','PALB2',
    'ATM','CHEK2','CDH1','STK11','MUTYH','EPCAM','RAD51C','RAD51D',
    'BARD1','BRIP1','NBN','RECQL','FANCD2','FANCA',
]
CHROMOSOMES = [str(i) for i in range(1,23)] + ['X']
BASES = list('ACGT')
AA_CODES = list('ACDEFGHIKLMNPQRSTVWY')
CONSEQUENCES = [
    'missense_variant','synonymous_variant','stop_gained',
    'frameshift_variant','splice_donor_variant','splice_acceptor_variant',
    'inframe_deletion','inframe_insertion','start_lost',
]

def generate_vcf(n_variants, outdir):
    lines = [
        '##fileformat=VCFv4.2',
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">',
        '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele Count">',
        '##INFO=<ID=AN,Number=1,Type=Integer,Description="Total allele count">',
        '##INFO=<ID=GENE,Number=1,Type=String,Description="Gene symbol">',
        '##INFO=<ID=CSQ,Number=1,Type=String,Description="Consequence">',
        '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO',
    ]
    for i in range(n_variants):
        chrom = random.choice(CHROMOSOMES)
        pos   = random.randint(1_000_000, 200_000_000)
        ref   = random.choice(BASES)
        alt   = random.choice([b for b in BASES if b != ref])
        af    = min(float(np.random.beta(0.5, 100)), 0.001)
        ac    = max(1, int(af * 10_000))
        gene  = random.choice(GENES)
        csq   = random.choice(CONSEQUENCES)
        qual  = random.randint(30, 60)
        info  = f"AF={af:.8f};AC={ac};AN=10000;GENE={gene};CSQ={csq}"
        lines.append(f"{chrom}\t{pos}\trs_synth_{i:06d}\t{ref}\t{alt}\t{qual}\tPASS\t{info}")
    path = os.path.join(outdir, 'test_variants.vcf')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  ✓  VCF:                  {path}  ({n_variants} variants)")

def generate_clinvar_labels(n_labeled, outdir):
    records = []
    for i in range(n_labeled):
        use_vcf_id = random.random() < 0.40
        vid = f"rs_synth_{random.randint(0,499):06d}" if use_vcf_id else f"rs_clinvar_{i:06d}"
        is_path = random.random() < 0.30
        clinsig = random.choice(['Pathogenic','Likely_pathogenic'] if is_path else ['Benign','Likely_benign'])
        records.append({
            'variant_id': vid, 'gene': random.choice(GENES),
            'chrom': random.choice(CHROMOSOMES), 'pos': random.randint(1_000_000, 200_000_000),
            'ref': random.choice(BASES), 'alt': random.choice(BASES),
            'ref_aa': random.choice(AA_CODES), 'alt_aa': random.choice(AA_CODES),
            'protein_position': random.randint(1,1500), 'protein_length': random.randint(200,3000),
            'consequence': random.choice(CONSEQUENCES),
            'clinical_significance': clinsig,
            'review_status': 'criteria_provided_single_submitter',
            'condition': 'hereditary_cancer',
            'CADD_PHRED': round(float(np.random.uniform(5,45)), 2),
            'CADD_RAW': round(float(np.random.uniform(-2,8)), 4),
            'SIFT_score': round(float(np.random.uniform(0,1)), 4),
            'SIFT_pred': random.choice(['D','T']),
            'Polyphen2_HDIV_score': round(float(np.random.uniform(0,1)), 4),
            'Polyphen2_HVAR_score': round(float(np.random.uniform(0,1)), 4),
            'REVEL_score': round(float(np.random.uniform(0,1)), 4),
            'M_CAP_score': round(float(np.random.uniform(0,1)), 4),
            'AF': float(min(np.random.beta(0.5,100), 0.001)),
        })
    df = pd.DataFrame(records)
    path = os.path.join(outdir, 'clinvar_variants.tsv')
    df.to_csv(path, sep='\t', index=False)
    n_p = df['clinical_significance'].str.contains('athogenic').sum()
    n_b = df['clinical_significance'].str.contains('enign').sum()
    print(f"  ✓  ClinVar labels:        {path}  ({n_labeled} entries: {n_p} pathogenic, {n_b} benign)")

def generate_ppi_network(n_edges, outdir):
    all_proteins = GENES + [f'GENE_{i:04d}' for i in range(200)]
    seen, records = set(), []
    attempts = 0
    while len(records) < n_edges and attempts < n_edges * 3:
        attempts += 1
        p1, p2 = random.choice(all_proteins), random.choice(all_proteins)
        key = (min(p1,p2), max(p1,p2))
        if p1 == p2 or key in seen:
            continue
        seen.add(key)
        records.append({'protein1': p1, 'protein2': p2, 'combined_score': random.randint(150,999)})
    df = pd.DataFrame(records)
    path = os.path.join(outdir, 'string_ppi.tsv')
    df.to_csv(path, sep='\t', index=False)
    print(f"  ✓  PPI network:           {path}  ({len(df)} interactions)")

def generate_conservation_scores(n_positions, outdir):
    df = pd.DataFrame({
        'chrom': random.choices(CHROMOSOMES, k=n_positions),
        'pos':   [random.randint(1_000_000, 200_000_000) for _ in range(n_positions)],
        'phyloP100way_vertebrate':    np.random.normal(0, 2, n_positions),
        'phastCons100way_vertebrate': np.random.beta(2, 5, n_positions),
        'GERP++_RS':                  np.random.normal(3, 2, n_positions).clip(-5, 10),
        'GERP++_NR':                  np.random.normal(5, 3, n_positions).clip(0, 15),
        'SiPhy_29way_logOdds':        np.random.exponential(3, n_positions),
    })
    path = os.path.join(outdir, 'conservation_scores.tsv')
    df.to_csv(path, sep='\t', index=False)
    print(f"  ✓  Conservation scores:   {path}  ({n_positions} positions)")

def generate_gene_annotations(outdir):
    all_genes = GENES + [f'GENE_{i:04d}' for i in range(200)]
    records = [{
        'gene': g, 'pLI': float(np.random.beta(2,2)), 'LOEUF': float(np.random.uniform(0,2)),
        'mis_z': float(np.random.normal(0,1)), 'syn_z': float(np.random.normal(0,1)),
        'n_exons': random.randint(2,80), 'cds_length': random.randint(300,15000),
        'gtex_blood': float(max(0, np.random.normal(5,3))),
        'gtex_brain': float(max(0, np.random.normal(4,3))),
        'gtex_liver': float(max(0, np.random.normal(6,4))),
        'omim_disease': int(random.random() < 0.30),
        'essential_gene': int(random.random() < 0.15),
    } for g in all_genes]
    df = pd.DataFrame(records)
    path = os.path.join(outdir, 'gene_annotations.tsv')
    df.to_csv(path, sep='\t', index=False)
    print(f"  ✓  Gene annotations:      {path}  ({len(df)} genes)")

def parse_args():
    p = argparse.ArgumentParser(description='Generate synthetic test data for rare-variant-gnn.')
    p.add_argument('--outdir',      default='data/example', help='Output directory [data/example]')
    p.add_argument('--n-variants',  type=int, default=500,   help='VCF variants [500]')
    p.add_argument('--n-labeled',   type=int, default=300,   help='ClinVar entries [300]')
    p.add_argument('--n-edges',     type=int, default=5000,  help='PPI edges [5000]')
    p.add_argument('--n-positions', type=int, default=10000, help='Conservation positions [10000]')
    p.add_argument('--seed',        type=int, default=42,    help='Random seed [42]')
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"\nGenerating synthetic test data → {args.outdir}/\n")
    generate_vcf(args.n_variants, args.outdir)
    generate_clinvar_labels(args.n_labeled, args.outdir)
    generate_ppi_network(args.n_edges, args.outdir)
    generate_conservation_scores(args.n_positions, args.outdir)
    generate_gene_annotations(args.outdir)
    print(f"\n✓  All files ready. Run the pipeline with:")
    print(f"   nextflow run main.nf \\")
    print(f"     --input_vcf      {args.outdir}/test_variants.vcf \\")
    print(f"     --known_variants {args.outdir}/clinvar_variants.tsv \\")
    print(f"     --ppi_network    {args.outdir}/string_ppi.tsv \\")
    print(f"     --gene_annot     {args.outdir}/gene_annotations.tsv \\")
    print(f"     --conservation   {args.outdir}/conservation_scores.tsv \\")
    print(f"     --outdir results/\n")

if __name__ == '__main__':
    main()
