/*
========================================================================================
    Subworkflow: PREPROCESSING
    - Parse VCF → TSV
    - Filter rare variants by allele frequency
    - Merge with known labeled variants (ClinVar/HGMD)
    - VEP annotation
    - Generate train/val/test split labels
========================================================================================
*/

include { PARSE_VCF          } from '../modules/parse_vcf.nf'
include { FILTER_VARIANTS    } from '../modules/filter_variants.nf'
include { ANNOTATE_VARIANTS  } from '../modules/annotate_variants.nf'
include { MERGE_LABELS       } from '../modules/merge_labels.nf'
include { SPLIT_DATASET      } from '../modules/split_dataset.nf'

workflow PREPROCESSING {
    take:
    ch_vcf          // Channel: path to input VCF (may be empty)
    ch_variants_pre // Channel: pre-processed TSV (may be empty)
    ch_known        // Channel: labeled variants TSV

    main:
    // Parse VCF if provided, otherwise use pre-processed TSV
    if (params.input_vcf) {
        PARSE_VCF(ch_vcf)
        ch_raw = PARSE_VCF.out.variants_tsv
    } else {
        ch_raw = ch_variants_pre
    }

    // Filter to rare variants only
    FILTER_VARIANTS(ch_raw)

    // VEP annotation (functional consequences)
    ANNOTATE_VARIANTS(FILTER_VARIANTS.out.filtered_variants)

    // Merge with known pathogenicity labels
    MERGE_LABELS(
        ANNOTATE_VARIANTS.out.annotated_variants,
        ch_known
    )

    // Generate reproducible dataset splits
    SPLIT_DATASET(MERGE_LABELS.out.merged_variants)

    emit:
    variants = ANNOTATE_VARIANTS.out.annotated_variants
    labels   = SPLIT_DATASET.out.split_labels
}
