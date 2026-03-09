/*
========================================================================================
    Subworkflow: FEATURE_EXTRACTION
    - Sequence-based features (CADD, SIFT, PolyPhen)
    - Conservation scores (PhyloP, PhastCons, GERP)
    - Structural features (domain, PTM, secondary structure)
    - Gene-level features (pLI, LOEUF, expression)
    - Combine all features into node feature matrix
========================================================================================
*/

include { EXTRACT_SEQUENCE_FEATURES    } from '../modules/extract_sequence_features.nf'
include { EXTRACT_CONSERVATION         } from '../modules/extract_conservation.nf'
include { EXTRACT_STRUCTURAL_FEATURES  } from '../modules/extract_structural_features.nf'
include { EXTRACT_GENE_FEATURES        } from '../modules/extract_gene_features.nf'
include { COMBINE_FEATURES             } from '../modules/combine_features.nf'

workflow FEATURE_EXTRACTION {
    take:
    ch_variants    // Channel: annotated variants TSV
    ch_gene_annot  // Channel: gene annotations
    ch_conservation // Channel: conservation scores

    main:
    // Parallel feature extraction
    EXTRACT_SEQUENCE_FEATURES(ch_variants)
    EXTRACT_CONSERVATION(ch_variants, ch_conservation)
    EXTRACT_STRUCTURAL_FEATURES(ch_variants)
    EXTRACT_GENE_FEATURES(ch_variants, ch_gene_annot)

    // Combine all feature vectors
    COMBINE_FEATURES(
        EXTRACT_SEQUENCE_FEATURES.out.seq_features,
        EXTRACT_CONSERVATION.out.cons_features,
        EXTRACT_STRUCTURAL_FEATURES.out.struct_features,
        EXTRACT_GENE_FEATURES.out.gene_features
    )

    emit:
    node_features   = COMBINE_FEATURES.out.node_features
    feature_names   = COMBINE_FEATURES.out.feature_names
}
