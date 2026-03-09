process COMBINE_FEATURES {
    tag "combine features"
    publishDir "${params.outdir}/features/combined", mode: 'copy'

    input:
    path seq_features
    path cons_features
    path struct_features
    path gene_features

    output:
    path "node_features.tsv", emit: node_features
    path "feature_names.txt", emit: feature_names

    script:
    """
    python ${projectDir}/bin/combine_features.py \
        --seq-features    "${seq_features}" \
        --cons-features   "${cons_features}" \
        --struct-features "${struct_features}" \
        --gene-features   "${gene_features}" \
        --output          node_features.tsv \
        --names           feature_names.txt
    """
}
