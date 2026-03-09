process EXTRACT_STRUCTURAL_FEATURES {
    tag "structural features"
    publishDir "${params.outdir}/features/structural", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants

    output:
    path "struct_features.tsv", emit: struct_features

    script:
    """
    python ${projectDir}/bin/extract_structural_features.py \
        --input  "${annotated_variants}" \
        --output struct_features.tsv
    """
}
