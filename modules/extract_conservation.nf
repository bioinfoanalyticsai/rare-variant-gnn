process EXTRACT_CONSERVATION {
    tag "conservation"
    publishDir "${params.outdir}/features/conservation", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants
    path conservation_scores

    output:
    path "cons_features.tsv", emit: cons_features

    script:
    """
    python ${projectDir}/bin/extract_conservation.py \
        --variants     "${annotated_variants}" \
        --conservation "${conservation_scores}" \
        --output       cons_features.tsv
    """
}
