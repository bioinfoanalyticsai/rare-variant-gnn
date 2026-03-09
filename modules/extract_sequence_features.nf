process EXTRACT_SEQUENCE_FEATURES {
    tag "sequence features"
    publishDir "${params.outdir}/features/sequence", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants

    output:
    path "seq_features.tsv", emit: seq_features

    script:
    """
    python ${projectDir}/bin/extract_sequence_features.py \
        --input  "${annotated_variants}" \
        --output seq_features.tsv
    """
}
