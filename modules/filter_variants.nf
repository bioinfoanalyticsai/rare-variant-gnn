process FILTER_VARIANTS {
    tag "AF<${params.af_threshold}"
    publishDir "${params.outdir}/preprocessing/filtered", mode: 'copy', enabled: params.publish_all

    input:
    path variants_tsv

    output:
    path "filtered_variants.tsv", emit: filtered_variants
    path "filter_stats.txt",      emit: filter_stats

    script:
    def af = params.af_threshold ?: 0.001
    """
    python ${projectDir}/bin/filter_variants.py \
        --input    "${variants_tsv}" \
        --af-max   ${af} \
        --output   filtered_variants.tsv \
        --stats    filter_stats.txt
    """
}
