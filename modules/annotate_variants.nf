process ANNOTATE_VARIANTS {
    tag "VEP annotation"
    publishDir "${params.outdir}/preprocessing/annotated", mode: 'copy'

    input:
    path filtered_variants

    output:
    path "annotated_variants.tsv", emit: annotated_variants
    path "vep_summary.txt",        emit: vep_summary

    script:
    def build = params.genome_build ?: 'GRCh38'
    """
    python ${projectDir}/bin/annotate_variants.py \
        --input  "${filtered_variants}" \
        --output annotated_variants.tsv \
        --genome "${build}" \
        --summary vep_summary.txt
    """
}
