process MERGE_LABELS {
    tag "merge labels"
    publishDir "${params.outdir}/preprocessing/labeled", mode: 'copy'

    input:
    path annotated_variants
    path known_variants

    output:
    path "merged_variants.tsv", emit: merged_variants
    path "label_stats.txt",     emit: label_stats

    script:
    """
    python ${projectDir}/bin/merge_labels.py \
        --variants "${annotated_variants}" \
        --labels   "${known_variants}" \
        --output   merged_variants.tsv \
        --stats    label_stats.txt
    """
}
