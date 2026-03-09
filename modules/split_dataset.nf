process SPLIT_DATASET {
    tag "split"
    publishDir "${params.outdir}/preprocessing/splits", mode: 'copy'

    input:
    path merged_variants

    output:
    path "split_labels.tsv", emit: split_labels
    path "split_stats.txt",  emit: split_stats

    script:
    """
    python ${projectDir}/bin/split_dataset.py \
        --input  "${merged_variants}" \
        --train  ${params.train_split} \
        --val    ${params.val_split} \
        --test   ${params.test_split} \
        --seed   ${params.seed} \
        --output split_labels.tsv \
        --stats  split_stats.txt
    """
}
