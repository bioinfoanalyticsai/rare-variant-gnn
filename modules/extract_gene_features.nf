process EXTRACT_GENE_FEATURES {
    tag "gene features"
    publishDir "${params.outdir}/features/gene", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants
    path gene_annotations

    output:
    path "gene_features.tsv", emit: gene_features

    script:
    """
    python ${projectDir}/bin/extract_gene_features.py \
        --variants    "${annotated_variants}" \
        --annotations "${gene_annotations}" \
        --output      gene_features.tsv
    """
}
