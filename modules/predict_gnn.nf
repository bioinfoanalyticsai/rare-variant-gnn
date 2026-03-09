process PREDICT_GNN {
    tag "GNN inference"
    publishDir "${params.outdir}/predictions", mode: 'copy'

    input:
    path graph_data
    path model_checkpoint

    output:
    path "predictions.tsv",    emit: predictions
    path "probabilities.tsv",  emit: probabilities
    path "variant_scores.tsv", emit: variant_scores

    script:
    """
    python ${projectDir}/bin/predict_gnn.py \
        --graph       "${graph_data}" \
        --model       "${model_checkpoint}" \
        --predictions predictions.tsv \
        --probs       probabilities.tsv \
        --scores      variant_scores.tsv
    """
}
