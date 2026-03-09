process TRAIN_GNN {
    tag "train GAT"
    publishDir "${params.outdir}/models", mode: 'copy'

    input:
    path graph_data
    path split_labels

    output:
    path "best_model.pt",      emit: best_model
    path "training_logs.json", emit: training_logs
    path "checkpoints/",       emit: checkpoints, optional: true

    script:
    """
    mkdir -p checkpoints
    python ${projectDir}/bin/train_gnn.py \
        --graph      "${graph_data}" \
        --labels     "${split_labels}" \
        --epochs     ${params.gnn_epochs} \
        --lr         ${params.gnn_lr} \
        --hidden     ${params.gnn_hidden} \
        --layers     ${params.gnn_layers} \
        --batch-size ${params.batch_size} \
        --seed       ${params.seed} \
        --model-dir  checkpoints/ \
        --best-model best_model.pt \
        --logs       training_logs.json
    """
}
