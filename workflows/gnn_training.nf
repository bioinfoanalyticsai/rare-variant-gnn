/*
========================================================================================
    Subworkflow: GNN_TRAINING
    - Train Graph Attention Network (GAT) model
    - Hyperparameter tracking
    - Checkpoint saving (best val AUC)
    - Learning curve logging
========================================================================================
*/

include { TRAIN_GNN           } from '../modules/train_gnn.nf'
include { HYPERPARAMETER_TUNE } from '../modules/hyperparameter_tune.nf'

workflow GNN_TRAINING {
    take:
    ch_graph_data  // Channel: serialized PyG graph (.pt)
    ch_labels      // Channel: split labels (train/val/test)

    main:
    TRAIN_GNN(ch_graph_data, ch_labels)

    emit:
    best_model     = TRAIN_GNN.out.best_model
    training_logs  = TRAIN_GNN.out.training_logs
    checkpoints    = TRAIN_GNN.out.checkpoints
}
