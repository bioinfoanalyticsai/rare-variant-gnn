/*
========================================================================================
    Subworkflow: GNN_PREDICTION
========================================================================================
*/
include { PREDICT_GNN } from '../modules/predict_gnn.nf'

workflow GNN_PREDICTION {
    take:
    ch_graph_data  // Channel: serialized PyG graph (.pt)
    ch_model       // Channel: trained model checkpoint

    main:
    PREDICT_GNN(ch_graph_data, ch_model)

    emit:
    predictions    = PREDICT_GNN.out.predictions
    probabilities  = PREDICT_GNN.out.probabilities
    variant_scores = PREDICT_GNN.out.variant_scores
}
