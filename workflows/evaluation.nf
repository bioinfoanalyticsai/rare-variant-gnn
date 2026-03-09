/*
========================================================================================
    Subworkflow: EVALUATION
========================================================================================
*/
include { EVALUATE_MODEL } from '../modules/evaluate_model.nf'

workflow EVALUATION {
    take:
    ch_predictions  // Channel: model predictions TSV
    ch_labels       // Channel: true labels TSV

    main:
    EVALUATE_MODEL(ch_predictions, ch_labels)

    emit:
    metrics       = EVALUATE_MODEL.out.metrics
    plots         = EVALUATE_MODEL.out.plots
    report        = EVALUATE_MODEL.out.report
}
