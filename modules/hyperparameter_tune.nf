/*
    Nextflow Module: HYPERPARAMETER_TUNE
*/

/*
========================================================================================
    Module: HYPERPARAMETER_TUNE — Optuna-based HPO (optional)
========================================================================================
*/

process HYPERPARAMETER_TUNE {
    tag "HPO search"
    label 'process_gpu'
    publishDir "${params.outdir}/hpo", mode: 'copy'

    input:
    path graph_data
    path split_labels

    output:
    path "best_hparams.json", emit: best_hparams
    path "hpo_study.pkl",     emit: hpo_study

    script:
    """
    hyperparameter_tune.py \\
        --graph   ${graph_data} \\
        --labels  ${split_labels} \\
        --trials  50 \\
        --output  best_hparams.json \\
        --study   hpo_study.pkl
    """
}
