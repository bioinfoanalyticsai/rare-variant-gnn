process EVALUATE_MODEL {
    tag "evaluation"
    publishDir "${params.outdir}/evaluation", mode: 'copy'

    input:
    path predictions
    path split_labels

    output:
    path "metrics.json",     emit: metrics
    path "plots/",           emit: plots
    path "eval_report.html", emit: report

    script:
    """
    mkdir -p plots
    python ${projectDir}/bin/evaluate_model.py \
        --predictions "${predictions}" \
        --labels      "${split_labels}" \
        --metrics     metrics.json \
        --plots-dir   plots/ \
        --report      eval_report.html
    """
}
