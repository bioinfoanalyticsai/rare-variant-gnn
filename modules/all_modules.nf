/*
========================================================================================
    Module: FILTER_VARIANTS — Keep rare variants below AF threshold
========================================================================================
*/
process FILTER_VARIANTS {
    tag "AF<${params.af_threshold}"
    label 'process_low'
    publishDir "${params.outdir}/preprocessing/filtered", mode: 'copy', enabled: params.publish_all

    input:
    path variants_tsv

    output:
    path "filtered_variants.tsv", emit: filtered_variants
    path "filter_stats.txt",      emit: filter_stats

    script:
    """
    filter_variants.py \\
        --input    ${variants_tsv} \\
        --af-max   ${params.af_threshold} \\
        --output   filtered_variants.tsv \\
        --stats    filter_stats.txt
    """
}

/*
========================================================================================
    Module: ANNOTATE_VARIANTS — VEP functional annotation
========================================================================================
*/
process ANNOTATE_VARIANTS {
    tag "VEP annotation"
    label 'process_high'
    publishDir "${params.outdir}/preprocessing/annotated", mode: 'copy'

    input:
    path filtered_variants

    output:
    path "annotated_variants.tsv", emit: annotated_variants
    path "vep_summary.html",       emit: vep_summary

    script:
    """
    annotate_variants.py \\
        --input  ${filtered_variants} \\
        --output annotated_variants.tsv \\
        --genome ${params.genome_build} \\
        --summary vep_summary.html
    """
}

/*
========================================================================================
    Module: MERGE_LABELS — Merge with ClinVar/HGMD known labels
========================================================================================
*/
process MERGE_LABELS {
    tag "merge labels"
    label 'process_low'
    publishDir "${params.outdir}/preprocessing/labeled", mode: 'copy'

    input:
    path annotated_variants
    path known_variants

    output:
    path "merged_variants.tsv", emit: merged_variants
    path "label_stats.txt",     emit: label_stats

    script:
    """
    merge_labels.py \\
        --variants ${annotated_variants} \\
        --labels   ${known_variants} \\
        --output   merged_variants.tsv \\
        --stats    label_stats.txt
    """
}

/*
========================================================================================
    Module: SPLIT_DATASET — Train/Val/Test splits with stratification
========================================================================================
*/
process SPLIT_DATASET {
    tag "split ${params.train_split}/${params.val_split}/${params.test_split}"
    label 'process_low'
    publishDir "${params.outdir}/preprocessing/splits", mode: 'copy'

    input:
    path merged_variants

    output:
    path "split_labels.tsv",  emit: split_labels
    path "split_stats.txt",   emit: split_stats

    script:
    """
    split_dataset.py \\
        --input      ${merged_variants} \\
        --train      ${params.train_split} \\
        --val        ${params.val_split} \\
        --test       ${params.test_split} \\
        --seed       ${params.seed} \\
        --output     split_labels.tsv \\
        --stats      split_stats.txt
    """
}

/*
========================================================================================
    Module: EXTRACT_SEQUENCE_FEATURES — CADD, SIFT, PolyPhen, ESM embeddings
========================================================================================
*/
process EXTRACT_SEQUENCE_FEATURES {
    tag "sequence features"
    label 'process_medium'
    publishDir "${params.outdir}/features/sequence", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants

    output:
    path "seq_features.tsv", emit: seq_features

    script:
    """
    extract_sequence_features.py \\
        --input   ${annotated_variants} \\
        --output  seq_features.tsv
    """
}

/*
========================================================================================
    Module: EXTRACT_CONSERVATION — PhyloP, PhastCons, GERP++
========================================================================================
*/
process EXTRACT_CONSERVATION {
    tag "conservation"
    label 'process_medium'
    publishDir "${params.outdir}/features/conservation", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants
    path conservation_scores

    output:
    path "cons_features.tsv", emit: cons_features

    script:
    """
    extract_conservation.py \\
        --variants     ${annotated_variants} \\
        --conservation ${conservation_scores} \\
        --output       cons_features.tsv
    """
}

/*
========================================================================================
    Module: EXTRACT_STRUCTURAL_FEATURES — Domain, PTM, secondary structure
========================================================================================
*/
process EXTRACT_STRUCTURAL_FEATURES {
    tag "structural features"
    label 'process_medium'
    publishDir "${params.outdir}/features/structural", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants

    output:
    path "struct_features.tsv", emit: struct_features

    script:
    """
    extract_structural_features.py \\
        --input  ${annotated_variants} \\
        --output struct_features.tsv
    """
}

/*
========================================================================================
    Module: EXTRACT_GENE_FEATURES — pLI, LOEUF, GTEx expression
========================================================================================
*/
process EXTRACT_GENE_FEATURES {
    tag "gene features"
    label 'process_low'
    publishDir "${params.outdir}/features/gene", mode: 'copy', enabled: params.publish_all

    input:
    path annotated_variants
    path gene_annotations

    output:
    path "gene_features.tsv", emit: gene_features

    script:
    """
    extract_gene_features.py \\
        --variants    ${annotated_variants} \\
        --annotations ${gene_annotations} \\
        --output      gene_features.tsv
    """
}

/*
========================================================================================
    Module: COMBINE_FEATURES — Merge all feature sets into unified matrix
========================================================================================
*/
process COMBINE_FEATURES {
    tag "combine features"
    label 'process_low'
    publishDir "${params.outdir}/features/combined", mode: 'copy'

    input:
    path seq_features
    path cons_features
    path struct_features
    path gene_features

    output:
    path "node_features.tsv",  emit: node_features
    path "feature_names.txt",  emit: feature_names

    script:
    """
    combine_features.py \\
        --seq-features    ${seq_features} \\
        --cons-features   ${cons_features} \\
        --struct-features ${struct_features} \\
        --gene-features   ${gene_features} \\
        --output          node_features.tsv \\
        --names           feature_names.txt
    """
}

/*
========================================================================================
    Module: BUILD_PROTEIN_GRAPH — Build PPI graph topology from STRING
========================================================================================
*/
process BUILD_PROTEIN_GRAPH {
    tag "build PPI graph"
    label 'process_medium'
    publishDir "${params.outdir}/graph/topology", mode: 'copy', enabled: params.publish_all

    input:
    path ppi_network

    output:
    path "graph_topology.pkl", emit: graph_topology
    path "graph_stats.txt",    emit: topology_stats

    script:
    """
    build_protein_graph.py \\
        --ppi       ${ppi_network} \\
        --k-hops    ${params.graph_k_hops} \\
        --output    graph_topology.pkl \\
        --stats     graph_stats.txt
    """
}

/*
========================================================================================
    Module: ASSIGN_NODE_FEATURES — Map variant features to graph nodes
========================================================================================
*/
process ASSIGN_NODE_FEATURES {
    tag "assign node features"
    label 'process_medium'

    input:
    path graph_topology
    path node_features

    output:
    path "featured_graph.pkl", emit: featured_graph

    script:
    """
    assign_node_features.py \\
        --graph    ${graph_topology} \\
        --features ${node_features} \\
        --output   featured_graph.pkl
    """
}

/*
========================================================================================
    Module: SERIALIZE_GRAPH — Convert to PyTorch Geometric Data object
========================================================================================
*/
process SERIALIZE_GRAPH {
    tag "serialize to PyG"
    label 'process_medium'
    publishDir "${params.outdir}/graph", mode: 'copy'

    input:
    path featured_graph

    output:
    path "pyg_graph.pt",    emit: pyg_graph
    path "graph_stats.json", emit: graph_stats

    script:
    """
    serialize_graph.py \\
        --input  ${featured_graph} \\
        --output pyg_graph.pt \\
        --stats  graph_stats.json
    """
}

/*
========================================================================================
    Module: TRAIN_GNN — Train Graph Attention Network
========================================================================================
*/
process TRAIN_GNN {
    tag "train GAT model"
    label 'process_gpu'
    publishDir "${params.outdir}/models", mode: 'copy'

    input:
    path graph_data
    path split_labels

    output:
    path "best_model.pt",      emit: best_model
    path "training_logs.json", emit: training_logs
    path "checkpoints/",       emit: checkpoints

    script:
    """
    train_gnn.py \\
        --graph       ${graph_data} \\
        --labels      ${split_labels} \\
        --epochs      ${params.gnn_epochs} \\
        --lr          ${params.gnn_lr} \\
        --hidden      ${params.gnn_hidden} \\
        --layers      ${params.gnn_layers} \\
        --batch-size  ${params.batch_size} \\
        --seed        ${params.seed} \\
        --model-dir   checkpoints/ \\
        --best-model  best_model.pt \\
        --logs        training_logs.json
    """
}

/*
========================================================================================
    Module: PREDICT_GNN — Run inference with trained model
========================================================================================
*/
process PREDICT_GNN {
    tag "GNN inference"
    label 'process_gpu'
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
    predict_gnn.py \\
        --graph        ${graph_data} \\
        --model        ${model_checkpoint} \\
        --predictions  predictions.tsv \\
        --probs        probabilities.tsv \\
        --scores       variant_scores.tsv
    """
}

/*
========================================================================================
    Module: EVALUATE_MODEL — Compute AUC, precision, recall, F1, plots
========================================================================================
*/
process EVALUATE_MODEL {
    tag "evaluation"
    label 'process_low'
    publishDir "${params.outdir}/evaluation", mode: 'copy'

    input:
    path predictions
    path split_labels

    output:
    path "metrics.json",    emit: metrics
    path "plots/",          emit: plots
    path "eval_report.html", emit: report

    script:
    """
    evaluate_model.py \\
        --predictions ${predictions} \\
        --labels      ${split_labels} \\
        --metrics     metrics.json \\
        --plots-dir   plots/ \\
        --report      eval_report.html
    """
}

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
