#!/usr/bin/env nextflow

/*
========================================================================================
    Rare Variant Pathogenicity Prediction using Graph Neural Networks
========================================================================================
    Author      : GNN-RareVariant Pipeline
    Description : End-to-end pipeline for predicting pathogenicity of rare genetic
                  variants using Graph Neural Networks with protein interaction,
                  conservation, and functional annotation features.
========================================================================================
*/

nextflow.enable.dsl = 2

// ─── Import Subworkflows ───────────────────────────────────────────────────────
include { PREPROCESSING     } from './workflows/preprocessing.nf'
include { FEATURE_EXTRACTION } from './workflows/feature_extraction.nf'
include { GRAPH_CONSTRUCTION } from './workflows/graph_construction.nf'
include { GNN_TRAINING       } from './workflows/gnn_training.nf'
include { GNN_PREDICTION     } from './workflows/gnn_prediction.nf'
include { EVALUATION         } from './workflows/evaluation.nf'

// ─── Parameter Defaults ───────────────────────────────────────────────────────
params.help            = false
params.mode            = 'full'          // full | train | predict | evaluate
params.input_vcf       = null
params.input_variants  = null
params.known_variants  = "${projectDir}/data/example/clinvar_variants.tsv"
params.ppi_network     = "${projectDir}/data/example/string_ppi.tsv"
params.gene_annot      = "${projectDir}/data/example/gene_annotations.tsv"
params.conservation    = "${projectDir}/data/example/conservation_scores.tsv"
params.outdir          = "${projectDir}/results"
params.model_dir       = "${params.outdir}/models"
params.pretrained_model = null
params.genome_build    = 'GRCh38'
params.af_threshold    = 0.001          // MAF threshold for "rare"
params.graph_k_hops    = 2             // k-hop neighborhood
params.gnn_epochs      = 100
params.gnn_lr          = 0.001
params.gnn_hidden      = 128
params.gnn_layers      = 3
params.batch_size      = 64
params.train_split     = 0.7
params.val_split       = 0.15
params.test_split      = 0.15
params.seed            = 42
params.skip_training   = false
params.publish_all     = false

// ─── Help Message ─────────────────────────────────────────────────────────────
def helpMessage() {
    log.info """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║          Rare Variant Pathogenicity Prediction — GNN Pipeline           ║
    ╚══════════════════════════════════════════════════════════════════════════╝

    USAGE:
        nextflow run main.nf [OPTIONS]

    REQUIRED:
        --input_vcf         Path to input VCF file (gzipped or plain)
        --known_variants    TSV file with labeled variants (ClinVar/HGMD)

    OPTIONAL:
        --input_variants    Pre-processed variant TSV (skip VCF parsing)
        --ppi_network       STRING PPI network TSV (gene1, gene2, score)
        --gene_annot        Gene annotation file (gene features)
        --conservation      Conservation scores TSV per variant
        --outdir            Output directory [${params.outdir}]
        --model_dir         Model checkpoint directory
        --pretrained_model  Use pretrained model (skip training)
        --genome_build      Reference genome [${params.genome_build}]
        --af_threshold      Allele frequency threshold [${params.af_threshold}]
        --graph_k_hops      GNN neighborhood hops [${params.graph_k_hops}]
        --gnn_epochs        Training epochs [${params.gnn_epochs}]
        --gnn_lr            Learning rate [${params.gnn_lr}]
        --gnn_hidden        Hidden layer size [${params.gnn_hidden}]
        --gnn_layers        Number of GNN layers [${params.gnn_layers}]
        --batch_size        Batch size [${params.batch_size}]
        --train_split       Train fraction [${params.train_split}]
        --val_split         Validation fraction [${params.val_split}]
        --test_split        Test fraction [${params.test_split}]
        --seed              Random seed [${params.seed}]
        --mode              Pipeline mode: full|train|predict|evaluate
        --skip_training     Skip training, load pretrained model

    EXAMPLES:
        # Full pipeline with VCF input
        nextflow run main.nf \\
            --input_vcf variants.vcf.gz \\
            --known_variants clinvar.tsv \\
            --outdir results/

        # Prediction only with pretrained model
        nextflow run main.nf \\
            --mode predict \\
            --input_vcf new_variants.vcf.gz \\
            --pretrained_model results/models/best_model.pt
    """
}

if (params.help) {
    helpMessage()
    exit 0
}

// ─── Validate Inputs ──────────────────────────────────────────────────────────
def validateParams() {
    if (!params.input_vcf && !params.input_variants) {
        error "ERROR: Provide --input_vcf or --input_variants"
    }
    if (params.train_split + params.val_split + params.test_split != 1.0) {
        error "ERROR: train_split + val_split + test_split must equal 1.0"
    }
}

// ─── Print Pipeline Info ──────────────────────────────────────────────────────
log.info """
╔══════════════════════════════════════════════════════════════════════════════╗
║          Rare Variant Pathogenicity Prediction — GNN Pipeline               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Mode           : ${params.mode}
║  Genome Build   : ${params.genome_build}
║  AF Threshold   : ${params.af_threshold}
║  GNN Layers     : ${params.gnn_layers}  Hidden: ${params.gnn_hidden}
║  Epochs         : ${params.gnn_epochs}  LR: ${params.gnn_lr}
║  Output Dir     : ${params.outdir}
╚══════════════════════════════════════════════════════════════════════════════╝
"""

// ─── Workflow ─────────────────────────────────────────────────────────────────
workflow {
    validateParams()

    // Input channels
    ch_vcf = params.input_vcf
        ? Channel.fromPath(params.input_vcf, checkIfExists: true)
        : Channel.empty()

    ch_variants_pre = params.input_variants
        ? Channel.fromPath(params.input_variants, checkIfExists: true)
        : Channel.empty()

    ch_known     = Channel.fromPath(params.known_variants, checkIfExists: true)
    ch_ppi       = Channel.fromPath(params.ppi_network,    checkIfExists: true)
    ch_gene_annot = Channel.fromPath(params.gene_annot,    checkIfExists: true)
    ch_conservation = Channel.fromPath(params.conservation, checkIfExists: true)

    // ── Step 1: Preprocessing ──────────────────────────────────────────────
    PREPROCESSING(ch_vcf, ch_variants_pre, ch_known)

    // ── Step 2: Feature Extraction ────────────────────────────────────────
    FEATURE_EXTRACTION(
        PREPROCESSING.out.variants,
        ch_gene_annot,
        ch_conservation
    )

    // ── Step 3: Graph Construction ────────────────────────────────────────
    GRAPH_CONSTRUCTION(
        FEATURE_EXTRACTION.out.node_features,
        ch_ppi
    )

    // ── Step 4: GNN Training (if not predict-only) ────────────────────────
    if (params.mode in ['full', 'train'] && !params.skip_training) {
        GNN_TRAINING(
            GRAPH_CONSTRUCTION.out.graph_data,
            PREPROCESSING.out.labels
        )
        ch_model = GNN_TRAINING.out.best_model
    } else {
        ch_model = params.pretrained_model
            ? Channel.fromPath(params.pretrained_model, checkIfExists: true)
            : Channel.empty()
    }

    // ── Step 5: Prediction ─────────────────────────────────────────────────
    GNN_PREDICTION(
        GRAPH_CONSTRUCTION.out.graph_data,
        ch_model
    )

    // ── Step 6: Evaluation ─────────────────────────────────────────────────
    if (params.mode != 'predict') {
        EVALUATION(
            GNN_PREDICTION.out.predictions,
            PREPROCESSING.out.labels
        )
    }
}

// ─── Completion Handler ───────────────────────────────────────────────────────
workflow.onComplete {
    log.info """
    ════════════════════════════════════════════════════
    Pipeline completed!
    Status    : ${workflow.success ? 'SUCCESS ✓' : 'FAILED ✗'}
    Duration  : ${workflow.duration}
    Output    : ${params.outdir}
    ════════════════════════════════════════════════════
    """
}

workflow.onError {
    log.error "Pipeline failed: ${workflow.errorMessage}"
}
