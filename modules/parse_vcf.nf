/*
========================================================================================
    Module: PARSE_VCF
    Parse VCF file into a structured TSV.
    Resources defined in nextflow.config withName:'PARSE_VCF' block.
========================================================================================
*/

process PARSE_VCF {
    tag "$vcf"
    publishDir "${params.outdir}/preprocessing/vcf_parsed", mode: 'copy', enabled: params.publish_all

    input:
    path vcf

    output:
    path "parsed_variants.tsv", emit: variants_tsv
    path "vcf_stats.txt",       emit: vcf_stats

    script:
    def build = params.genome_build ?: 'GRCh38'
    """
    python ${projectDir}/bin/parse_vcf.py \
        --vcf "${vcf}" \
        --genome-build "${build}" \
        --output parsed_variants.tsv \
        --stats  vcf_stats.txt
    """
}
