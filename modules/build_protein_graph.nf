process BUILD_PROTEIN_GRAPH {
    tag "build PPI graph"
    publishDir "${params.outdir}/graph/topology", mode: 'copy', enabled: params.publish_all

    input:
    path ppi_network

    output:
    path "graph_topology.pkl", emit: graph_topology
    path "graph_stats.txt",    emit: topology_stats

    script:
    """
    python ${projectDir}/bin/build_protein_graph.py \
        --ppi    "${ppi_network}" \
        --k-hops ${params.graph_k_hops} \
        --output graph_topology.pkl \
        --stats  graph_stats.txt
    """
}
