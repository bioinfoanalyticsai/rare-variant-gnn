process ASSIGN_NODE_FEATURES {
    tag "assign node features"
    publishDir "${params.outdir}/graph", mode: 'copy', enabled: params.publish_all

    input:
    path graph_topology
    path node_features

    output:
    path "featured_graph.pkl", emit: featured_graph

    script:
    """
    python ${projectDir}/bin/assign_node_features.py \
        --graph    "${graph_topology}" \
        --features "${node_features}" \
        --output   featured_graph.pkl
    """
}
