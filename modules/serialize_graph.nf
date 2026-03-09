process SERIALIZE_GRAPH {
    tag "serialize to PyG"
    publishDir "${params.outdir}/graph", mode: 'copy'

    input:
    path featured_graph

    output:
    path "pyg_graph.pt",     emit: pyg_graph
    path "graph_stats.json", emit: graph_stats

    script:
    """
    python ${projectDir}/bin/serialize_graph.py \
        --input  "${featured_graph}" \
        --output pyg_graph.pt \
        --stats  graph_stats.json
    """
}
