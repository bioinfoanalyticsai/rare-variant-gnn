/*
========================================================================================
    Subworkflow: GRAPH_CONSTRUCTION
    - Build protein-protein interaction graph from STRING network
    - Assign node features to graph nodes (variants mapped to genes)
    - Add k-hop neighborhood edges
    - Serialize graph to PyTorch Geometric format (.pt)
========================================================================================
*/

include { BUILD_PROTEIN_GRAPH  } from '../modules/build_protein_graph.nf'
include { ASSIGN_NODE_FEATURES } from '../modules/assign_node_features.nf'
include { SERIALIZE_GRAPH      } from '../modules/serialize_graph.nf'

workflow GRAPH_CONSTRUCTION {
    take:
    ch_node_features  // Channel: combined node feature matrix
    ch_ppi            // Channel: PPI network TSV

    main:
    // Build the base graph topology from PPI
    BUILD_PROTEIN_GRAPH(ch_ppi)

    // Assign variant-level features to graph nodes
    ASSIGN_NODE_FEATURES(
        BUILD_PROTEIN_GRAPH.out.graph_topology,
        ch_node_features
    )

    // Serialize to PyG-compatible format
    SERIALIZE_GRAPH(ASSIGN_NODE_FEATURES.out.featured_graph)

    emit:
    graph_data     = SERIALIZE_GRAPH.out.pyg_graph
    graph_stats    = SERIALIZE_GRAPH.out.graph_stats
}
