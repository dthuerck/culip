/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIB_DATA_STRUCTURES_GRAPH_H_
#define __CULIP_LIB_DATA_STRUCTURES_GRAPH_H_

#include <vector>
#include <memory>
#include <utility>
#include <tuple>
#include <random>

#include <libs/utils/defines.h>
#include <libs/utils/params.h>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

template<typename T>
using edge = std::tuple<index_t, index_t, T>;

/**
 * Models an weighted, undirected graph. Edge doublettes are automatically
 * summed up.
 * Stores data as adjacency lists, best used for (very) sparse graphs.
 */
template<typename T>
class Graph
{
public:
    Graph(const params_ptr<T>& params, const size_t num_nodes);
    ~Graph();

    /**
     * Returns the number of nodes created in this graph.
     */
    size_t get_num_nodes();

    /**
     * Get read access to the set of edges.
     */
    const std::vector<edge<T>>& get_edges();

    /**
     * Return adjaceny list (to nodes) with edge weights for node n.
     */
    const std::vector<std::pair<index_t, T>> operator[](const index_t n);

    /**
     * Add edge between two - already available - nodes.
     * If there is already such an edge, add 1 to its weight.
     */
    void add_edge(const index_t n1, const index_t n2, const T weight = 1.0);

    /**
     * Fill distance (ignoring weights) markers given a start node by
     * a queue-based BFS.
     */
    void mark_bfs(const index_t start_node, std::vector<size_t>& markers);

    /**
     * Insert 0-edges to connect graph.
     */
    void connect();

    /**
     * Find longest (shortest) path between a pair of nodes and return that path.
     */
    size_t longest_shortest_path(index_t& n1, index_t& n2);

protected:
    size_t m_num_nodes;
    std::vector<std::vector<index_t>> m_adj_lists;
    std::vector<edge<T>> m_edges;

    params_ptr<T> m_params;
};

template<typename T>
using Graph_ptr = std::shared_ptr<Graph<T>>;

NS_DATA_STRUCTURES_END
NS_CULIP_END

#endif /* __CULIP_LIBS_DATA_STRUCTURES_GRAPH_H_ */
