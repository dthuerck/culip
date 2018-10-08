/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/data_structures/graph.h>

#include <queue>
#include <iostream>
#include <climits>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

/**
 * *****************************************************************************
 *                             Graph<T> - public
 * *****************************************************************************
 */

template<typename T>
Graph<T>::
Graph(
    const params_ptr<T>& params,
    const size_t num_nodes)
: m_params(params),
  m_num_nodes(num_nodes),
  m_adj_lists(num_nodes),
  m_edges()
{

}

/* ************************************************************************** */

template<typename T>
Graph<T>::
~Graph()
{

}

/* ************************************************************************** */

template<typename T>
size_t
Graph<T>::
get_num_nodes()
{
    return m_num_nodes;
}

/* ************************************************************************** */

template<typename T>
const
std::vector<edge<T>>&
Graph<T>::
get_edges()
{
    return m_edges;
}

/* ************************************************************************** */

template<typename T>
const std::vector<std::pair<index_t, T>>
Graph<T>::
operator[](
    const index_t n)
{
    std::vector<std::pair<index_t, T>> adjacent;

    if (n >= m_num_nodes)
        return adjacent;

    for (index_t i = 0; i < m_adj_lists[n].size(); ++i)
    {
        const edge<T>& e = m_edges[m_adj_lists[n][i]];
        adjacent.push_back(std::make_pair(std::get<0>(e) == n ?
            std::get<1>(e) : std::get<0>(e), std::get<2>(e)));
    }

    return adjacent;
}

/* ************************************************************************** */

template<typename T>
void
Graph<T>::
add_edge(
    const index_t n1,
    const index_t n2,
    const T weight)
{
    if (n1 >= m_num_nodes || n2 >= m_num_nodes || n1 == n2)
        return;

    /* check if there is already such an edge */
    for (const std::pair<index_t, T>& i : (*this)[n1])
    {
        if (i.first == n2)
        {
            /* find this edge and increase weight by 1 */
            if (m_params->p_edge_mode != culip_graph_edge_mode_t::UNIT)
            {
                for (const index_t& e_id : m_adj_lists[n1])
                {
                    if (std::get<0>(m_edges[e_id]) == n2 ||
                        std::get<1>(m_edges[e_id]) == n2)
                        std::get<2>(m_edges[e_id]) += weight;
                }
            }
            return;
        }
    }

    m_edges.push_back(edge<T>(n1, n2, weight));
    m_adj_lists[n1].push_back(m_edges.size() - 1);
    m_adj_lists[n2].push_back(m_edges.size() - 1);
}

/* ************************************************************************** */

template<typename T>
void
Graph<T>::
mark_bfs(
    const index_t start_node,
    std::vector<size_t>& markers)
{
    markers.clear();
    markers.resize(m_num_nodes, -1);

    std::queue<int> qu;
    qu.push(start_node);

    markers[start_node] = 0;
    while(!qu.empty())
    {
        const int cur = qu.front();
        qu.pop();

        for (const std::pair<int, T>& e : (*this)[cur])
        {
            if (markers[e.first] < 0)
            {
                markers[e.first] = markers[cur] + 1;
                qu.push(e.first);
            }
        }
    }
}

/* ************************************************************************** */

template<typename T>
void
Graph<T>::
connect()
{
    /**
     * Execute BFS from random nodes, each node that is
     * not traversed is then augmented by a virtual
     * edge.
     * Repeat until only one connected component left.
     */
    std::cout << "Connecting graph..." << std::endl;

    std::vector<size_t> marker;
    bool connected = false;

    while(!connected)
    {
        connected = true;
        const index_t start_node = rand() % m_num_nodes;
        mark_bfs(start_node, marker);

        for (index_t i = 0; i < m_num_nodes; ++i)
        {
            if (marker[i] < 0)
            {
                connected = false;
                for (index_t j = i + 1; j < m_num_nodes; ++j)
                {
                    if (marker[j] >= 0)
                    {
                        add_edge(i, j, 0);
                    }
                }
            }
        }
    }
}

/* ************************************************************************** */

template<typename T>
size_t
Graph<T>::
longest_shortest_path(
    index_t& n1,
    index_t& n2)
{
    /**
     * Determine longest (shortest) path by executing a BFS from each node
     * as start node, then selecting the longest path in the
     * resulting (implicit) matrix.
     */
    std::cout << "Finding all-pairs shorted paths..." << std::endl;
    size_t longest_path = 0;

    #pragma omp parallel for shared(longest_path)
    for (index_t nd1 = 0; nd1 < m_num_nodes; ++nd1)
    {
        std::vector<size_t> markers(m_num_nodes, 0);

        mark_bfs(nd1, markers);

        size_t local_max_path = 0;
        index_t local_max_id = 0;
        for (index_t nd2 = nd1 + 1; nd2 < m_num_nodes; ++nd2)
        {
            if (markers[nd2] > local_max_path)
            {
                local_max_path = markers[nd2];
                local_max_id = nd2;
            }
        }

        #pragma omp critical
        {
            if (local_max_path > longest_path)
            {
                longest_path = local_max_path;
                n1 = nd1;
                n2 = local_max_id;
            }
        }
    }

    return longest_path;
}

NS_DATA_STRUCTURES_END
NS_CULIP_END
