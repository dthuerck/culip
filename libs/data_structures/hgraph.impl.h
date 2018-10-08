/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/data_structures/hgraph.h>

#include <algorithm>
#include <assert.h>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

/**
 * *****************************************************************************
 *                            HGraph<T> - public
 * *****************************************************************************
 */

template<typename T>
HGraph<T>::
HGraph(
    const params_ptr<T>& params)
    : m_params(params),
      m_hedges(),
      m_nodes(),
      m_modified(),
      m_hashes(),
      m_must_update((size_t) 0)
{
    
}

template<typename T>
HGraph<T>::
~HGraph()
{
    
}

template<typename T>
const size_t 
HGraph<T>::
num_nodes() 
const
{
    return m_nodes.size();
}

template<typename T>
const size_t 
HGraph<T>::
num_hedges() 
const
{
    return m_hedges.size();
}

template<typename T>
inc_list& 
HGraph<T>::
get_inc(
    const index_t id, 
    const bool node)
{    
    if (node)
    {
        assert(id < m_nodes.size());
        return m_nodes[id].first;
    }
    else
    {
        assert(id < m_hedges.size());
        m_modified[id] = 1;
        
        // user could potentially execute unexpected changes to hypergraph, so
        // need to mark graph as changed before next operation that
        // assumes anything about the data structures
        m_must_update = 1; 
        return m_hedges[id].first;  
    }
}

template<typename T>
T& 
HGraph<T>::
weight(
    const index_t id, 
    const bool node)
{    
    if (node)
    {
        assert(id < m_nodes.size());
        return m_nodes[id].second;
    }
    else
    {
        assert(id < m_hedges.size());
        return m_hedges[id].second;
    }
}

template<typename T>
void 
HGraph<T>::
add_hyperedge(
    const inc_list& inc_nodes, 
    const T weight)
{   
    if (m_must_update > 0)
        update_modified();

    inc_list ordered_inc_nodes = inc_nodes;
    std::sort(ordered_inc_nodes.begin(), ordered_inc_nodes.end());
    
    /* try to find similar hyperedges (avoid doublettes) */
    const index_t my_hash = hash_hedge(inc_nodes);
    for (index_t i = 0; i < m_hedges.size(); ++i)
    {
        if (my_hash == m_hashes[i] && hedge_equal(ordered_inc_nodes, 
            m_hedges[i].first))
        {
            m_hedges[i].second += weight;
            return;
        }
    }
    
    /* add new nodes as necessary, until highest node ID covered */
    const index_t max_id = ordered_inc_nodes.back();
    if (max_id >= m_nodes.size())
        m_nodes.resize(max_id + 1, h_node<T>(inc_list(), 1.0f));
    
    const index_t new_id = m_hedges.size();
    m_hedges.push_back(std::make_pair(ordered_inc_nodes, weight));
    m_modified.push_back(0);
    m_hashes.push_back(my_hash);
    
    /* add hyperedge id to all incident's node lists (and sort their lists) */
    for (const index_t& n : ordered_inc_nodes)
    {
        m_nodes[n].first.push_back(new_id);
        std::sort(m_nodes[n].first.begin(), m_nodes[n].first.end());
    }
    
    /**
     * no need to update the graph here - only the new edge is affected
     * and these changes are controlled
     */
}


/**
 * *****************************************************************************
 * ************************* HGraph<T> - protected *****************************
 * *****************************************************************************
 */

template<typename T>
bool 
HGraph<T>::
hedge_equal(
    const inc_list& i1, 
    const inc_list& i2)
{
    if (i1.size() != i2.size())
        return false;
    
    for (index_t i = 0; i < i1.size(); ++i)
        if (i1[i] != i2[i])
            return false;
    
    return true;
}

template<typename T>
void
HGraph<T>::
update_modified()
{
    update_hashes();
    order_lists();
    
    std::fill(m_modified.begin(), m_modified.end(), 0);
    m_must_update = 0;
}

template<typename T> 
const index_t
HGraph<T>::
hash_hedge(
    const inc_list& inc)
{
    index_t hash = 0;
    for (const index_t& i : inc)
        hash += i;
    
    return hash;
}

template<typename T>
void 
HGraph<T>::
update_hashes()
{
    #pragma omp parallel for
    for (index_t i = 0; i < m_hedges.size(); ++i)
        if (m_modified[i])
            m_hashes[i] = hash_hedge(m_hedges[i].first);
}

template<typename T>
void
HGraph<T>::
order_lists()
{
    #pragma omp parallel for
    for (index_t i = 0; i < m_nodes.size(); ++i)
        if (m_modified[i])
            std::sort(m_nodes[i].first.begin(), m_nodes[i].first.end());
    
    #pragma omp parallel for
    for (index_t i = 0; i < m_hedges.size(); ++i)
        if (m_modified[i])
            std::sort(m_hedges[i].first.begin(), m_hedges[i].first.end());
}

NS_DATA_STRUCTURES_END
NS_CULIP_END
