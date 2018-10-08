/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_DATA_STRUCTURES_HGRAPH_H_
#define __CULIP_LIBS_DATA_STRUCTURES_HGRAPH_H_

#include <vector>
#include <memory>
#include <utility>

#include <libs/utils/defines.h>
#include <libs/utils/params.h>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

/**
 * Model both hyperedges and nodes as incidence list of the respective 
 * other + (optional) weights
 */

using inc_list = std::vector<index_t>;
template<typename T>
using h_edge = std::pair<inc_list, T>;
template<typename T>
using h_node = std::pair<inc_list, T>;

/**
 * Represents a hypergraph with weighted nets and nodes. Each net 
 * is modelled as a list of incident nodes.
 */
template<typename T>
class HGraph
{
public:
    HGraph(const params_ptr<T>& params);
    ~HGraph();
    
    /**
     * Retruns the number of nodes in this hypergraph.
     */
    const size_t num_nodes() const;
    
    /**
     * Returns the number of hyperedges in this graph.
     */
    const size_t num_hedges() const;
    
    /** 
     * Grants access to incidence lists:
     *  node = true -- returns a list of hyperedges containing
     *                 node n,
     *  node = false -- returns a hyperedge as its list of pins
     */
    inc_list& get_inc(const index_t id, const bool node);
    
    /**
     * Returns or sets the weight of a hyperedge or node; syntax similar
     * to get_inc.
     */
    T& weight(const int index_t, const bool node);
    
    /**
     * Add a weighted hyperedge to the graph; new nodes are created
     * automagically.
     */ 
    void add_hyperedge(const inc_list& inc_nodes, const T weight = (T) 1.0);
    
protected:
    /* note: assume i1, i2 to be ordered in the same fashion */
    bool hedge_equal(const inc_list& i1, const inc_list& i2);
    void update_modified();
    
    const index_t hash_hedge(const inc_list& inc);
    void update_hashes();
    void order_lists();
    
protected:
    params_ptr<T> m_params;
    std::vector<h_edge<T>> m_hedges;
    std::vector<h_node<T>> m_nodes;
    
    std::vector<index_t> m_modified;
    std::vector<index_t> m_hashes;
    
    size_t m_must_update;
};

template<typename T>
using HGraph_ptr = std::shared_ptr<HGraph<T>>;

NS_DATA_STRUCTURES_END
NS_CULIP_END

#endif /* __CULIP_LIB_DATA_STRUCTURES_HGRAPH_H_ */
