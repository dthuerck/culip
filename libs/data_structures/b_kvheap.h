/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_DATA_STRUCTURES_B_KVHEAP_H_
#define __CULIP_LIBS_DATA_STRUCTURES_B_KVHEAP_H_

#include <libs/utils/defines.h>

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

/**
 * Comparators for b_kvheap, producing a min- and max-heap. Use std::comparator
 * as base, i.e. std::less or std::greater.
 */

/**
 * A simple implementation of a binary min-heap with updateable values. Elements
 * are indexed, i.e. every value is an index into an array that contains the
 * element's current position in the storage std::vector.
 *
 * This means: Updating an element does not require querying the whole heap,
 * every element moved in an operation means an additional write [requires
 * a contiguous set of keys, though].
 *
 * Relies on std-stuff as much as possible.
 */
template<typename K, typename V, class OP>
class b_kvheap
{
public:
    b_kvheap();
    b_kvheap(const K * keys, const V * vals, const mat_int_t num_entries);
    ~b_kvheap();

    void push(const K key, const V val);
    void update(const K key, const V new_val);
    void remove(const K key);

    void top(K& key, V& val);
    void pop();
    bool empty();

    V value(const K key);
    mat_int_t size();

    void print();

protected:
    void sift_up(const mat_int_t internal_id);
    void sift_down(const mat_int_t internal_id);

protected:
    /* key <-> internal index lookup */
    std::map<K, mat_int_t> m_fwd_keys;
    std::vector<K> m_rev_keys;

    /* values & positions saved w.r.t internal index */
    std::vector<K> m_positions;
    std::vector<V> m_vals;

    /* actual heap */
    std::vector<mat_int_t> m_data;

    /* keep track of used / free internal IDs */
    std::vector<mat_int_t> m_free_internal_ids;
    mat_int_t m_next_internal_id;

    /* copy of the comparison operator */
    OP m_comp;
};

/* prepare some heap times */
template<typename K, typename V>
using bmin_kvheap = b_kvheap<K, V, std::less<V>>;

template<typename K, typename V>
using bmax_kvheap = b_kvheap<K, V, std::greater<V>>;

NS_DATA_STRUCTURES_END
NS_CULIP_END

#endif /* __CULIP_LIBS_DATA_STRUCTURES_B_KVHEAP_H_ */
