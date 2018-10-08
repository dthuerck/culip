/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIB_DATA_STRUCTURES_B_IXHEAP_H_
#define __CULIP_LIB_DATA_STRUCTURES_B_IXHEAP_H_

#include <libs/utils/defines.h>

#include <iostream>
#include <algorithm>
#include <vector>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

/**
 * Comparators for b_ixheap, producing a min- and max-heap. Use std::comparator
 * as base, i.e. std::less or std::greater.
 */

/**
 * A simple implementation of a binary min-heap with updateable keys. Elements
 * are indexed, i.e. every value is an index into an array that contains the
 * element's current position in the storage std::vector.
 *
 * This means: Updating an element does not require querying the whole heap,
 * every element moved in an operation means an additional write.
 *
 * Relies on std-stuff as much as possible.
 */
template<typename IXT, class OP>
class b_ixheap
{
public:
    b_ixheap();
    b_ixheap(std::vector<IXT> * positions);
    b_ixheap(std::vector<IXT> * positions, const IXT * keys,
        const IXT * vals, const IXT num_insert);
    ~b_ixheap();

    IXT push(const IXT key, const IXT val);
    IXT update(const IXT val, const IXT pos, const IXT delta);
    bool remove(const IXT val, const IXT pos);
    bool top(IXT& key, IXT& val);
    IXT pop();
    bool empty();
    IXT size();

    void print();

protected:
    bool sift_up(const IXT pos);
    bool sift_down(const IXT pos);

protected:
    std::vector<IXT> * m_positions;
    std::vector<IXT> m_keys;
    std::vector<IXT> m_vals;
    IXT m_size;

    IXT m_err;
    OP m_comp;
};

/* prepare some heap times */
template<typename IXT>
using bmin_ixheap = b_ixheap<IXT, std::less<IXT>>;

template<typename IXT>
using bmax_ixheap = b_ixheap<IXT, std::greater<IXT>>;

NS_DATA_STRUCTURES_END
NS_CULIP_END

#endif /* __CULIP_UTIL_DATA_STRUCTURES_B_IXHEAP_H_ */
