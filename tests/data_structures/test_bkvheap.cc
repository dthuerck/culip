/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <iostream>
#include <limits>
#include <algorithm>
#include <numeric>

#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>

#include <libs/utils/defines.h>
#include <libs/data_structures/b_kvheap.h>

NS_CULIP_BEGIN
NS_TEST_BEGIN

using namespace NS_DATA_STRUCTURES;

template<typename T>
class BKVHeapTest : public ::testing::Test
{
public:
    const mat_int_t m_capacity = 10;
    using K = typename T::K;
    using V = typename T::V;

public:
    BKVHeapTest()
    {

    }

    ~BKVHeapTest()
    {

    }

    void
    SetUp()
    {
        m_min_heap = bmin_kvheap<K, V>();
        m_max_heap = bmax_kvheap<K, V>();
    }

    void
    TearDown()
    {

    }

protected:

    /* units under test */
    bmin_kvheap<K, V> m_min_heap;
    bmax_kvheap<K, V> m_max_heap;
};
TYPED_TEST_CASE_P(BKVHeapTest);

TYPED_TEST_P(BKVHeapTest, TestOrderedInsert)
{
    typedef typename TypeParam::K K;
    typedef typename TypeParam::V V;

    /* insert numbers ordered, with key + 0.5 = value */
    for(K i = 0; i < this->m_capacity; ++i)
    {
        this->m_min_heap.push(i, i + 0.5);
        this->m_max_heap.push(this->m_capacity - 1 - i,
            this->m_capacity - 1 - i + 0.5);
    }

    /* extract numbers ordered */
    V last_min = -std::numeric_limits<V>::max();
    V last_max = std::numeric_limits<V>::max();

    K pop_min_k, pop_max_k;
    V pop_min_v, pop_max_v;
    for(K i = 0; i < this->m_capacity; ++i)
    {
        ASSERT_FALSE(this->m_min_heap.empty());
        ASSERT_FALSE(this->m_max_heap.empty());

        this->m_min_heap.top(pop_min_k, pop_min_v);
        this->m_max_heap.top(pop_max_k, pop_max_v);

        /* min-heap: each popped value is larger than its predecessor */
        ASSERT_GT(pop_min_v, last_min);

        /* min-heap: each popped value is smaller than its predecessor */
        ASSERT_LT(pop_max_v, last_max);

        last_min = pop_min_v;
        last_max = pop_max_v;

        this->m_min_heap.pop();
        this->m_max_heap.pop();
    }

    ASSERT_TRUE(this->m_min_heap.empty());
    ASSERT_TRUE(this->m_max_heap.empty());
}

TYPED_TEST_P(BKVHeapTest, TestUnorderedInsert)
{
    typedef typename TypeParam::K K;
    typedef typename TypeParam::V V;

    std::vector<K> key_order(this->m_capacity);
    std::iota(key_order.begin(), key_order.end(), 0);
    std::random_shuffle(key_order.begin(), key_order.end());

    /* insert numbers ordered, with key + 0.5 = value */
    for(K i = 0; i < this->m_capacity; ++i)
    {
        this->m_min_heap.push(key_order[i], key_order[i] + 0.5);
        this->m_max_heap.push(this->m_capacity - 1 - key_order[i],
            this->m_capacity - 1 - key_order[i] + 0.5);
    }

    /* extract numbers ordered */
    V last_min = -std::numeric_limits<V>::max();
    V last_max = std::numeric_limits<V>::max();

    K pop_min_k, pop_max_k;
    V pop_min_v, pop_max_v;
    for(K i = 0; i < this->m_capacity; ++i)
    {
        ASSERT_FALSE(this->m_min_heap.empty());
        ASSERT_FALSE(this->m_max_heap.empty());

        this->m_min_heap.top(pop_min_k, pop_min_v);
        this->m_max_heap.top(pop_max_k, pop_max_v);

        /* min-heap: each popped value is larger than its predecessor */
        ASSERT_GT(pop_min_v, last_min);

        /* min-heap: each popped value is smaller than its predecessor */
        ASSERT_LT(pop_max_v, last_max);

        last_min = pop_min_v;
        last_max = pop_max_v;

        this->m_min_heap.pop();
        this->m_max_heap.pop();
    }

    ASSERT_TRUE(this->m_min_heap.empty());
    ASSERT_TRUE(this->m_max_heap.empty());
}

TYPED_TEST_P(BKVHeapTest, TestUpdate)
{
    typedef typename TypeParam::K K;
    typedef typename TypeParam::V V;

    /* insert numbers ordered, with key + 0.5 = value */
    for(K i = 0; i < this->m_capacity; ++i)
    {
        this->m_min_heap.push(i, i + 0.5);
        this->m_max_heap.push(this->m_capacity - 1 - i,
            this->m_capacity - 1 - i + 0.5);
    }

    /* invert all keys and expect inverse order (min <-> max) */
    for(K i = 0; i < this->m_capacity; ++i)
    {
        this->m_min_heap.update(i, -(i + 0.5));
        this->m_max_heap.update(this->m_capacity - 1 - i,
            -(this->m_capacity - 1 - i + 0.5));
    }

    /* extract numbers ordered */
    V last_min = -std::numeric_limits<V>::max();
    V last_max = std::numeric_limits<V>::max();

    K pop_min_k, pop_max_k;
    V pop_min_v, pop_max_v;
    for(K i = 0; i < this->m_capacity; ++i)
    {
        ASSERT_FALSE(this->m_min_heap.empty());
        ASSERT_FALSE(this->m_max_heap.empty());

        this->m_min_heap.top(pop_min_k, pop_min_v);
        this->m_max_heap.top(pop_max_k, pop_max_v);

        /* min-heap: each popped value is larger than its predecessor */
        ASSERT_LT(-pop_min_v, -last_min);

        /* min-heap: each popped value is smaller than its predecessor */
        ASSERT_GT(-pop_max_v, -last_max);

        last_min = pop_min_v;
        last_max = pop_max_v;

        this->m_min_heap.pop();
        this->m_max_heap.pop();
    }

    ASSERT_TRUE(this->m_min_heap.empty());
    ASSERT_TRUE(this->m_max_heap.empty());
}

TYPED_TEST_P(BKVHeapTest, TestRemove)
{
    typedef typename TypeParam::K K;
    typedef typename TypeParam::V V;

    std::vector<K> key_order(this->m_capacity);
    std::iota(key_order.begin(), key_order.end(), 0);
    std::random_shuffle(key_order.begin(), key_order.end());

    /* insert numbers ordered, with key + 0.5 = value */
    for(K i = 0; i < this->m_capacity; ++i)
    {
        this->m_min_heap.push(key_order[i], key_order[i] + 0.5);
        this->m_max_heap.push(this->m_capacity - 1 - key_order[i],
            this->m_capacity - 1 - key_order[i] + 0.5);
    }

    /* delete every second key */
    std::vector<bool> key_exists(this->m_capacity, true);
    for(K i = 0; i < this->m_capacity; i += 2)
    {
        this->m_min_heap.remove(key_order[i]);
        this->m_max_heap.remove(key_order[i]);

        key_exists[key_order[i]] = false;
    }

    /* extract numbers ordered */
    V last_min = -std::numeric_limits<V>::max();
    V last_max = std::numeric_limits<V>::max();

    K pop_min_k, pop_max_k;
    V pop_min_v, pop_max_v;
    for(K i = 0; i < this->m_capacity; i += 2)
    {
        ASSERT_FALSE(this->m_min_heap.empty());
        ASSERT_FALSE(this->m_max_heap.empty());

        this->m_min_heap.top(pop_min_k, pop_min_v);
        this->m_max_heap.top(pop_max_k, pop_max_v);

        /* min-heap: each popped value is larger than its predecessor */
        ASSERT_GT(pop_min_v, last_min);
        ASSERT_TRUE(key_exists[pop_min_k]);

        /* min-heap: each popped value is smaller than its predecessor */
        ASSERT_LT(pop_max_v, last_max);
        ASSERT_TRUE(key_exists[pop_max_k]);

        last_min = pop_min_v;
        last_max = pop_max_v;

        this->m_min_heap.pop();
        this->m_max_heap.pop();
    }

    ASSERT_TRUE(this->m_min_heap.empty());
    ASSERT_TRUE(this->m_max_heap.empty());
}

#define TEST_CASE(name, key_t, val_t) \
    struct name { \
        typedef key_t K; \
        typedef val_t V; \
    };

TEST_CASE(heap_int_float, mat_int_t, float);
TEST_CASE(heap_int_double, mat_int_t, double);

REGISTER_TYPED_TEST_CASE_P(BKVHeapTest,
    TestOrderedInsert,
    TestUnorderedInsert,
    TestUpdate,
    TestRemove);
typedef ::testing::Types<
    heap_int_float,
    heap_int_double
    > TestTupleInstances;
INSTANTIATE_TYPED_TEST_CASE_P(HeapTest,
    BKVHeapTest, TestTupleInstances);

NS_TEST_END
NS_CULIP_END
