/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIB_UTILS_MEM_POOL_H_
#define __CULIP_LIB_UTILS_MEM_POOL_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>
#include <libs/data_structures/b_kvheap.h>

#include <vector>
#include <list>
#include <map>

using namespace NS_CULIP::NS_DATA_STRUCTURES;

NS_CULIP_BEGIN
NS_UTILS_BEGIN

/**
 * *****************************************************************************
 * ******************************** MEM STRUCTS ********************************
 * *****************************************************************************
 */
struct MemElem
{
    bool in_use;

    mat_size_t offset;
    mat_size_t aligned_offset;
    mat_size_t size;

    MemElem * prev;
    MemElem * next;
};

class MemSlice
{
public:
    MemSlice(const mat_size_t size, const char * ptr,
        const mat_size_t split_threshold, const mat_size_t max_elems);
    MemSlice(MemSlice&& other);
    ~MemSlice();

    char * request_mem(const mat_size_t size, const mat_size_t align);
    bool release_mem(const char * ptr);

    mat_size_t max_size_avail();
    mat_size_t size_in_use();
    mat_size_t size_free();

    const char * raw_ptr();
    void invalidate();

    void print();

protected:
    /* slice memory */
    const mat_size_t m_size;
    const char * m_ptr;

    /* list of elements */
    mat_size_t m_num_elems;
    MemElem * m_first;

    /* keep track of largest available memory */
    mat_size_t m_max_size_avail;

    /* parameters */
    const mat_size_t m_split_threshold;
    const mat_size_t m_max_elems;

    /* mark invalidated slice (does not delete members) */
    bool m_dnd;
};

/**
 * *****************************************************************************
 * ********************************* _MEMPOOL **********************************
 * *****************************************************************************
 */

/* "Hidden class", only used through MemPool */
template<bool on_device>
class _MemPool
{
public:
    _MemPool(const mat_int_t dev_id, const mat_size_t split_threshold,
        const mat_size_t max_elems_per_slice);
    ~_MemPool();

    /* manually allocate slices */
    void allocate_slice(const mat_size_t size);

    /* memory management */
    char * request_mem(const mat_size_t size, const mat_size_t align);
    void release_mem(const char * ptr);

    /* information */
    mat_size_t mem_unassigned();
    mat_size_t mem_inuse();

    void print();

    /* frees all slices */
    void clean();

protected:
    char * _alloc(const mat_size_t size);
    void _free(const char * ptr);

protected:
    mat_int_t m_dev_id;

    /* slice management */
    std::vector<MemSlice> m_slices;
    bmax_kvheap<mat_int_t, mat_size_t> m_slice_heap;

    /* save ptr -> slice association */
    std::map<const char *, mat_int_t> m_ptr_in_slice;

    /* count memory unassigned/in use */
    mat_size_t m_mem_unassigned;
    mat_size_t m_mem_inuse;

    /* parameters */
    const mat_size_t m_split_threshold;
    const mat_size_t m_max_elems_per_slice;
};

/**
 * *****************************************************************************
 * ********************************** MEMPOOL **********************************
 * *****************************************************************************
 */

/**
 * MemPool: Designed to be a singleton instance for both CPU and GPU,
 *          reusing previously allocated memory.
 */
class MemPool
{
public:
    MemPool();
    ~MemPool();

    _MemPool<true>& GPU();
    _MemPool<false>& CPU();

    char * request_memory(const bool on_device, const mat_size_t size,
        const mat_size_t align = 16);
    void release_memory(const bool on_device, const char * ptr);

    void provision(const bool on_device, const mat_size_t size);
    void cleanup();

private:
    _MemPool<true> m_pool_gpu;
    _MemPool<false> m_pool_cpu;
};

NS_UTILS_END

/* static instance */
NS_UTILS::MemPool& GlobalMemPool();

NS_CULIP_END

#endif /* __CULIP_LIBS_UTILS_MEM_POOL_H_ */

