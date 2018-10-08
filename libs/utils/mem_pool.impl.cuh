/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/utils/mem_pool.cuh>

#include <algorithm>

NS_CULIP_BEGIN
NS_UTILS_BEGIN

template<bool on_device>
_MemPool<on_device>::
_MemPool(
    const mat_int_t dev_id,
    const mat_size_t split_threshold,
    const mat_size_t max_elems_per_slice)
: m_dev_id(dev_id),
  m_slices(),
  m_slice_heap(),
  m_ptr_in_slice(),
  m_mem_unassigned(0),
  m_mem_inuse(0),
  m_split_threshold(split_threshold),
  m_max_elems_per_slice(max_elems_per_slice)
{

}

/* ************************************************************************** */

template<bool on_device>
_MemPool<on_device>::
~_MemPool()
{
    clean();
}

/* ************************************************************************** */

template<bool on_device>
void
_MemPool<on_device>::
allocate_slice(
    const mat_size_t size)
{
    /* generate memory slice */
    char * ptr = _alloc(size);
    m_slices.emplace_back(MemSlice(size, ptr, m_split_threshold,
        m_max_elems_per_slice));

    /* put into the heap */
    m_slice_heap.push(m_slices.size() - 1, size);

    /* count as unassigned memory */
    m_mem_unassigned += size;
}

/* ************************************************************************** */

template<bool on_device>
char *
_MemPool<on_device>::
request_mem(
    const mat_size_t size,
    const mat_size_t align)
{
    const mat_size_t aligned_size = size + align - 1;

    // /* check if the memory can come from the largest slice */
    // mat_size_t largest_size = 0;
    // mat_int_t use_slice_id = -1;

    // if(!m_slice_heap.empty())
    //     m_slice_heap.top(use_slice_id, largest_size);

    // if(largest_size < aligned_size)
    // {
    //     /* create a new slice */
    //     allocate_slice(4 * aligned_size);

    //     /* update selection */
    //     m_slice_heap.top(use_slice_id, largest_size);
    // }

    // /* allocate memory from slice */
    // char * ptr = m_slices[use_slice_id].request_mem(size, align);

    // /* update slice in heap */
    // m_slice_heap.update(use_slice_id,
    //     m_slices[use_slice_id].max_size_avail());

    // /* udpate stats */
    // m_mem_unassigned -= size;
    // m_mem_inuse += size;

    // /* mark pointer as used */
    // m_ptr_in_slice[ptr] = use_slice_id;

    // /* return memory */
    // return ptr;

    return _alloc(size);
}

/* ************************************************************************** */

template<bool on_device>
void
_MemPool<on_device>::
release_mem(
    const char * ptr)
{
    // /* find slice to process */
    // const mat_int_t slice_id = m_ptr_in_slice[ptr];

    // /* delegate releasing the memory to this slice */
    // m_slices[slice_id].release_mem(ptr);

    _free(ptr);
}

/* ************************************************************************** */

template<bool on_device>
std::size_t
_MemPool<on_device>::
mem_unassigned()
{
    return m_mem_unassigned;
}

/* ************************************************************************** */

template<bool on_device>
std::size_t
_MemPool<on_device>::
mem_inuse()
{
    return m_mem_inuse;
}

/* ************************************************************************** */

template<bool on_device>
void
_MemPool<on_device>::
print()
{
    printf("MemPool on ");
    if(on_device)
        printf("GPU\n");
    else
        printf("CPU\n");

    printf("> Memory unassigned: %ld\n", m_mem_unassigned);
    printf("> Memory in use: %ld\n", m_mem_inuse);

    printf("> Slices: %ld\n", m_slices.size());

    for(mat_size_t i = 0; i < m_slices.size(); ++i)
    {
        printf("> Slice %ld:\n", i);
        m_slices[i].print();
    }
}

/* ************************************************************************** */

template<bool on_device>
void
_MemPool<on_device>::
clean()
{
    for(MemSlice& m : m_slices)
    {
        _free(m.raw_ptr());
        m.invalidate();
    }

    m_slices.clear();
}

NS_UTILS_END
NS_CULIP_END
