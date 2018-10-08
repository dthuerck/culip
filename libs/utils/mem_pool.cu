/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/utils/mem_pool.cuh>
#include <libs/utils/mem_pool.impl.cuh>

NS_CULIP_BEGIN
NS_UTILS_BEGIN

/**
 * *****************************************************************************
 * ******************************** MEM STRUCTS ********************************
 * *****************************************************************************
 */

MemSlice::
MemSlice(
    const mat_size_t size,
    const char * ptr,
    const mat_size_t split_threshold,
    const mat_size_t max_elems)
: m_size(size),
  m_ptr(ptr),
  m_num_elems(1),
  m_max_size_avail(size),
  m_split_threshold(split_threshold),
  m_max_elems(max_elems),
  m_dnd(false)
{
    /* create default element */
    m_first = new MemElem{.in_use = false, .offset = 0, .aligned_offset = 0,
        .size = size, .prev = nullptr, .next = nullptr};
}

/* ************************************************************************** */

MemSlice::
MemSlice(
    MemSlice&& other)
: m_size(other.m_size),
  m_ptr(other.m_ptr),
  m_num_elems(other.m_num_elems),
  m_first(other.m_first),
  m_max_size_avail(other.m_max_size_avail),
  m_split_threshold(other.m_split_threshold),
  m_max_elems(other.m_max_elems),
  m_dnd(other.m_dnd)
{
    /* tell other object to not delete its ptr */
    other.invalidate();
}

/* ************************************************************************** */

MemSlice::
~MemSlice()
{
    if(m_dnd)
        return;

    /* delete all elements - memory is deleted externally by pool */
    MemElem * ptr = m_first;
    while(ptr != nullptr)
    {
        MemElem * old_ptr = ptr;

        ptr = ptr->next;
        delete old_ptr;
    }
}

/* ************************************************************************** */

char *
MemSlice::
request_mem(
    const mat_size_t size,
    const mat_size_t align)
{
    const char * mem = nullptr;
    mat_size_t next_max_size = 0;

    const mat_size_t aligned_size = size + align - 1;

    /* cancel if requested size is more than the largest element offers */
    if(aligned_size > m_max_size_avail)
        return (char *) mem;

    /* look for first-fit free element */
    MemElem * ptr = m_first;
    while(ptr != nullptr)
    {
        /* skip used elements */
        if(ptr->in_use)
        {
            ptr = ptr->next;
            continue;
        }

        /* skip elements that are too small */
        if(ptr->size < aligned_size || mem != nullptr)
        {
            /* also count if a suitable element was found */
            next_max_size = std::max(next_max_size, ptr->size);

            ptr = ptr->next;
            continue;
        }

        /* use block */
        if(mem == nullptr)
        {
            /* split element if permitted */
            if(m_num_elems < m_max_elems &&
                (ptr->size - aligned_size) >= m_split_threshold)
            {
                MemElem * new_elem = new MemElem{.in_use = false, .offset =
                    ptr->offset + aligned_size, .aligned_offset = ptr->offset +
                    aligned_size, .size = ptr->size - aligned_size,
                    .prev = ptr, .next = ptr->next};

                /* update prev pointer in element to the right of the new */
                if(ptr->next != nullptr)
                    ptr->next->prev = new_elem;

                /* reduce size of chosen element and update pointer */
                ptr->size = aligned_size;
                ptr->next = new_elem;

                /* update slice */
                ++m_num_elems;
            }

            /* mark element as used and save return pointer */
            ptr->in_use = true;
            mem = m_ptr + ptr->offset;

            /* align pointer and record that */
            ptr->aligned_offset = ptr->offset +
                (align - ((uintptr_t) mem % align));
            mem += (align - ((uintptr_t) mem % align));

            ptr = ptr->next;
        }
    }

    m_max_size_avail = next_max_size;
    return (char *) mem;
}

/* ************************************************************************** */

bool
MemSlice::
release_mem(
    const char * ptr)
{
    /* check is we are in the correct slice */
    if(ptr < m_ptr || ptr >= m_ptr + m_size)
        return false;

    /* use find the element in question */
    const mat_size_t offset = ptr - m_ptr;

    MemElem * ind = m_first;
    while(ind != nullptr)
    {
        if(ind->aligned_offset == offset)
        {
            /* mark elem as not in use any more */
            ind->in_use = false;

            /* fuse with all free elements to right */
            MemElem * right = ind->next;
            mat_size_t add_size_right = 0;
            while(right != nullptr && !right->in_use)
            {
                add_size_right += right->size;
                --m_num_elems;

                MemElem * hold = right->next;
                delete right;
                right = hold;
            }
            ind->next = right;
            if(right != nullptr)
                right->prev = ind;
            ind->size += add_size_right;

            /* fuse with all free elements on the left */
            MemElem * left = ind;
            mat_size_t add_size_left = 0;
            while(left->prev != nullptr && !left->prev->in_use)
            {
                MemElem * hold = left->prev;
                add_size_left += left->size;
                if(left != ind)
                {
                    delete left;
                    --m_num_elems;
                }
                left = hold;
            }

            if(left != ind)
            {
                left->size += add_size_left;
                left->next = ind->next;

                if(ind->next != nullptr)
                    ind->next->prev = left;

                delete ind;
                --m_num_elems;
            }

            /* set aligned pointer = normal pointer */
            left->aligned_offset = left->offset;

            return true;
        }

        ind = ind->next;
    }

    return false;
}

/* ************************************************************************** */

mat_size_t
MemSlice::
max_size_avail()
{
    return m_max_size_avail;
}

/* ************************************************************************** */

mat_size_t
MemSlice::
size_in_use()
{
    mat_size_t in_use = 0;

    MemElem * ptr = m_first;
    while(ptr != nullptr)
    {
        in_use += (ptr->in_use ? ptr->size : 0);
        ptr = ptr->next;
    }

    return in_use;
}

/* ************************************************************************** */

mat_size_t
MemSlice::
size_free()
{
    mat_size_t not_in_use = 0;

    MemElem * ptr = m_first;
    while(ptr != nullptr)
    {
        not_in_use += (!ptr->in_use ? ptr->size : 0);
        ptr = ptr->next;
    }

    return not_in_use;
}

/* ************************************************************************** */

const char *
MemSlice::
raw_ptr()
{
    return m_ptr;
}

/* ************************************************************************** */

void
MemSlice::
invalidate()
{
    m_dnd = true;
}

/* ************************************************************************** */

void
MemSlice::
print()
{
    printf("Memory slice at %p has %ld elements, uses %ld bytes, " \
        "has %ld bytes free and the largest free element is %ld bytes:\n",
        m_ptr, m_num_elems, size_in_use(), size_free(), max_size_avail());
    MemElem * ptr = m_first;
    for(mat_size_t i = 0; i < m_num_elems; ++i)
    {
        printf("\tElem %ld: %ld (%ld) / %ld / %ld (ptr: %p)\n", i, ptr->offset,
            ptr->aligned_offset, ptr->size, ptr->in_use,
            m_ptr + ptr->aligned_offset);
        if(ptr->prev != nullptr)
        {
            printf("\t\tLeft: %p\n", m_ptr + ptr->prev->aligned_offset);
        }
        if(ptr->next != nullptr)
        {
            printf("\t\tRight: %p\n", m_ptr + ptr->next->aligned_offset);
        }
        ptr = ptr->next;
    }
}

/**
 * *****************************************************************************
 * ********************************* _MEMPOOL **********************************
 * *****************************************************************************
 */

/**
 * device-specific alloc/free functions
 */
template<>
char *
_MemPool<true>::
_alloc(
    const mat_size_t size)
{
    char * ptr;
    CHECK_CUDA(cudaMalloc((void **) &ptr, size));

    return ptr;
}

template<>
char *
_MemPool<false>::
_alloc(
    const mat_size_t size)
{
    return (new char[size]);
}

/* ************************************************************************** */

template<>
void
_MemPool<true>::
_free(
    const char * ptr)
{
    CHECK_CUDA(cudaFree((void *) ptr));
}

template<>
void
_MemPool<false>::
_free(
    const char * ptr)
{
    delete[] ptr;
}

/* ************************************************************************** */

template class _MemPool<true>;
template class _MemPool<false>;

/**
 * *****************************************************************************
 * ********************************** MEMPOOL **********************************
 * *****************************************************************************
 */

MemPool::
MemPool()
: m_pool_gpu(0, 1024, 100),
  m_pool_cpu(0, 1024, 100)
{

}

/* ************************************************************************** */

MemPool::
~MemPool()
{

}

/* ************************************************************************** */

_MemPool<true>&
MemPool::
GPU()
{
    return m_pool_gpu;
}

/* ************************************************************************** */

_MemPool<false>&
MemPool::
CPU()
{
    return m_pool_cpu;
}

/* ************************************************************************** */

char *
MemPool::
request_memory(
    const bool on_device,
    const mat_size_t size,
    const mat_size_t align)
{
    if(on_device)
    {
        return GPU().request_mem(size, align);
    }
    else
    {
        return CPU().request_mem(size, align);
    }
}

/* ************************************************************************** */

void
MemPool::
release_memory(
    const bool on_device,
    const char * ptr)
{
    if(on_device)
    {
        return GPU().release_mem(ptr);
    }
    else
    {
        return CPU().release_mem(ptr);
    }
}

/* ************************************************************************** */

void
MemPool::
provision(
    const bool on_device,
    const mat_size_t size)
{
    if(on_device)
    {
        GPU().allocate_slice(size);
    }
    else
    {
        CPU().allocate_slice(size);
    }
}

/* ************************************************************************** */

void
MemPool::
cleanup()
{
    CPU().clean();
    GPU().clean();
}

NS_UTILS_END

/* ************************************************************************** */

NS_UTILS::MemPool& GlobalMemPool()
{
   static NS_UTILS::MemPool instance;

   return instance;
}

NS_CULIP_END
