/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/data_structures/b_ixheap.h>

#include <limits>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

/**
 * *****************************************************************************
 * ******************************** B_IXHEAP ***********************************
 * *****************************************************************************
 */

#define _LCHILD(i) ((i << 1) + 1)
#define _RCHILD(i) ((i << 1) + 2)
#define _PARENT(i) ((i - 1) >> 1)

/**
 * *****************************************************************************
 * ********************************* PUBLIC ************************************
 * *****************************************************************************
 */

template<typename IXT, class OP>
b_ixheap<IXT, OP>::
b_ixheap()
{

}

/* ************************************************************************** */

template<typename IXT, class OP>
b_ixheap<IXT, OP>::
b_ixheap(
    std::vector<IXT> * positions)
: m_positions(positions),
  m_keys(10),
  m_vals(10),
  m_size(0),
  m_err(std::numeric_limits<IXT>::max()),
  m_comp()
{
}

/* ************************************************************************** */

template<typename IXT, class OP>
b_ixheap<IXT, OP>::
b_ixheap(
    std::vector<IXT> * positions,
    const IXT * keys,
    const IXT * vals,
    const IXT num_entries)
: b_ixheap(positions)
{
    for(IXT i = 0; i < num_entries; ++i)
        push(keys[i], vals[i]);
}

/* ************************************************************************** */

template<typename IXT, class OP>
b_ixheap<IXT, OP>::
~b_ixheap()
{
}

/* ************************************************************************** */

template<typename IXT, class OP>
IXT
b_ixheap<IXT, OP>::
push(
    const IXT key,
    const IXT val)
{
    /* increase container size */
    if(m_size + 1 > m_keys.capacity())
    {
        m_keys.resize(2 * m_keys.capacity());
        m_vals.resize(2 * m_vals.capacity());
    }

    /* append element to back */
    m_keys[m_size] = key;
    m_vals[m_size] = val;
    (*m_positions)[val] = m_size;

    ++m_size;

    /* sift element up as far as possible */
    sift_up(m_size - 1);

    return (*m_positions)[val];
}

/* ************************************************************************** */

template<typename IXT, class OP>
IXT
b_ixheap<IXT, OP>::
update(
    const IXT val,
    const IXT pos,
    const IXT delta)
{
    /* check if an update is necessary */
    if(delta == 0)
        return m_err;

    /* check is the position really contains the value we asked for */
    if(pos >= m_size)
        return m_err;

    IXT real_pos = pos;
    if(m_vals[real_pos] != val)
    {
        /* if not: find the damn thing! */
        bool found = false;
        for(IXT i = 0; found && i < m_size; ++i)
        {
            if(m_vals[i] == val)
            {
                real_pos = i;
                found = true;
            }
        }

        if(!found)
            return m_err;
    }

    /* update key */
    const IXT old_key = m_keys[real_pos];
    m_keys[real_pos] += delta;

    /* restore heap property - delta == 0 is caught above */
    if(m_comp(delta, 0))
        sift_up(real_pos);
    else
        sift_down(real_pos);

    return (*m_positions)[val];
}

/* ************************************************************************** */

template<typename IXT, class OP>
bool
b_ixheap<IXT, OP>::
remove(
    const IXT val,
    const IXT pos)
{
    /* check is the position really contains the value we asked for */
    if(pos >= m_size)
        return false;

    IXT real_pos = pos;
    if(m_vals[real_pos] != val)
    {
        /* if not: find the damn thing! */
        bool found = false;
        for(IXT i = 0; found && i < m_size; ++i)
        {
            if(m_vals[i] == val)
            {
                real_pos = i;
                found = true;
            }
        }

        if(!found)
            return m_err;
    }

    /* invalidate deleted element */
    (*m_positions)[m_vals[real_pos]] = 0;

    /* pull in last element */
    IXT delta = m_keys[m_size - 1] - m_keys[real_pos];

    m_keys[real_pos] = m_keys[m_size - 1];
    m_vals[real_pos] = m_vals[m_size - 1];
    (*m_positions)[m_vals[real_pos]] = real_pos;

    /* reduce size */
    --m_size;

    /* restore heap property */
    if(delta != 0)
    {
        if(m_comp(delta, 0))
            sift_up(real_pos);
        else
            sift_down(real_pos);
    }

    return true;
}

/* ************************************************************************** */

template<typename IXT, class OP>
bool
b_ixheap<IXT, OP>::
top(
    IXT& key,
    IXT& val)
{
    if(empty())
        return false;

    key = m_keys[0];
    val = m_vals[0];

    return true;
}

/* ************************************************************************** */

template<typename IXT, class OP>
IXT
b_ixheap<IXT, OP>::
pop()
{
    if(empty())
        return m_err;

    /* set last element to root (maintain almost complete tree) */
    m_keys[0] = m_keys[m_size - 1];
    m_vals[0] = m_vals[m_size - 1];
    (*m_positions)[m_vals[m_size - 1]] = 0;

    /* reduce size */
    --m_size;

    /* now sift node down - restore heap property */
    sift_down(0);

    return m_size;
}

/* ************************************************************************** */

template<typename IXT, class OP>
bool
b_ixheap<IXT, OP>::
empty()
{
    return (m_size == 0);
}

/* ************************************************************************** */

template<typename IXT, class OP>
IXT
b_ixheap<IXT, OP>::
size()
{
    return m_size;
}

/**
 * *****************************************************************************
 * ******************************** PROTECTED **********************************
 * *****************************************************************************
 */

template<typename IXT, class OP>
bool
b_ixheap<IXT, OP>::
sift_up(
    const IXT pos)
{
    /* sift up: move element until parent's key is smaller or equal */
    if(pos == 0)
        return false;

    IXT me = pos;
    /* comparator does not need to handle equality, hence do manually */
    while(me > 0 && m_keys[_PARENT(me)] != m_keys[me] &&
        !m_comp(m_keys[_PARENT(me)], m_keys[me]))
    {
        /* parent's key smaller than mine -> swap */
        std::swap(m_keys[_PARENT(me)], m_keys[me]);
        std::swap((*m_positions)[m_vals[_PARENT(me)]],
            (*m_positions)[m_vals[me]]);
        std::swap(m_vals[_PARENT(me)], m_vals[me]);

        me = _PARENT(me);
    }

    return true;
}

/* ************************************************************************** */

template<typename IXT, class OP>
bool
b_ixheap<IXT, OP>::
sift_down(
    const IXT pos)
{
    /* sift down: move element until children's keys are larger or equal */
    if (pos == m_size - 1)
        return false;

    IXT lkey, rkey;
    IXT me = pos;
    bool did_sift = true;
    while(did_sift)
    {
        did_sift = false;

        /* retrieve children's keys */
        IXT cand;
        IXT cand_ix;
        if(std::max(_LCHILD(me), _RCHILD(me)) < m_size)
        {
            lkey = m_keys[_LCHILD(me)];
            rkey = m_keys[_RCHILD(me)];
            cand = m_comp(lkey, rkey) ? lkey : rkey;
            cand_ix = m_comp(lkey, rkey) ? _LCHILD(me) : _RCHILD(me);
        }
        else if(_LCHILD(me) < m_size)
        {
            cand = m_keys[_LCHILD(me)];
            cand_ix = _LCHILD(me);
        }
        else if(_RCHILD(me) < m_size)
        {
            cand = m_keys[_RCHILD(me)];
            cand_ix = _RCHILD(me);
        }
        else
        {
            break;
        }

        /* check if any of the children has a smaller key, then swap */
        if(!m_comp(m_keys[me], cand) && m_keys[me] != cand)
        {
            std::swap(m_keys[cand_ix], m_keys[me]);
            std::swap((*m_positions)[m_vals[cand_ix]],
                (*m_positions)[m_vals[me]]);
            std::swap(m_vals[cand_ix], m_vals[me]);

            me = cand_ix;
            did_sift = true;
        }
    }

    return true;
}

/* ************************************************************************** */

template<typename IXT, class OP>
void
b_ixheap<IXT, OP>::
print()
{
    IXT lvl_size = 1;
    IXT it = 0;
    IXT lvl_it = 0;
    while(it < m_size)
    {
        /* +1 as MATLAB fix */
        std::cout << m_keys[it] << "/" << m_vals[it] << " ";
        ++it;
        ++lvl_it;

        if(lvl_it == lvl_size)
        {
            std::cout << std::endl;
            lvl_size *= 2;
            lvl_it = 0;
        }
    }
    if(lvl_it < lvl_size && lvl_it > 0)
        std::cout << std::endl;
    std::cout << std::endl;
}

NS_DATA_STRUCTURES_END
NS_CULIP_END
