/**
 * Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/data_structures/b_kvheap.h>

#include <limits>

NS_CULIP_BEGIN
NS_DATA_STRUCTURES_BEGIN

/**
 * *****************************************************************************
 * ******************************** B_KVHEAP ***********************************
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

template<typename K, typename V, class OP>
b_kvheap<K, V, OP>::
b_kvheap()
: m_fwd_keys(),
  m_rev_keys(),
  m_positions(),
  m_vals(),
  m_free_internal_ids(),
  m_next_internal_id(0),
  m_comp()
{
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
b_kvheap<K, V, OP>::
b_kvheap(
    const K * keys,
    const V * vals,
    const mat_int_t num_entries)
: b_kvheap()
{
    for(mat_int_t i = 0; i < num_entries; ++i)
        push(keys[i], vals[i]);
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
b_kvheap<K, V, OP>::
~b_kvheap()
{
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
push(
    const K key,
    const V val)
{
    /* check if key is already there (and if so, rather perform an update) */
    if(m_fwd_keys.count(key) > 0)
    {
        update(key, val);
        return;
    }

    /* retrieve internal ID for key */
    mat_int_t internal_id = -1;

    /* try to recycle one before creating a new one*/
    if(!m_free_internal_ids.empty())
    {
        internal_id = m_free_internal_ids.back();
        m_free_internal_ids.pop_back();

        m_rev_keys[internal_id] = key;
        m_vals[internal_id] = val;
        m_positions[internal_id] = m_data.size();
    }
    else
    {
        internal_id = m_next_internal_id;
        ++m_next_internal_id;

        m_rev_keys.push_back(key);
        m_vals.push_back(val);
        m_positions.push_back(m_data.size());
    }

    /* add key to dictionary */
    m_fwd_keys[key] = internal_id;

    /* append value to back */
    m_data.push_back(internal_id);

    /* sift element up as far as possible */
    sift_up(internal_id);
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
update(
    const K key,
    const V new_val)
{
    /* check if key exists */
    if(m_fwd_keys.count(key) == 0)
        return;

    /* find element and save old value */
    const K internal_id = m_fwd_keys[key];
    const V old_val = m_vals[internal_id];

    /* exit if update unnecessary */
    if(new_val == old_val)
        return;

    /* save new value */
    m_vals[internal_id] = new_val;

    /* restore heap property - delta == 0 is caught above */
    if(m_comp(new_val, old_val))
        sift_up(internal_id);
    else
        sift_down(internal_id);
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
remove(
    const K key)
{
    /* check if key exists */
    if(m_fwd_keys.count(key) == 0)
        return;

    /* find element and save old value */
    const K internal_id = m_fwd_keys[key];
    const mat_int_t old_pos = m_positions[internal_id];
    const V old_val = m_vals[internal_id];

    /* remove deleted element from key lookup */
    m_fwd_keys.erase(key);

    /* repurpose internal id */
    m_free_internal_ids.push_back(internal_id);

    /* pull in last element */
    const V delta = m_vals[m_data.back()] - old_val;

    const mat_int_t overwrite_id = m_data.back();
    m_positions[overwrite_id] = old_pos;
    m_data[old_pos] = overwrite_id;
    m_data.pop_back();

    /* restore heap property */
    if(delta != 0)
    {
        if(m_comp(delta, 0))
            sift_up(overwrite_id);
        else
            sift_down(overwrite_id);
    }
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
top(
    K& key,
    V& val)
{
    if(empty())
        return;

    key = m_rev_keys[m_data[0]];
    val = m_vals[m_data[0]];
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
pop()
{
    if(empty())
        return;

    /* set last element to root (maintain almost complete tree) */
    remove(m_rev_keys[m_data[0]]);
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
bool
b_kvheap<K, V, OP>::
empty()
{
    return m_data.empty();
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
V
b_kvheap<K, V, OP>::
value(
    const K key)
{
    if(m_fwd_keys.count(key) == 0)
        return (V) 0;

    const mat_int_t internal_id = m_fwd_keys[key];
    return m_vals[internal_id];
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
mat_int_t
b_kvheap<K, V, OP>::
size()
{
    return m_data.size();
}

/**
 * *****************************************************************************
 * ******************************** PROTECTED **********************************
 * *****************************************************************************
 */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
sift_up(
    const mat_int_t internal_id)
{
    const mat_int_t start_pos = m_positions[internal_id];

    /* sift up: move element until parent's value is smaller or equal */
    if(start_pos == 0)
        return;

    mat_int_t me = start_pos;

    /* comparator does not need to handle equality, hence do manually */
    while(me > 0 && m_vals[m_data[_PARENT(me)]] != m_vals[m_data[me]] &&
        !m_comp(m_vals[m_data[_PARENT(me)]], m_vals[m_data[me]]))
    {
        /* parent's key smaller than mine -> swap */
        std::swap(m_positions[m_data[_PARENT(me)]], m_positions[m_data[me]]);
        std::swap(m_data[_PARENT(me)], m_data[me]);

        me = _PARENT(me);
    }
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
sift_down(
    const mat_int_t internal_id)
{
    const mat_int_t start_pos = m_positions[internal_id];

    /* sift down: move element until children's keys are larger or equal */
    if (start_pos >= m_data.size() - 1)
        return;

    V lval, rval;
    mat_int_t me = start_pos;
    bool did_sift = true;
    while(did_sift)
    {
        did_sift = false;

        /* retrieve children's keys */
        V cand;
        mat_int_t cand_ix;
        if(std::max(_LCHILD(me), _RCHILD(me)) < m_data.size())
        {
            lval = m_vals[m_data[_LCHILD(me)]];
            rval = m_vals[m_data[_RCHILD(me)]];
            cand = m_comp(lval, rval) ? lval : rval;
            cand_ix = m_comp(lval, rval) ? _LCHILD(me) : _RCHILD(me);
        }
        else if(_LCHILD(me) < m_data.size())
        {
            cand = m_vals[m_data[_LCHILD(me)]];
            cand_ix = _LCHILD(me);
        }
        else if(_RCHILD(me) < m_data.size())
        {
            cand = m_vals[m_data[_RCHILD(me)]];
            cand_ix = _RCHILD(me);
        }
        else
        {
            break;
        }

        /* check if any of the children has a smaller key, then swap */
        if(!m_comp(m_vals[m_data[me]], cand) && m_vals[m_data[me]] != cand)
        {
            std::swap(m_positions[m_data[cand_ix]], m_positions[m_data[me]]);
            std::swap(m_data[cand_ix], m_data[me]);

            me = cand_ix;
            did_sift = true;
        }
    }
}

/* ************************************************************************** */

template<typename K, typename V, class OP>
void
b_kvheap<K, V, OP>::
print()
{
    mat_int_t lvl_size = 1;
    mat_int_t it = 0;
    mat_int_t lvl_it = 0;
    while(it < m_data.size())
    {
        const mat_int_t it_internal_id = m_data[it];

        /* +1 as MATLAB fix */
        std::cout << "(" << m_rev_keys[it_internal_id] << " [" << it_internal_id
            << "]" << " / " << m_vals[it_internal_id] << ") ";
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
