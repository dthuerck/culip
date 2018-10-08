/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/algorithms/matching.cuh>
#include <libs/algorithms/matching.impl.cuh>

#include <queue>

NS_CULIP_BEGIN
NS_ALGORITHMS_BEGIN

UnweightedBipartiteMatching::
UnweightedBipartiteMatching(
    const mat_int_t m,
    const mat_int_t n,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col)
: m_m(m),
  m_n(n),
  m_csr_row(csr_row),
  m_csr_col(csr_col),
  m_front(),
  m_r_match_c(make_managed_dense_vector_ptr<mat_int_t>(m, false)),
  m_c_match_r(make_managed_dense_vector_ptr<mat_int_t>(n, false)),
  m_label(make_managed_dense_vector_ptr<mat_int_t>(m, false))
{

}

/* ************************************************************************** */

UnweightedBipartiteMatching::
~UnweightedBipartiteMatching()
{

}

/* ************************************************************************** */

mat_int_t
UnweightedBipartiteMatching::
match(
    dense_vector_ptr<mat_int_t>& match_m,
    dense_vector_ptr<mat_int_t>& match_n)
{
    /* allocate output data */
    match_m = make_managed_dense_vector_ptr<mat_int_t>(m_m, false);
    match_n = make_managed_dense_vector_ptr<mat_int_t>(m_n, false);

    /* call matching on preallocated vectors */
    match(match_m.get(), match_n.get());
}

/* ************************************************************************** */

mat_int_t
UnweightedBipartiteMatching::
match(
    dense_vector_t<mat_int_t> * match_m,
    dense_vector_t<mat_int_t> * match_n,
    const bool use_previous)
{
    if(!use_previous)
    {
        /* start from empty matching */
        std::fill(m_r_match_c->dense_val, m_r_match_c->dense_val + m_m,
            m_inv_label);
        std::fill(m_c_match_r->dense_val, m_c_match_r->dense_val + m_n,
            m_inv_label);

        /* use a greedy initialization */
        phase_i();
    }
    else
    {
        m_num_matched = 0;

        *m_r_match_c = match_m;
        *m_c_match_r = match_n;

        for(mat_int_t i = 0; i < m_m; ++i)
            m_num_matched += ((*m_r_match_c)[i] != m_inv_label);
    }

    /* use augmenting paths for further optimization */
    phase_ii();

    /* copy output data */
    *match_m = m_r_match_c.get();
    *match_n = m_c_match_r.get();

    return m_num_matched;
}

/* ************************************************************************** */

void
UnweightedBipartiteMatching::
export_labels(
    dense_vector_t<mat_int_t> * labels)
{
    std::copy(m_label->dense_val, m_label->dense_val + m_label->m,
        labels->dense_val);
}

/* ************************************************************************** */

bool
UnweightedBipartiteMatching::
stage(
    mat_int_t& end_r,
    mat_int_t& end_c)
{
    /* clear labels */
    std::fill(m_label->dense_val, m_label->dense_val + m_label->m,
        m_inv_label);

    /* clear queue */
    m_front = std::queue<mat_int_t>();

    /**
     * A Path starts at unmatched r, stops at unmatched c, only consider r;
     * once at a c-node, the back-edge is unique
     */

    mat_int_t qu_ptr = 0;
    while(!m_front.empty() || qu_ptr < m_m)
    {
        mat_int_t cur = m_inv_label;

        if(!m_front.empty())
        {
            /* take next row from queue */
            cur = m_front.front();
            m_front.pop();
        }
        else
        {
            /* take next unmarked row */
            while(qu_ptr < m_m && (*m_r_match_c)[qu_ptr] != m_inv_label)
                ++qu_ptr;
            cur = qu_ptr;
        }

        if(cur >= m_m)
            break;

        /* push all unmatched, not-yet visited rows to the front */
        for(mat_int_t j = m_csr_row[cur]; j < m_csr_row[cur + 1]; ++j)
        {
            const mat_int_t inter_c = m_csr_col[j];

            /**
             * only matched columns can provide a way back - but not matched
             * means we found a path
             */
            const mat_int_t succ_r = (*m_c_match_r)[inter_c];
            if(succ_r == m_inv_label)
            {
                end_r = cur;
                end_c = inter_c;

                return true;
            }

            /* skip labelled succ r nodes */
            if((*m_label)[succ_r] != m_inv_label)
                continue;

            /* set label for succ r */
            (*m_label)[succ_r] = cur;
            m_front.push(succ_r);
        }
    }

    return false;
}

/* ************************************************************************** */

void
UnweightedBipartiteMatching::
augment(
    const mat_int_t end_r,
    const mat_int_t end_c)
{
    mat_int_t node_r = end_r;
    mat_int_t node_c = end_c;

    while(node_r != m_inv_label)
    {
        const mat_int_t pred_c = (*m_r_match_c)[node_r];

        /* remove prev matching of last c */
        if((*m_r_match_c)[node_r] != m_inv_label)
            (*m_c_match_r)[(*m_r_match_c)[node_r]] = m_inv_label;

        /* match connected c to this r */
        (*m_r_match_c)[node_r] = node_c;
        (*m_c_match_r)[node_c] = node_r;

        node_c = pred_c;
        node_r = (*m_label)[node_r];
    }

    /* every augmenting path increases the number of matched node by one */
    ++m_num_matched;
}

/* ************************************************************************** */

void
UnweightedBipartiteMatching::
phase_i()
{
    /* get number of nonzeros and max marker */
    const mat_int_t max_T = std::numeric_limits<mat_int_t>::max();
    const mat_int_t nnz = m_csr_row[m_m];

    /* determine column degrees */
    dense_vector_ptr<mat_int_t> deg_c =
        make_managed_dense_vector_ptr<mat_int_t>(m_n, false);
    std::fill(deg_c->dense_val, deg_c->dense_val + deg_c->m, 0);

    for(mat_int_t i = 0; i < nnz; ++i)
        ++(*deg_c)[m_csr_col[i]];

    /* Phase I: greedily match edges for rows */
    m_num_matched = 0;
    for(mat_int_t r = 0; r < m_m; ++r)
    {
        /* use lowest-degree unmatched admissible c counterpart */
        const mat_int_t adj_offset = m_csr_row[r];
        const mat_int_t adj_size = m_csr_row[r + 1] - adj_offset;

        mat_int_t max_degree = max_T;
        mat_int_t adj_c = -1;
        for(mat_int_t cix = 0; cix < adj_size; ++cix)
        {
            const mat_int_t c = m_csr_col[adj_offset + cix];

            if((*m_c_match_r)[c] == m_inv_label && (*deg_c)[c] < max_degree)
            {
                max_degree = (*deg_c)[c];
                adj_c = c;
            }
        }

        /* found unmatched c? great, let's match! */
        if(adj_c != m_inv_label)
        {
            (*m_r_match_c)[r] = adj_c;
            (*m_c_match_r)[adj_c] = r;

            ++m_num_matched;
        }
    }
}

/* ************************************************************************** */

void
UnweightedBipartiteMatching::
phase_ii()
{
    mat_int_t end_r, end_c;

    while(stage(end_r, end_c)) { augment(end_r, end_c); }
}

/* ************************************************************************** */

template class WeightedBipartiteMatching<mat_int_t>;
template class WeightedBipartiteMatching<float>;
template class WeightedBipartiteMatching<double>;

NS_ALGORITHMS_END
NS_CULIP_END