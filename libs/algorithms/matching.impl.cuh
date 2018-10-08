/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/algorithms/matching.cuh>

#include <limits>

NS_CULIP_BEGIN
NS_ALGORITHMS_BEGIN

template<typename T>
WeightedBipartiteMatching<T>::
WeightedBipartiteMatching(
    const mat_int_t m,
    const mat_int_t n,
    const mat_int_t * csr_row,
    const mat_int_t * csr_col)
: m_m(m),
  m_n(n),
  m_nnz(csr_row[m]),
  m_csr_row(csr_row),
  m_csr_col(csr_col),
  m_r_match_c(make_managed_dense_vector_ptr<mat_int_t>(m, false)),
  m_c_match_r(make_managed_dense_vector_ptr<mat_int_t>(n, false)),
  m_s_r(make_managed_dense_vector_ptr<T>(m, false)),
  m_s_c(make_managed_dense_vector_ptr<T>(n, false)),
  m_alpha(make_managed_dense_vector_ptr<mat_int_t>(m, false)),
  m_beta(make_managed_dense_vector_ptr<mat_int_t>(n, false)),
  m_slack(make_managed_dense_vector_ptr<mat_int_t>(n, false)),
  m_nbor(make_managed_dense_vector_ptr<mat_int_t>(n, false)),
  m_r_updated(),
  m_c_updated(),
  m_r_label(make_managed_dense_vector_ptr<mat_int_t>(m, false)),
  m_c_label(make_managed_dense_vector_ptr<mat_int_t>(n, false))
{

}

/* ************************************************************************** */

template<typename T>
WeightedBipartiteMatching<T>::
~WeightedBipartiteMatching()
{

}

/* ************************************************************************** */

template<typename T>
T
WeightedBipartiteMatching<T>::
match(
    const T * csr_val,
    dense_vector_ptr<mat_int_t>& match_m,
    dense_vector_ptr<mat_int_t>& match_n,
    bool& infeasible)
{
    infeasible = false;

    /* transform edge weights to inverted integers for max matching */
    m_csr_val.resize(m_nnz);

    /* find column maxima */
    dense_vector_ptr<T> col_max = make_managed_dense_vector_ptr<T>(m_n, false);
    std::fill(col_max->dense_val, col_max->dense_val + m_n,
        0);

    for(mat_int_t i = 0; i < m_nnz; ++i)
    {
        (*col_max)[m_csr_col[i]] = std::max((*col_max)[m_csr_col[i]],
            std::abs(csr_val[i]));
    }

    /* shift & invert values, find global max/min */
    T gbl_max = 0;
    T gbl_min = std::numeric_limits<T>::max();
    for(mat_int_t i = 0; i < m_nnz; ++i)
    {
        const T i_val = std::log2((*col_max)[m_csr_col[i]]) -
            std::log2(std::abs(csr_val[i]));

        gbl_max = std::max(gbl_max, i_val);
        gbl_min = std::min(gbl_min, i_val);
    }

    const T new_gbl_max = (T) std::numeric_limits<mat_int_t>::max() /
        (m_nnz + 1);
    const T new_gbl_min = 1.0;

    for(mat_int_t i = 0; i < m_nnz; ++i)
    {
        const T i_val = std::log2((*col_max)[m_csr_col[i]]) -
            std::log2(std::abs(csr_val[i]));

        m_csr_val[i] = std::round(
            new_gbl_min + ((i_val - gbl_min) / (gbl_max - gbl_min)) *
            (new_gbl_max - new_gbl_min));
    }

    /* initialize an empty matching */
    std::fill(m_r_match_c->dense_val, m_r_match_c->dense_val + m_m,
        m_inv_label);
    std::fill(m_c_match_r->dense_val, m_c_match_r->dense_val + m_n,
        m_inv_label);

    m_num_matched = 0;
    m_cur_obj = 0;

    /* initialize dual variables and slacks */
    dual_init();

    /* find heuristic primal solution */
    find_initial_primal_partial_solution();

    /* complete solution using augmented paths */
    while(m_num_matched < std::min(m_m, m_n))
    {
        if(!stage())
        {
            infeasible = true;
            return 0;
        }
    }

    /* compute scaling from duals (inverse transformation) */
    const T m_2 = new_gbl_max - new_gbl_min;
    const T m_1 = gbl_max - gbl_min;

    dense_vector_ptr<T> real_u = make_managed_dense_vector_ptr<T>(m_m, false);
    dense_vector_ptr<T> real_v = make_managed_dense_vector_ptr<T>(m_n, false);

    for(mat_int_t i = 0; i < m_m; ++i)
    {
        const T i_dual = (*m_alpha)[i];
        const T transform_dual = (m_1 / m_2) * (i_dual - 0.5 * new_gbl_min
            + 0.5 * (m_2 / m_1) * gbl_min);

        (*real_u)[i] = transform_dual;
        (*m_s_r)[i] = std::pow(2, - transform_dual);
    }

    for(mat_int_t j = 0; j < m_n; ++j)
    {
        const T j_dual = (*m_beta)[j];
        const T transform_dual = (m_1 / m_2) * (j_dual - 0.5 * new_gbl_min
            + 0.5 * (m_2 / m_1) * gbl_min);

        (*real_v)[j] = transform_dual;
        (*m_s_c)[j] = std::pow(2, std::log2((*col_max)[j])
            - transform_dual);
    }

    /* export solution */
    match_m = make_managed_dense_vector_ptr<mat_int_t>(m_m, false);
    match_n = make_managed_dense_vector_ptr<mat_int_t>(m_n, false);

    std::copy(m_r_match_c->dense_val, m_r_match_c->dense_val + m_m,
        match_m->dense_val);
    std::copy(m_c_match_r->dense_val, m_c_match_r->dense_val + m_n,
        match_n->dense_val);

    return compute_objective();
}

/* ************************************************************************** */

template<typename T>
void
WeightedBipartiteMatching<T>::
get_scaling(
    dense_vector_ptr<T>& s_r,
    dense_vector_ptr<T>& s_c)
{
    s_r = make_managed_dense_vector_ptr<T>(m_m, false);
    s_c = make_managed_dense_vector_ptr<T>(m_n, false);

    std::copy(m_s_r->dense_val, m_s_r->dense_val + m_s_r->m,
        s_r->dense_val);
    std::copy(m_s_c->dense_val, m_s_c->dense_val + m_s_c->m,
        s_c->dense_val);
}

/* ************************************************************************** */

template<typename T>
void
WeightedBipartiteMatching<T>::
dual_init()
{
    /* set beta_v to min_u c_uv */
    std::fill(m_beta->dense_val, m_beta->dense_val + m_n,
        std::numeric_limits<mat_int_t>::max());

    for(mat_int_t i = 0; i < m_m; ++i)
        for(mat_int_t j = m_csr_row[i]; j < m_csr_row[i + 1]; ++j)
            (*m_beta)[m_csr_col[j]] = std::min(m_csr_val[j],
                (*m_beta)[m_csr_col[j]]);

    /* set all alphas to min c_ij - beta_j */
    std::fill(m_alpha->dense_val, m_alpha->dense_val + m_m,
        std::numeric_limits<mat_int_t>::max());

    for(mat_int_t i = 0; i < m_m; ++i)
        for(mat_int_t j = m_csr_row[i]; j < m_csr_row[i + 1]; ++j)
            (*m_alpha)[i] = std::min((*m_alpha)[i], (m_csr_val[j] -
                (*m_beta)[m_csr_col[j]]));
}

/* ************************************************************************** */

template<typename T>
void
WeightedBipartiteMatching<T>::
find_initial_primal_partial_solution()
{
    std::fill(m_r_match_c->dense_val, m_r_match_c->dense_val + m_r_match_c->m,
        m_inv_label);
    std::fill(m_c_match_r->dense_val, m_c_match_r->dense_val + m_c_match_r->m,
        m_inv_label);

    m_num_matched = 0;

    /* greedily assign columns to rows with a 1-sized pushback option */
    bool found, replaced;
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        if((*m_r_match_c)[i] != m_inv_label)
            continue;

        found = false;
        for(mat_int_t j = m_csr_row[i]; j < m_csr_row[i + 1] && !found; ++j)
        {
            const mat_int_t col = m_csr_col[j];
            const mat_int_t slack = m_csr_val[j] - (*m_alpha)[i] -
                (*m_beta)[col];

            if(slack == 0)
            {
                const mat_int_t conflict_i = (*m_c_match_r)[col];

                if(conflict_i == m_inv_label)
                {
                    /* found an admissible, unmatched column -> match */
                    (*m_r_match_c)[i] = col;
                    (*m_c_match_r)[col] = i;

                    ++m_num_matched;

                    found = true;
                }
                else
                {
                    /* try to find another column for row conflict_i */
                    replaced = false;

                    for(mat_int_t cj = m_csr_row[conflict_i];
                        cj < m_csr_row[conflict_i + 1] && !replaced; ++cj)
                    {
                        const mat_int_t ccol = m_csr_col[cj];
                        const T cslack = m_csr_val[cj] -
                            (*m_alpha)[conflict_i] -
                            (*m_beta)[ccol];

                        if(cslack == 0 && (*m_c_match_r)[ccol] ==
                            m_inv_label)
                        {
                            /* assign this column to conflicting row */
                            (*m_r_match_c)[conflict_i] = ccol;
                            (*m_c_match_r)[ccol] = conflict_i;

                            (*m_r_match_c)[i] = col;
                            (*m_c_match_r)[col] = i;

                            ++m_num_matched;

                            replaced = true;
                            found = true;
                        }
                    }
                }
            }
        }
    }
}

/* ************************************************************************** */

template<typename T>
bool
WeightedBipartiteMatching<T>::
stage()
{
    /* reset slacks and labels */
    std::fill(m_slack->dense_val, m_slack->dense_val + m_slack->m,
        std::numeric_limits<mat_int_t>::max());
    std::fill(m_nbor->dense_val, m_nbor->dense_val + m_nbor->m,
        m_inv_label);
    std::fill(m_r_label->dense_val, m_r_label->dense_val + m_r_label->m,
        m_inv_label);
    std::fill(m_c_label->dense_val, m_c_label->dense_val + m_c_label->m,
        m_inv_label);

    /* find augmenting path */
    m_z_ctr = 0;
    m_r_updated.clear();
    m_c_updated.clear();

    /* initially add first unmatched column to front */
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        if((*m_r_match_c)[i] == m_inv_label)
        {
            (*m_r_label)[i] = i;
            m_r_updated.push_back(i);

            break;
        }
    }

    /* execute stage: update duals until an augmenting path is found */
    while(true)
    {
        /* scan operation - update slacks for newly discovered rows */
        while(m_z_ctr < m_r_updated.size())
        {
            const mat_int_t cur_i = m_r_updated[m_z_ctr];

            /* update slacks */
            for(mat_int_t j = m_csr_row[cur_i]; j < m_csr_row[cur_i + 1]; ++j)
            {
                const mat_int_t col = m_csr_col[j];
                const mat_int_t slack = m_csr_val[j] - (*m_alpha)[cur_i] -
                    (*m_beta)[col];

                if(slack < (*m_slack)[col])
                {
                    /* make sure each col is only once in the queue */
                    if((*m_slack)[col] == std::numeric_limits<mat_int_t>::max())
                        m_c_updated.push_back(col);
                    (*m_c_label)[col] = 1;

                    (*m_slack)[col] = slack;
                    (*m_nbor)[col] = cur_i;
                }
            }

            ++m_z_ctr;
        }

        /* search for a path opened by the new duals */
        for(const mat_int_t cur_c : m_c_updated)
        {
            if((*m_c_label)[cur_c] != m_inv_label && (*m_slack)[cur_c] == 0)
            {
                /* check new edge if it connects to a path */
                const mat_int_t conn_r = (*m_c_match_r)[cur_c];

                if(conn_r == m_inv_label)
                {
                    augment((*m_nbor)[cur_c], cur_c);

                    return true;
                }
                else
                {
                    /* mark as predecessor for path finding purposes */
                    (*m_r_label)[conn_r] = (*m_nbor)[cur_c];

                    /* otherwise: follow link to next row */
                    m_r_updated.push_back(conn_r);
                }

                /* reset 'updated' label */
                (*m_c_label)[cur_c] = m_inv_label;
            }
        }

        /* no path found: update duals */
        if(m_z_ctr == m_r_updated.size())
        {
            if(!update_dual())
                return false;
        }
    }
}

/* ************************************************************************** */

template<typename T>
void
WeightedBipartiteMatching<T>::
augment(
    const mat_int_t end_r,
    const mat_int_t end_c)
{
    mat_int_t prev_r = end_r;
    mat_int_t prev_c = end_c;

    mat_int_t node_r, node_c;

    do {
        node_r = prev_r;
        node_c = prev_c;

        prev_r = (*m_r_label)[node_r];
        prev_c = (*m_r_match_c)[node_r];

        (*m_r_match_c)[node_r] = node_c;
        (*m_c_match_r)[node_c] = node_r;

    } while((*m_r_label)[node_r] != node_r);

    ++m_num_matched;
}

/* ************************************************************************** */

template<typename T>
bool
WeightedBipartiteMatching<T>::
update_dual()
{
    mat_int_t theta1 = std::numeric_limits<mat_int_t>::max();

    for(const mat_int_t j : m_c_updated)
        if((*m_slack)[j] > 0)
            theta1 = std::min(theta1, (*m_slack)[j]);

    /* non nonzero slack exists -> no feasible assignment */
    if(theta1 == std::numeric_limits<mat_int_t>::max())
        return false;

    /* otherwise: update duals and slacks */
    for(const mat_int_t i : m_r_updated)
        (*m_alpha)[i] += theta1;

    for(const mat_int_t j : m_c_updated)
    {
        if((*m_slack)[j] == 0)
            (*m_beta)[j] -= theta1;
        else
            (*m_slack)[j] -= theta1;
    }

    return true;
}

/* ************************************************************************** */

template<typename T>
T
WeightedBipartiteMatching<T>::
compute_objective()
{
    m_cur_obj = 0;

    bool found;
    for(mat_int_t i = 0; i < m_m; ++i)
    {
        found = false;
        for(mat_int_t j = m_csr_row[i]; j < m_csr_row[i + 1] && !found; ++j)
        {
            if((*m_r_match_c)[i] == m_csr_col[j])
            {
                m_cur_obj += m_csr_val[j];
                found = true;
            }
        }
    }

    mat_int_t test_dual = 0;
    for(mat_int_t i = 0; i < m_m; ++i)
        test_dual += (*m_alpha)[i];
    for(mat_int_t i = 0; i < m_n; ++i)
        test_dual += (*m_beta)[i];

    return m_cur_obj;
}

NS_ALGORITHMS_END
NS_CULIP_END