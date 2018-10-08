/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIB_ALGORITHMS_MATCHING_CUH_
#define __CULIP_LIB_ALGORITHMS_MATCHING_CUH_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

#include <queue>

NS_CULIP_BEGIN
NS_ALGORITHMS_BEGIN

class UnweightedBipartiteMatching
{
public:
    UnweightedBipartiteMatching(const mat_int_t m, const mat_int_t n,
        const mat_int_t * csr_row, const mat_int_t * csr_col);
    ~UnweightedBipartiteMatching();

    mat_int_t match(
        dense_vector_ptr<mat_int_t>& match_m,
        dense_vector_ptr<mat_int_t>& match_n);

    /* allows starting from an input matching */
    mat_int_t match(
        dense_vector_t<mat_int_t> * match_m,
        dense_vector_t<mat_int_t> * match_n,
        const bool use_previous = false);

    /* use with care */
    void export_labels(dense_vector_t<mat_int_t> * labels);

protected:
    bool stage(mat_int_t& end_r, mat_int_t& end_c);
    void augment(const mat_int_t end_r, const mat_int_t end_c);

    void phase_i();
    void phase_ii();

protected:
    const mat_int_t m_m;
    const mat_int_t m_n;
    const mat_int_t * m_csr_row;
    const mat_int_t * m_csr_col;

    /* save current matching */
    dense_vector_ptr<mat_int_t> m_r_match_c;
    dense_vector_ptr<mat_int_t> m_c_match_r;

    /* path finding */
    std::queue<mat_int_t> m_front;
    dense_vector_ptr<mat_int_t> m_label;

    mat_int_t m_num_matched;

    /* invalid / unlabelled label */
    const mat_int_t m_inv_label = std::numeric_limits<mat_int_t>::max();
};

/* ************************************************************************** */

template<typename T>
class WeightedBipartiteMatching
{
public:
    WeightedBipartiteMatching(const mat_int_t m, const mat_int_t n,
        const mat_int_t * csr_row, const mat_int_t * csr_col);
    ~WeightedBipartiteMatching();

    T match(const T * csr_val, dense_vector_ptr<mat_int_t>&
        match_m, dense_vector_ptr<mat_int_t>& match_n, bool& infeasible);

    void get_scaling(dense_vector_ptr<T>& s_r, dense_vector_ptr<T>& s_c);

protected:
    void dual_init();
    void find_initial_primal_partial_solution();
    bool stage();
    void augment(const mat_int_t end_r, const mat_int_t end_c);
    bool update_dual();
    T compute_objective();

protected:
    const mat_int_t m_m;
    const mat_int_t m_n;
    const mat_int_t m_nnz;
    const mat_int_t * m_csr_row;
    const mat_int_t * m_csr_col;
    std::vector<mat_int_t> m_csr_val;

    /* save current matching and its value */
    dense_vector_ptr<mat_int_t> m_r_match_c;
    dense_vector_ptr<mat_int_t> m_c_match_r;

    mat_int_t m_num_matched;
    mat_int_t m_cur_obj;

    /* save scaling factors */
    dense_vector_ptr<T> m_s_r;
    dense_vector_ptr<T> m_s_c;

    /* dual variables and slack */
    dense_vector_ptr<mat_int_t> m_alpha;
    dense_vector_ptr<mat_int_t> m_beta;

    dense_vector_ptr<mat_int_t> m_slack;
    dense_vector_ptr<mat_int_t> m_nbor;

    /* path finding */
    mat_int_t m_z_ctr = 0;
    std::vector<mat_int_t> m_r_updated;
    std::vector<mat_int_t> m_c_updated;

    dense_vector_ptr<mat_int_t> m_r_label;
    dense_vector_ptr<mat_int_t> m_c_label;

    /* invalid / unlabelled label */
    const mat_int_t m_inv_label = std::numeric_limits<mat_int_t>::max();
};

NS_ALGORITHMS_END
NS_CULIP_END

#endif /* __CULIP_LIB_ALGORITHMS_MATCHING_CUH_ */