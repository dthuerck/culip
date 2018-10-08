/**
 *  Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 *  This software may be modified and distributed under the terms
 *  of the BSD 3-clause license. See the LICENSE file for details.
 */

#include <libs/algorithms/sprank.cuh>
#include <libs/algorithms/matching.cuh>

NS_CULIP_BEGIN
NS_ALGORITHMS_BEGIN

mat_int_t
sprank(
    const mat_int_t m,
    const mat_int_t n,
    const mat_int_t * sp_A_csr_row,
    const mat_int_t * sp_A_csr_col,
    mat_int_t * row_perm,
    mat_int_t * col_perm)
{
    UnweightedBipartiteMatching umatch(m, n, sp_A_csr_row, sp_A_csr_col);

    /* map to unweighted matching with full graph admissible */
    dense_vector_ptr<mat_int_t> rperm;
    dense_vector_ptr<mat_int_t> cperm;
    const mat_int_t obj = umatch.match(rperm, cperm);

    if(row_perm != nullptr)
        std::copy(rperm->dense_val, rperm->dense_val + m, row_perm);
    if(col_perm != nullptr)
        std::copy(cperm->dense_val, cperm->dense_val + n, col_perm);

    return obj;
}

NS_ALGORITHMS_END
NS_CULIP_END
