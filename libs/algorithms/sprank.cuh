/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_ALGORITHMS_SPRANK_H_
#define __CULIP_LIBS_ALGORITHMS_SPRANK_H_

#include <libs/utils/defines.h>

NS_CULIP_BEGIN
NS_ALGORITHMS_BEGIN

/**
 * Compute the structural rank of a given matrix in CSR format, using
 * a maximum bipartite matching algorithm.
 *
 * Outputs a row permutation such that the permuted matrix has a NZ diagonal.
 * if col_perm == nullptr, no output is generated -- otherwise, it needs to be
 * a vector of length m.
 */
mat_int_t
sprank(
    const mat_int_t m,
    const mat_int_t n,
    const mat_int_t * sp_A_csr_row,
    const mat_int_t * sp_A_csr_col,
    mat_int_t * row_perm = nullptr,
    mat_int_t * col_perm = nullptr);

NS_ALGORITHMS_END
NS_CULIP_END

#endif /* __CULIP_LIBS_ALGORITHMS_SPRANK_H_ */
