/**
 * Copyright (c) 2017, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_LIBS_ALGORITHMS_PERMUTE_BTF_H_
#define __CULIP_LIBS_ALGORITHMS_PERMUTE_BTF_H_

#include <libs/utils/defines.h>

#include <libs/data_structures/b_kvheap.h>

using namespace NS_CULIP::NS_DATA_STRUCTURES;

NS_CULIP_BEGIN
NS_ALGORITHMS_BEGIN

/**
 * Retrieve a lower bound on matrix A's rank by permuting it into
 * a BTF-version such that A(row_order, col_order) is BTF; the return
 * value is the number of rows _NOT_ in the rest (i.e. the lower
 * bound on its rank).
 *
 * The input is matrix A, both as csr and csc (i.e. A^\top) formats.
 */
class BTF
{
public:
    BTF();
    ~BTF();

    mat_int_t permute(const mat_int_t, const mat_int_t n,
        const mat_int_t * sp_A_csc_col, const mat_int_t * sp_A_csc_row,
        const mat_int_t * sp_A_csr_row, const mat_int_t * sp_A_csr_col,
        mat_int_t * row_order, mat_int_t * col_order);

protected:
    bmin_kvheap<mat_int_t, mat_int_t> m_c_pqueue;
    bmin_kvheap<mat_int_t, mat_int_t> m_r_pqueue;
};

NS_ALGORITHMS_END
NS_CULIP_END

#endif /* __CULIP_LIBS_ALGORITHMS_PERMUTE_BTF_H_ */
