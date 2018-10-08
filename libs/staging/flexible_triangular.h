/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_STAGING_FLEXIBLE_TRIANGULAR_H_
#define __CULIP_STAGING_FLEXIBLE_TRIANGULAR_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

#include <libs/staging/triangular.h>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

/* for convenience, only supports lower triangular */
template<typename T>
class FlexibleTriangular
{
public:
    FlexibleTriangular(const mat_int_t m);
    FlexibleTriangular(const csr_matrix_t<T> * A);
    FlexibleTriangular(const Triangular<T> * L);
    ~FlexibleTriangular();

    mat_int_t m() const;
    mat_int_t nnz() const;

    /* row access to matrix data via iterators */
    mat_int_t row(const mat_int_t row, mat_int_t *& ix, T *& val) const;
    mat_int_t col(const mat_int_t col, mat_int_t *& ix) const;

    /* modifications */
    void set_row(const mat_int_t row, const mat_int_t nz_len,
        const mat_int_t * ix, const T * val);
    void pivot(const mat_int_t r1, const mat_int_t r2);

    /* sanitize matrix - all elements should be in full order */
    void order();

    /* higher-level operations on submatrices */
    mat_int_t sub_sanalysis(const mat_int_t sub_m, const mat_int_t b_len,
        const mat_int_t * b_ix) const;
    void sub_sanalysis_import(const mat_int_t b_len, const mat_int_t * b_ix,
        const mat_int_t x_len, const mat_int_t * x_ix) const;
    void sub_sanalysis_export(mat_int_t * x_ix) const;

    void sub_sfsolve(const mat_int_t sub_m, const T * b_val, T * x_val) const;
    void sub_dfsolve(const mat_int_t sub_m, const T * b_val, T * x_val) const;

    void dfsolve(const T * b_val, T * x_val) const;
    void dbsolve(const T * b_val, T * x_val) const;

    /* export to fixed structure */
    Triangular_ptr<T> to_fixed();

protected:
    void init(const mat_int_t m, const mat_int_t nnz, const mat_int_t * csr_row,
        const mat_int_t * csr_col, const T * csr_val);

protected:
    mat_int_t m_m;

    /* storage matrix in flexible layout with padding for pivots */
    mat_int_t m_nnz;

    std::vector<std::vector<mat_int_t>*> m_row_ix;
    std::vector<std::vector<mat_int_t>*> m_col_ix;
    std::vector<std::vector<T>*> m_row_val;

    /* data for sparse solve */
    mutable mat_int_t m_b_len;
    mutable std::vector<mat_int_t> m_b_ix;
    mutable std::vector<mat_int_t> m_x_ix;
};

template<typename T>
using FlexibleTriangular_ptr = std::unique_ptr<FlexibleTriangular<T>>;

NS_STAGING_END
NS_CULIP_END

#endif /* __CULIP_STAGING_FLEXIBLE_TRIANGULAR_H_ */