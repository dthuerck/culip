/**
 * Copyright (c) 2018, Daniel Thuerck, TU Darmstadt - GCC. All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-clause license. See the LICENSE file for details.
 */

#ifndef __CULIP_STAGING_HEADER_TRIANGULAR_H_
#define __CULIP_STAGING_HEADER_TRIANGULAR_H_

#include <libs/utils/defines.h>
#include <libs/utils/types.cuh>

NS_CULIP_BEGIN
NS_STAGING_BEGIN

/* forward declarations for pointer */
template<typename T>
class Triangular;

template<typename T>
using Triangular_ptr = std::unique_ptr<Triangular<T>>;

template<typename T>
class Triangular
{
public:
    Triangular(const mat_int_t m, const mat_int_t nnz, const bool lower = true);
    Triangular(const mat_int_t nnz, const mat_int_t * A_i,
        const mat_int_t * A_j, const T * A_k);
    Triangular(const csr_matrix_t<T> * A);
    ~Triangular();

    mat_int_t m() const;
    mat_int_t nnz() const;
    void resize(const mat_int_t new_nnz);

    /* (raw) access to matrix data */
    mat_int_t row_length(const mat_int_t row) const;

    /* read-only access to matrix data */
    const mat_int_t * row_col(const mat_int_t row) const;
    const T * row_val(const mat_int_t row) const;

    /* read-write access to matrix data */
    mat_int_t * row_col_rw(const mat_int_t row);
    T * row_val_rw(const mat_int_t row);

    mat_int_t * raw_row_ptr();
    mat_int_t * raw_col_ptr();
    T * raw_val_ptr();

    csr_matrix_ptr<T> to_csr() const;

    /* higher-level operations on the whole matrix */
    mat_int_t sanalysis(const mat_int_t b_len, const mat_int_t * b_ix) const;
    void sanalysis_import(const mat_int_t b_len, const mat_int_t * b_ix,
        const mat_int_t x_len, const mat_int_t * x_ix) const;
    void sanalysis_export(mat_int_t * x_ix) const;

    void sfsolve(const T * b, T * x) const;
    void dfsolve(const T * b, T * x) const;

    void dbsolve(const T * b, T * x) const;

    /* higher-level operations on submatrices */
    mat_int_t sub_sanalysis(const mat_int_t sub_m, const mat_int_t b_len,
        const mat_int_t * b_ix) const;
    void sub_sanalysis_import(const mat_int_t b_len, const mat_int_t * b_ix,
        const mat_int_t x_len, const mat_int_t * x_ix) const;
    void sub_sanalysis_export(mat_int_t * x_ix) const;

    void sub_sfsolve(const mat_int_t sub_m, const T * b_val, T * x_val) const;
    void sub_dfsolve(const mat_int_t sub_m, const T * b_val, T * x_val) const;

    /* const operations for parallel access, w/o analysis */
    void sfsolve(const mat_int_t b_len, const mat_int_t * b_ix,
        const T * b_val, const mat_int_t x_len, const mat_int_t * x_ix,
        T * x_val) const;
    void sub_sfsolve(const mat_int_t sub_m, const mat_int_t b_len,
        const mat_int_t * b_ix,
        const T * b_val, const mat_int_t x_len, const mat_int_t * x_ix,
        T * x_val) const;

    /* analysis for parallelism */
    mat_int_t level_schedule() const;
    mat_int_t level_size(const mat_int_t level);
    mat_int_t * level_nodes(const mat_int_t level);

    /* conversion */
    Triangular_ptr<T> copy() const;
    Triangular_ptr<T> transpose() const;

    /* debugging */
    void dprint() const;
    void sub_dprint(const mat_int_t sub_m) const;

protected:
    void ijk_compress(const mat_int_t * A_i, const mat_int_t * A_j,
        const T * A_k);

protected:
    bool m_lower;

    mat_int_t m_m;
    mat_int_t m_nnz;
    std::vector<mat_int_t> m_csr_row;
    std::vector<mat_int_t> m_csr_col;
    std::vector<T> m_csr_val;

    /* variables for level scheduling */
    mutable bool m_scheduled;
    mutable mat_int_t m_num_levels;
    mutable std::vector<mat_int_t> m_level_offsets;
    mutable std::vector<mat_int_t> m_levels;

    /* variables for solve operation */
    mutable std::vector<mat_int_t> m_x_ix;

    mutable mat_int_t m_b_len;
    mutable std::vector<mat_int_t> m_b_ix;
};

template<typename T>
using Triangular_ptr = std::unique_ptr<Triangular<T>>;

NS_STAGING_END
NS_CULIP_END

#endif /* __CULIP_STAGING_HEADER_TRIANGULAR_H_ */